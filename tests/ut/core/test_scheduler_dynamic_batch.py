# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import torch
from vllm.config import (CacheConfig, DeviceConfig, KVTransferConfig,
                         ModelConfig, SchedulerConfig, SpeculativeConfig,
                         VllmConfig)
from vllm.multimodal.inputs import (MultiModalFeatureSpec,
                                    MultiModalKwargsItem, PlaceholderRange)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import (get_request_block_hasher,
                                         init_none_hash)
from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec)
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from tests.ut.base import TestBase
from vllm_ascend.core.scheduler_dynamic_batch import SchedulerDynamicBatch

EOS_TOKEN_ID = 50256
MODEL = "Qwen3-0.6B"
ENABLE_PREFIX_CACHING = None
PROMPT_LOGPROBS = None
ENABLE_CHUNKED_PREFILL = False
MAX_NUM_BATCHED_TOKENS = 10000
LONG_PREFILL_TOKEN_THRESHOLD = 0
NUM_SPECULATIVE_TOKENS = None
MAX_NUM_SEQS = 16


class DummyProcessorInfo:

    supported_mm_limits = {"image": 1}
    allowed_mm_limits = {"image": 1}

    @staticmethod
    def get_mm_max_tokens_per_item(seq_len, mm_counts):
        del seq_len, mm_counts
        return {"image": 10}


class DummyProcessor:

    def __init__(self):
        self.info = DummyProcessorInfo()


class DummyMMRegistry:

    @staticmethod
    def supports_multimodal_inputs(model_config):
        del model_config
        return True

    @staticmethod
    def processor_only_cache_from_config(vllm_config):
        del vllm_config
        return None

    @staticmethod
    def create_processor(model_config, cache=None):
        del model_config, cache
        return DummyProcessor()

    @staticmethod
    def get_dummy_mm_inputs(model_config, mm_counts, processor):
        del model_config, mm_counts, processor
        return {"mm_placeholders": {"image": []}}


def create_requests(
    num_requests: int,
    num_tokens: int = 10,
    mm_positions: Optional[list[PlaceholderRange]] = None,
    max_tokens: int = 16,
    stop_token_ids: Optional[list[int]] = None,
    block_size: int = 3,
    hash_fn=sha256,
):
    init_none_hash(hash_fn)
    prompt_logprobs = PROMPT_LOGPROBS
    sampling_params = SamplingParams(ignore_eos=False,
                                     max_tokens=max_tokens,
                                     stop_token_ids=stop_token_ids,
                                     prompt_logprobs=prompt_logprobs)
    sampling_params._eos_token_id = EOS_TOKEN_ID
    requests = []
    for i in range(num_requests):
        mm_features = []
        if mm_positions is not None:
            mm_position = mm_positions[i]
            for j, position in enumerate(mm_position):
                identifier = f"hash{i}_{j}"
                mm_feature = MultiModalFeatureSpec(
                    data=MultiModalKwargsItem.dummy(1),
                    mm_position=position,
                    identifier=identifier,
                    modality="image")
                mm_features.append(mm_feature)
        request = Request(request_id=f"{i}",
                          prompt_token_ids=[i] * num_tokens,
                          sampling_params=sampling_params,
                          pooling_params=None,
                          mm_features=mm_features if mm_features else None,
                          block_hasher=get_request_block_hasher(
                              block_size, hash_fn))
        requests.append(request)
    return requests


def make_output(scheduler):
    req_ids = [req.request_id for req in scheduler.running]
    req_id_to_index = {
        req.request_id: i
        for i, req in enumerate(scheduler.running)
    }
    sampled_token_ids = [[1000]] * len(scheduler.running)

    logprobs = None

    modelrunner_output = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index=req_id_to_index,
        sampled_token_ids=sampled_token_ids,
        logprobs=logprobs,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    return modelrunner_output


class TestSchedulerDynamicBatch(TestBase):

    @patch("vllm.config.ModelConfig.__post_init__", MagicMock())
    @patch("vllm.config.VllmConfig.__post_init__", MagicMock())
    def create_scheduler(self):
        use_kv_connector = False
        block_size = 16

        scheduler_config = SchedulerConfig(
            max_num_seqs=16,
            is_encoder_decoder=False,
            max_model_len=MAX_NUM_BATCHED_TOKENS,
            long_prefill_token_threshold=LONG_PREFILL_TOKEN_THRESHOLD,
            disable_chunked_mm_input=False,
            enable_chunked_prefill=True,
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        )

        scheduler_config.max_num_encoder_input_tokens = 10000
        scheduler_config.encoder_cache_size = 10000
        scheduler_config.chunked_prefill_enabled = True
        scheduler_config.SLO_limits_for_dynamic_batch = 0

        fake_weight_path = os.path.join(os.path.dirname(__file__), "..", "fake_weight")
        model_config = ModelConfig(
            model=fake_weight_path,
            tokenizer=fake_weight_path,
            skip_tokenizer_init=True,
            dtype="float16",
            seed=42,
            max_model_len=MAX_NUM_BATCHED_TOKENS,
        )
        model_config.pooler_config = MagicMock()
        model_config.multimodal_config = MagicMock()
        model_config.__dict__["is_encoder_decoder"] = False
        model_config.hf_text_config = MagicMock()
        model_config.hf_text_config.is_encoder_decoder = False
        # Cache config, optionally force APC
        kwargs_cache: Dict[str,
                           Any] = ({} if ENABLE_PREFIX_CACHING is None else {
                               'enable_prefix_caching':
                               ENABLE_PREFIX_CACHING
                           })
        cache_config = CacheConfig(
            block_size=block_size,
            gpu_memory_utilization=0.9,
            swap_space=0,
            cache_dtype="auto",
            **kwargs_cache,
        )

        kv_transfer_config = KVTransferConfig(
            kv_connector="SharedStorageConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"shared_storage_path": "local_storage"},
        ) if use_kv_connector else None

        speculative_config: Optional[SpeculativeConfig] = None
        if NUM_SPECULATIVE_TOKENS is not None:
            speculative_config = SpeculativeConfig(
                model="ngram", num_speculative_tokens=NUM_SPECULATIVE_TOKENS)

        vllm_config = VllmConfig(
            scheduler_config=scheduler_config,
            model_config=model_config,
            cache_config=cache_config,
            kv_transfer_config=kv_transfer_config,
            speculative_config=speculative_config,
            device_config=DeviceConfig("cpu"),
        )

        kv_cache_config = KVCacheConfig(
            num_blocks=10000,  # A large number of blocks to hold all requests
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(['layer'],
                                 FullAttentionSpec(block_size=block_size,
                                                   num_kv_heads=1,
                                                   head_size=1,
                                                   dtype=torch.float32))
            ],
        )
        kv_cache_config.hash_block_size = block_size
        cache_config.num_gpu_blocks = 10000

        scheduler = SchedulerDynamicBatch(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            block_size=block_size,
            log_stats=True,
            mm_registry=DummyMMRegistry(),
            structured_output_manager=MagicMock(spec=StructuredOutputManager),
        )

        should_advance = MagicMock()
        should_advance.return_value = False
        scheduler.structured_output_manager.should_advance = should_advance

        return scheduler

    def test_add_requests(self):
        scheduler = self.create_scheduler()
        requests = create_requests(num_requests=10)

        for i, request in enumerate(requests):
            scheduler.add_request(request)
            self.assertIn(request.request_id, scheduler.requests)
            self.assertEqual(len(scheduler.waiting), i + 1)

    def test_finish_request(self):
        scheduler = self.create_scheduler()
        requests = create_requests(num_requests=10)
        for request in requests:
            scheduler.add_request(request)

        for i, request in enumerate(requests):
            scheduler.finish_requests(request.request_id,
                                      RequestStatus.FINISHED_ABORTED)
            self.assertNotIn(request.request_id, scheduler.requests)
            self.assertEqual(len(scheduler.waiting), 9 - i)

    def test_get_num_unfinished_requests(self):
        scheduler = self.create_scheduler()
        requests = create_requests(num_requests=10)
        for request in requests:
            scheduler.add_request(request)

        for i, request in enumerate(requests):
            scheduler.finish_requests(request.request_id,
                                      RequestStatus.FINISHED_STOPPED)
            self.assertEqual(scheduler.get_num_unfinished_requests(),
                             len(requests) - i - 1)

    def test_schedule(self):
        '''Test scheduling.
        Two cases: default APC/no prompt logprobs; APC=True + prompt logprobs
        '''
        scheduler = self.create_scheduler()
        scheduler.scheduler_config.chunked_prefill_enabled = True
        requests = create_requests(num_requests=10)
        for request in requests:
            scheduler.add_request(request)

        # Test initial scheduling
        output = scheduler.schedule()
        self.assertEqual(len(output.scheduled_new_reqs), len(requests))
        self.assertEqual(output.scheduled_cached_reqs.num_reqs, 0)
        self.assertEqual(len(output.finished_req_ids), 0)
        # Verify all requests are scheduled.
        for req_id, num_tokens in output.num_scheduled_tokens.items():
            self.assertEqual(num_tokens,
                             len(requests[int(req_id)].prompt_token_ids))

        # Verify requests moved from waiting to running
        self.assertEqual(len(scheduler.waiting), 0)
        self.assertEqual(len(scheduler.running), len(requests))
        for i, request in enumerate(requests):
            self.assertEqual(scheduler.running[i], request)

    def test_schedule_multimodal_requests(self):
        scheduler = self.create_scheduler()
        scheduler.scheduler_config.chunked_prefill_enabled = True
        mm_positions = [[PlaceholderRange(offset=i, length=10)]
                        for i in range(10)]
        requests = create_requests(
            num_requests=10,
            mm_positions=mm_positions,
        )
        for request in requests:
            scheduler.add_request(request)

        output = scheduler.schedule()
        self.assertEqual(len(output.scheduled_new_reqs), len(requests))
        self.assertEqual(output.scheduled_cached_reqs.num_reqs, 0)
        self.assertEqual(len(output.finished_req_ids), 0)
        for req_id, num_tokens in output.num_scheduled_tokens.items():
            assert num_tokens == len(requests[int(req_id)].prompt_token_ids)

        # Verify all requests are scheduled.
        for req_id, num_tokens in output.num_scheduled_tokens.items():
            self.assertEqual(num_tokens,
                             len(requests[int(req_id)].prompt_token_ids))
        self.assertEqual(len(output.scheduled_encoder_inputs), len(requests))
        for req_id, encoder_input in output.scheduled_encoder_inputs.items():
            assert len(encoder_input) == 1

        # Verify requests moved from waiting to running
        self.assertEqual(len(scheduler.waiting), 0)
        self.assertEqual(len(scheduler.running), len(requests))
        for i, request in enumerate(requests):
            self.assertEqual(scheduler.running[i], request)

    def test_schedule_enable_prefix_caching(self):
        '''Test scheduling.
        Two cases: default APC/no prompt logprobs; APC=True + prompt logprobs
        '''
        global ENABLE_PREFIX_CACHING
        ENABLE_PREFIX_CACHING = True
        global PROMPT_LOGPROBS
        PROMPT_LOGPROBS = 5
        scheduler = self.create_scheduler()
        scheduler.scheduler_config.chunked_prefill_enabled = False
        requests = create_requests(num_requests=10)
        for request in requests:
            scheduler.add_request(request)

        # Test initial scheduling
        output = scheduler.schedule()
        self.assertEqual(len(output.scheduled_new_reqs), len(requests))
        self.assertEqual(output.scheduled_cached_reqs.num_reqs, 0)
        self.assertEqual(len(output.finished_req_ids), 0)
        # Verify all requests are scheduled.
        for req_id, num_tokens in output.num_scheduled_tokens.items():
            self.assertEqual(num_tokens,
                             len(requests[int(req_id)].prompt_token_ids))

        # Verify requests moved from waiting to running
        self.assertEqual(len(scheduler.waiting), 0)
        self.assertEqual(len(scheduler.running), len(requests))
        for i, request in enumerate(requests):
            self.assertEqual(scheduler.running[i], request)

    def test_stop_via_update_from_output(self):
        """Test stopping behavior through update_from_output"""
        global NUM_SPECULATIVE_TOKENS
        NUM_SPECULATIVE_TOKENS = 1
        scheduler = self.create_scheduler()

        # Test case 1: Stop on EOS token
        requests = create_requests(num_requests=2, max_tokens=10)
        for req in requests:
            req.num_computed_tokens = req.num_tokens
            scheduler.requests[req.request_id] = req
            scheduler.running.append(req)
            req.status = RequestStatus.RUNNING

        scheduler_output = SchedulerOutput(scheduled_new_reqs=[],
                                           scheduled_cached_reqs=[],
                                           num_scheduled_tokens={
                                               requests[0].request_id: 1,
                                               requests[1].request_id: 2
                                           },
                                           total_num_scheduled_tokens=3,
                                           scheduled_encoder_inputs={},
                                           scheduled_spec_decode_tokens={
                                               requests[0].request_id: [],
                                               requests[1].request_id: [10]
                                           },
                                           num_common_prefix_blocks=0,
                                           finished_req_ids=set(),
                                           free_encoder_mm_hashes=[])
        model_output = ModelRunnerOutput(
            req_ids=[req.request_id for req in requests],
            req_id_to_index={
                req.request_id: i
                for i, req in enumerate(requests)
            },
            sampled_token_ids=[[EOS_TOKEN_ID], [10, 11]
                               ],  # First request hits EOS, second continues
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[])

        scheduler.update_from_output(scheduler_output, model_output)

        # Verify first request stopped, second continues
        self.assertEqual(len(scheduler.running), 1)
        self.assertEqual(scheduler.running[0].request_id,
                         requests[1].request_id)
        self.assertEqual(requests[0].status, RequestStatus.FINISHED_STOPPED)
        self.assertIn(requests[0].request_id, scheduler.finished_req_ids)
        self.assertEqual(list(requests[0].output_token_ids), [EOS_TOKEN_ID])
        self.assertEqual(list(requests[1].output_token_ids), [10, 11])

        # Test case 2: Stop on custom stop token
        NUM_SPECULATIVE_TOKENS = 2
        scheduler = self.create_scheduler()
        requests = create_requests(num_requests=2,
                                   max_tokens=10,
                                   stop_token_ids=[42, 43])
        for req in requests:
            req.num_computed_tokens = req.num_tokens
            scheduler.requests[req.request_id] = req
            scheduler.running.append(req)
            req.status = RequestStatus.RUNNING

        scheduler_output = SchedulerOutput(scheduled_new_reqs=[],
                                           scheduled_cached_reqs=[],
                                           num_scheduled_tokens={
                                               requests[0].request_id: 3,
                                               requests[1].request_id: 2
                                           },
                                           total_num_scheduled_tokens=5,
                                           scheduled_encoder_inputs={},
                                           scheduled_spec_decode_tokens={
                                               requests[0].request_id:
                                               [10, 42],
                                               requests[1].request_id: [13]
                                           },
                                           num_common_prefix_blocks=0,
                                           finished_req_ids=set(),
                                           free_encoder_mm_hashes=[])
        model_output = ModelRunnerOutput(
            req_ids=[req.request_id for req in requests],
            req_id_to_index={
                req.request_id: i
                for i, req in enumerate(requests)
            },
            sampled_token_ids=[[10, 42, 12],
                               [13, 14]],  # First request hits stop token
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[])

        scheduler.update_from_output(scheduler_output, model_output)

        # Verify first request stopped on custom token
        self.assertEqual(len(scheduler.running), 1)
        self.assertEqual(scheduler.running[0].request_id,
                         requests[1].request_id)
        self.assertEqual(requests[0].status, RequestStatus.FINISHED_STOPPED)
        self.assertEqual(requests[0].stop_reason, 42)
        self.assertIn(requests[0].request_id, scheduler.finished_req_ids)
        self.assertEqual(list(requests[0].output_token_ids), [10, 42])
        self.assertEqual(list(requests[1].output_token_ids), [13, 14])

        # Test case 3: Stop on max tokens
        NUM_SPECULATIVE_TOKENS = 2
        scheduler = self.create_scheduler()
        requests = create_requests(num_requests=2, max_tokens=2)
        for req in requests:
            req.num_computed_tokens = req.num_tokens
            scheduler.requests[req.request_id] = req
            scheduler.running.append(req)
            req.status = RequestStatus.RUNNING

        scheduler_output = SchedulerOutput(scheduled_new_reqs=[],
                                           scheduled_cached_reqs=[],
                                           num_scheduled_tokens={
                                               requests[0].request_id: 3,
                                               requests[1].request_id: 1
                                           },
                                           total_num_scheduled_tokens=4,
                                           scheduled_encoder_inputs={},
                                           scheduled_spec_decode_tokens={
                                               requests[0].request_id:
                                               [10, 11],
                                               requests[1].request_id: []
                                           },
                                           num_common_prefix_blocks=0,
                                           finished_req_ids=set(),
                                           free_encoder_mm_hashes=[])
        model_output = ModelRunnerOutput(
            req_ids=[req.request_id for req in requests],
            req_id_to_index={
                req.request_id: i
                for i, req in enumerate(requests)
            },
            sampled_token_ids=[[10, 11, 12],
                               [13]],  # First request exceeds max_tokens
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[])
        scheduler.update_from_output(scheduler_output, model_output)

        # Verify first request stopped due to length
        self.assertEqual(len(scheduler.running), 1)
        self.assertEqual(scheduler.running[0].request_id,
                         requests[1].request_id)
        self.assertEqual(requests[0].status,
                         RequestStatus.FINISHED_LENGTH_CAPPED)
        self.assertIn(requests[0].request_id, scheduler.finished_req_ids)
        self.assertEqual(list(requests[0].output_token_ids), [10, 11])
        self.assertEqual(list(requests[1].output_token_ids), [13])

        # Test case 4: Ignore EOS flag
        scheduler = self.create_scheduler()
        requests = create_requests(num_requests=1, max_tokens=10)
        requests[0].sampling_params.ignore_eos = True
        requests[0].sampling_params._eos_token_id = None
        requests[0].num_computed_tokens = requests[0].num_tokens
        scheduler.requests[requests[0].request_id] = requests[0]
        scheduler.running.append(requests[0])

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=[],
            num_scheduled_tokens={requests[0].request_id: 3},
            total_num_scheduled_tokens=3,
            scheduled_encoder_inputs={},
            scheduled_spec_decode_tokens={
                requests[0].request_id: [EOS_TOKEN_ID, 10]
            },
            num_common_prefix_blocks=0,
            finished_req_ids=set(),
            free_encoder_mm_hashes=[])
        model_output = ModelRunnerOutput(
            req_ids=[requests[0].request_id],
            req_id_to_index={requests[0].request_id: 0},
            sampled_token_ids=[[EOS_TOKEN_ID, 10, 11]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[])

        scheduler.update_from_output(scheduler_output, model_output)

        # Verify request continues past EOS
        self.assertEqual(len(scheduler.running), 1)
        self.assertFalse(requests[0].is_finished())
        self.assertEqual(list(requests[0].output_token_ids),
                         [EOS_TOKEN_ID, 10, 11])

    def test_schedule_concurrent_batches(self):
        global MAX_NUM_BATCHED_TOKENS
        global ENABLE_PREFIX_CACHING
        global ENABLE_CHUNKED_PREFILL
        global MAX_NUM_SEQS
        global PROMPT_LOGPROBS
        ENABLE_PREFIX_CACHING = None
        MAX_NUM_BATCHED_TOKENS = 1024
        MAX_NUM_SEQS = 2
        ENABLE_CHUNKED_PREFILL = True
        PROMPT_LOGPROBS = None

        enable_prefix_caching_list = [None, True]
        prompt_logprobs_list = [None, 5]

        for i in range(len(enable_prefix_caching_list)):
            ENABLE_PREFIX_CACHING = enable_prefix_caching_list[i]
            PROMPT_LOGPROBS = prompt_logprobs_list[i]
            scheduler = self.create_scheduler()
            requests = create_requests(
                num_requests=2,
                num_tokens=512,
            )

            # Schedule the first request.
            scheduler.add_request(requests[0])
            scheduler_output0 = scheduler.schedule()
            self.assertEqual(len(scheduler_output0.scheduled_new_reqs), 1)
            self.assertEqual(
                scheduler_output0.num_scheduled_tokens[requests[0].request_id],
                512)

            # The first request is still running, so only schedule the second request.
            scheduler.add_request(requests[1])
            scheduler_output1 = scheduler.schedule()
            self.assertEqual(len(scheduler_output1.scheduled_new_reqs), 1)
            self.assertEqual(
                scheduler_output1.num_scheduled_tokens[requests[1].request_id],
                512)

            # Model output of the first request.
            model_runner_output = ModelRunnerOutput(
                req_ids=[requests[0].request_id],
                req_id_to_index={requests[0].request_id: 0},
                sampled_token_ids=[[0]],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[])

            scheduler.update_from_output(scheduler_output0,
                                         model_runner_output)

            # Schedule the next step.
            # The first request can be scheduled again while the second
            # request is still running.
            scheduler.schedule()
            # Model output of the second request.
            model_runner_output = ModelRunnerOutput(
                req_ids=[requests[1].request_id],
                req_id_to_index={requests[1].request_id: 0},
                sampled_token_ids=[[0]],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[])

            scheduler.update_from_output(scheduler_output1,
                                         model_runner_output)

    def test_schedule_spec_decoding_stats(self):
        """Test scheduling behavior with speculative decoding.

        This test verifies that:
        1. Speculated tokens get scheduled correctly
        2. Spec decoding stats properly count number of draft and accepted tokens
        """
        spec_tokens_list: List[List[List[int]]] = [[[1, 2, 3]], [[1, 2, 3]],
                                                   [[1, 2], [3]], [[1]], [[]],
                                                   [[1, 2, 3], [4, 5, 6]]]
        output_tokens_list: List[List[List[int]]] = [[[1, 2, 3, 4]], [[1, 5]],
                                                     [[1, 2, 5], [3, 4]],
                                                     [[1, 2]], [[5]],
                                                     [[1, 2, 7], [4, 8]]]
        expected_list: List[Tuple[int, int,
                                  int, List[int]]] = [(1, 3, 3, [1, 1, 1]),
                                                      (1, 3, 1, [1, 0, 0]),
                                                      (2, 3, 3, [2, 1]),
                                                      (1, 1, 1, [1]),
                                                      (0, 0, 0, [0]),
                                                      (2, 6, 3, [2, 1, 0])]

        global NUM_SPECULATIVE_TOKENS
        for idx in range(len(spec_tokens_list)):
            spec_tokens = spec_tokens_list[idx]
            output_tokens = output_tokens_list[idx]
            expected = expected_list[idx]
            num_spec_tokens = max(1, max(len(t) for t in spec_tokens))
            NUM_SPECULATIVE_TOKENS = num_spec_tokens
            scheduler = self.create_scheduler()
            requests = create_requests(num_requests=len(spec_tokens),
                                       num_tokens=1)
            req_ids = []
            req_to_index = {}
            for i, request in enumerate(requests):
                scheduler.add_request(request)
                req_ids.append(request.request_id)
                req_to_index[request.request_id] = i

            # Schedule a decode, which will also draft speculative tokens
            output = scheduler.schedule()
            self.assertEqual(len(output.scheduled_new_reqs), len(requests))
            self.assertEqual(output.total_num_scheduled_tokens, len(requests))
            for i in range(len(requests)):
                req_id = requests[i].request_id
                self.assertEqual(output.num_scheduled_tokens[req_id], 1)
                self.assertNotIn(req_id, output.scheduled_spec_decode_tokens)

            model_runner_output = ModelRunnerOutput(
                req_ids=req_ids,
                req_id_to_index=req_to_index,
                sampled_token_ids=[[0] for _ in range(len(requests))],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[])
            draft_token_ids = DraftTokenIds(req_ids, spec_tokens)

            engine_core_outputs = scheduler.update_from_output(
                output, model_runner_output)
            scheduler.update_draft_token_ids(draft_token_ids)

            for i in range(len(requests)):
                running_req = scheduler.running[i]
                # The prompt token
                self.assertEqual(running_req.num_computed_tokens, 1)
                # The prompt token and the sampled token
                self.assertEqual(running_req.num_tokens, 2)
                # The prompt token, the sampled token, and the speculated tokens
                self.assertEqual(running_req.num_tokens_with_spec,
                                 2 + len(spec_tokens[i]))

            # No draft or accepted tokens counted yet
            self.assertTrue(
                not engine_core_outputs
                or (engine_core_outputs[0].scheduler_stats.spec_decoding_stats
                    is None))

            # Schedule the speculated tokens for validation
            output = scheduler.schedule()
            self.assertEqual(len(output.scheduled_new_reqs), 0)
            # The sampled token and speculated tokens
            self.assertEqual(
                output.total_num_scheduled_tokens,
                len(requests) + sum(len(ids) for ids in spec_tokens))
            for i in range(len(requests)):
                req_id = requests[i].request_id
                self.assertEqual(output.num_scheduled_tokens[req_id],
                                 1 + len(spec_tokens[i]))
                if spec_tokens[i]:
                    self.assertEqual(
                        len(output.scheduled_spec_decode_tokens[req_id]),
                        len(spec_tokens[i]))
                else:
                    self.assertNotIn(req_id,
                                     output.scheduled_spec_decode_tokens)

            model_runner_output = ModelRunnerOutput(
                req_ids=req_ids,
                req_id_to_index=req_to_index,
                sampled_token_ids=output_tokens,
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[])

            engine_core_outputs = scheduler.update_from_output(
                output, model_runner_output)

            scheduler_stats = engine_core_outputs[0].scheduler_stats \
                if engine_core_outputs else None
            if expected[0] == 0:
                self.assertIsNone(scheduler_stats.spec_decoding_stats)
            else:
                self.assertIsNotNone(scheduler_stats.spec_decoding_stats)
                stats = scheduler_stats.spec_decoding_stats
                self.assertEqual(stats.num_drafts, expected[0])
                self.assertEqual(stats.num_draft_tokens, expected[1])
                self.assertEqual(stats.num_accepted_tokens, expected[2])
                self.assertEqual(stats.num_accepted_tokens_per_pos,
                                 expected[3])

    def assert_scheduler_empty(self, scheduler):
        """Confirm the scheduler is "empty" - i.e. no leaks."""
        # Scheduler Metadata.
        scheduler = self.create_scheduler()
        self.assertEqual(len(scheduler.requests), 0)
        self.assertEqual(len(scheduler.waiting), 0)
        self.assertEqual(len(scheduler.running), 0)
        self.assertEqual(len(scheduler.finished_req_ids), 0)

        # EncoderCacheManager.
        self.assertEqual(len(scheduler.encoder_cache_manager.freed), 0)
        self.assertEqual(len(scheduler.encoder_cache_manager.cached), 0)

        # KVCache Manager.
        self.assertEqual(
            len(scheduler.kv_cache_manager.coordinator.single_type_managers[0].
                req_to_blocks), 0)
        self.assertEqual(
            len(scheduler.kv_cache_manager.coordinator.single_type_managers[0].
                num_cached_block), 0)
        num_free_blocks = (scheduler.kv_cache_manager.block_pool.
                           free_block_queue.num_free_blocks)
        self.assertEqual(
            num_free_blocks,
            scheduler.kv_cache_manager.block_pool.num_gpu_blocks - 1)

        # NOTE(rob): just the ref count on blocks will be 0. The hash
        # value, etc will remain since we lazily evict for prefix cache.
        for block in scheduler.kv_cache_manager.block_pool.blocks:
            self.assertEqual(block.ref_cnt, 0)

    def test_memory_leak(self):
        """Test that we do not have a memory leak."""
        scheduler = self.create_scheduler()
        NUM_REQUESTS = 5
        NUM_TOKENS = 10
        MAX_TOKENS = 10
        requests = create_requests(num_requests=NUM_REQUESTS,
                                   num_tokens=NUM_TOKENS,
                                   max_tokens=MAX_TOKENS)

        # Add each request.
        for request in requests:
            scheduler.add_request(request)
            scheduler_output = scheduler.schedule()
            model_runner_output = make_output(scheduler)
            scheduler.update_from_output(scheduler_output, model_runner_output)

        # Iterate until done.
        while True:
            scheduler_output = scheduler.schedule()
            if len(scheduler.running) == 0:
                break
            model_runner_output = make_output(scheduler)
            scheduler.update_from_output(scheduler_output, model_runner_output)

        # Confirm no memory leak.
        self.assert_scheduler_empty(scheduler)

    def test_schedule_running_request_with_zero_new_tokens_still_schedules_waiting(self):
        scheduler = self.create_scheduler()
        running_req, waiting_req = create_requests(num_requests=2)

        running_req.status = RequestStatus.RUNNING
        running_req.num_computed_tokens = running_req.num_tokens_with_spec
        scheduler.requests[running_req.request_id] = running_req
        scheduler.running.append(running_req)
        scheduler.add_request(waiting_req)

        output = scheduler.schedule()

        self.assertEqual(len(output.scheduled_new_reqs), 1)
        self.assertEqual(output.scheduled_new_reqs[0].req_id,
                 waiting_req.request_id)
        self.assertEqual(output.num_scheduled_tokens[waiting_req.request_id],
                 waiting_req.num_tokens)
        self.assertEqual(len(scheduler.running), 2)
        self.assertEqual(len(scheduler.waiting), 0)

    def test_schedule_skips_waiting_remote_and_fsm_requests(self):
        scheduler = self.create_scheduler()
        requests = create_requests(num_requests=3)
        for request in requests:
            scheduler.add_request(request)

        requests[0].status = RequestStatus.WAITING_FOR_REMOTE_KVS
        requests[1].status = RequestStatus.WAITING_FOR_FSM
        requests[1].structured_output_request = MagicMock(grammar=None)
        scheduler._update_waiting_for_remote_kv = MagicMock(return_value=False)

        output = scheduler.schedule()

        self.assertEqual(len(output.scheduled_new_reqs), 1)
        self.assertEqual(output.scheduled_new_reqs[0].req_id,
                         requests[2].request_id)
        self.assertEqual(len(scheduler.running), 1)
        self.assertEqual(scheduler.running[0].request_id,
                         requests[2].request_id)
        self.assertEqual(len(scheduler.waiting), 2)
        self.assertEqual(requests[0].status,
                         RequestStatus.WAITING_FOR_REMOTE_KVS)
        self.assertEqual(requests[1].status, RequestStatus.WAITING_FOR_FSM)

    def test_schedule_async_kv_load_keeps_request_waiting_for_remote(self):
        scheduler = self.create_scheduler()
        request = create_requests(num_requests=1)[0]
        scheduler.add_request(request)

        scheduler.connector = MagicMock()
        scheduler.connector.get_num_new_matched_tokens.return_value = (2, True)
        scheduler.connector.update_state_after_alloc = MagicMock()
        scheduler.connector.build_connector_meta.return_value = None
        scheduler.connector.take_events.return_value = None

        output = scheduler.schedule()

        self.assertEqual(output.total_num_scheduled_tokens, 0)
        self.assertEqual(len(scheduler.running), 0)
        self.assertEqual(len(scheduler.waiting), 1)
        self.assertEqual(request.status, RequestStatus.WAITING_FOR_REMOTE_KVS)
        scheduler.connector.update_state_after_alloc.assert_called_once()

    def test_schedule_preempts_running_request_when_allocation_fails(self):
        scheduler = self.create_scheduler()
        request = create_requests(num_requests=1)[0]

        request.status = RequestStatus.RUNNING
        scheduler.requests[request.request_id] = request
        scheduler.running.append(request)
        scheduler.kv_cache_manager.allocate_slots = MagicMock(return_value=None)
        scheduler.kv_cache_manager.free = MagicMock()

        output = scheduler.schedule()

        self.assertEqual(output.total_num_scheduled_tokens, 0)
        self.assertEqual(len(scheduler.running), 0)
        self.assertEqual(len(scheduler.waiting), 1)
        self.assertEqual(request.status, RequestStatus.PREEMPTED)
        scheduler.kv_cache_manager.free.assert_called_once_with(request)

    def test_schedule_resumes_preempted_request_from_waiting(self):
        scheduler = self.create_scheduler()
        request = create_requests(num_requests=1)[0]
        scheduler.add_request(request)
        request.status = RequestStatus.PREEMPTED

        output = scheduler.schedule()

        self.assertEqual(len(output.scheduled_new_reqs), 0)
        self.assertEqual(output.scheduled_cached_reqs.num_reqs, 1)
        self.assertEqual(len(scheduler.running), 1)
        self.assertEqual(scheduler.running[0].status, RequestStatus.RUNNING)

    def test_schedule_ready_remote_kv_request_is_scheduled(self):
        scheduler = self.create_scheduler()
        request = create_requests(num_requests=1)[0]
        scheduler.add_request(request)
        request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
        scheduler._update_waiting_for_remote_kv = MagicMock(return_value=True)

        output = scheduler.schedule()

        self.assertEqual(len(output.scheduled_new_reqs), 1)
        self.assertEqual(output.scheduled_new_reqs[0].req_id,
                         request.request_id)
        self.assertEqual(len(scheduler.running), 1)
        self.assertEqual(request.status, RequestStatus.RUNNING)

    def test_schedule_skips_request_when_connector_match_unknown(self):
        scheduler = self.create_scheduler()
        request = create_requests(num_requests=1)[0]
        scheduler.add_request(request)

        scheduler.connector = MagicMock()
        scheduler.connector.get_num_new_matched_tokens.return_value = (None,
                                                                       False)
        scheduler.connector.build_connector_meta.return_value = None
        scheduler.connector.take_events.return_value = None

        output = scheduler.schedule()

        self.assertEqual(output.total_num_scheduled_tokens, 0)
        self.assertEqual(len(scheduler.running), 0)
        self.assertEqual(len(scheduler.waiting), 1)
        self.assertEqual(scheduler.waiting.peek_request().request_id,
                         request.request_id)

    def test_schedule_breaks_when_chunked_prefill_disabled_and_budget_too_small(
            self):
        scheduler = self.create_scheduler()
        scheduler.scheduler_config.enable_chunked_prefill = False
        scheduler.max_num_scheduled_tokens = 1
        request = create_requests(num_requests=1, num_tokens=10)[0]
        scheduler.add_request(request)

        output = scheduler.schedule()

        self.assertEqual(output.total_num_scheduled_tokens, 0)
        self.assertEqual(len(output.scheduled_new_reqs), 0)
        self.assertEqual(len(scheduler.waiting), 1)

    def test_schedule_skips_waiting_request_when_lora_budget_is_full(self):
        scheduler = self.create_scheduler()
        running_req, waiting_req = create_requests(num_requests=2)
        running_req.status = RequestStatus.RUNNING
        running_req.lora_request = SimpleNamespace(lora_int_id=1)
        waiting_req.lora_request = SimpleNamespace(lora_int_id=2)
        scheduler.requests[running_req.request_id] = running_req
        scheduler.running.append(running_req)
        scheduler.add_request(waiting_req)
        scheduler.lora_config = SimpleNamespace(max_loras=1)

        output = scheduler.schedule()

        self.assertIn(running_req.request_id, output.num_scheduled_tokens)
        self.assertNotIn(waiting_req.request_id, output.num_scheduled_tokens)
        self.assertEqual(len(scheduler.waiting), 1)

    def test_schedule_priority_preempts_highest_priority_running_request(self):
        scheduler = self.create_scheduler()
        req0, req1 = create_requests(num_requests=2)
        for index, request in enumerate((req0, req1)):
            request.status = RequestStatus.RUNNING
            request.priority = index
            request.arrival_time = index
            scheduler.requests[request.request_id] = request
            scheduler.running.append(request)

        scheduler.policy = SchedulingPolicy.PRIORITY
        scheduler.kv_cache_manager.allocate_slots = MagicMock(return_value=None)
        scheduler.kv_cache_manager.free = MagicMock()

        output = scheduler.schedule()

        self.assertEqual(output.total_num_scheduled_tokens, 0)
        self.assertEqual(len(scheduler.running), 0)
        self.assertEqual(len(scheduler.waiting), 2)
        self.assertEqual(req1.status, RequestStatus.PREEMPTED)

    def test_schedule_stops_waiting_when_running_at_sequence_cap(self):
        scheduler = self.create_scheduler()
        running_req, waiting_req = create_requests(num_requests=2)
        running_req.status = RequestStatus.RUNNING
        scheduler.requests[running_req.request_id] = running_req
        scheduler.running.append(running_req)
        scheduler.add_request(waiting_req)
        scheduler.max_num_running_reqs = 1

        output = scheduler.schedule()

        self.assertEqual(output.total_num_scheduled_tokens,
                         running_req.num_tokens)
        self.assertEqual(len(output.scheduled_new_reqs), 0)
        self.assertEqual(len(scheduler.waiting), 1)

    def test_schedule_fsm_ready_request_returns_to_waiting_and_runs(self):
        scheduler = self.create_scheduler()
        request = create_requests(num_requests=1)[0]
        scheduler.add_request(request)
        request.status = RequestStatus.WAITING_FOR_FSM
        request.structured_output_request = SimpleNamespace(grammar=object())

        output = scheduler.schedule()

        self.assertEqual(len(output.scheduled_new_reqs), 1)
        self.assertEqual(output.scheduled_new_reqs[0].req_id,
                         request.request_id)
        self.assertEqual(request.status, RequestStatus.RUNNING)

    def test_schedule_publishes_combined_cache_and_connector_events(self):
        scheduler = self.create_scheduler()
        request = create_requests(num_requests=1)[0]
        scheduler.add_request(request)
        scheduler.kv_cache_manager.take_events = MagicMock(return_value=["kv"])
        scheduler.kv_event_publisher.publish = MagicMock()
        scheduler.connector = MagicMock()
        scheduler.connector.get_num_new_matched_tokens.return_value = (0, False)
        scheduler.connector.build_connector_meta.return_value = None
        scheduler.connector.take_events.return_value = ["connector"]
        scheduler.connector.update_state_after_alloc = MagicMock()

        scheduler.schedule()

        published_batch = scheduler.kv_event_publisher.publish.call_args[0][0]
        self.assertEqual(published_batch.events, ["kv", "connector"])
