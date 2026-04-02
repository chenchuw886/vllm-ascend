import os
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import CacheConfig, DeviceConfig, ModelConfig, SchedulerConfig, SpeculativeConfig, VllmConfig
from vllm.multimodal.inputs import MultiModalFeatureSpec, MultiModalKwargsItem, PlaceholderRange
from vllm.sampling_params import SamplingParams
from vllm.config import SchedulerConfig
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.engine import EngineCoreEventType
from vllm.v1.request import RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from vllm_ascend.core.recompute_scheduler import (
    AsyncRecomputeScheduler,
    RecomputeSchedulerOutput,
    RecomputeScheduler,
    RecomputeSchedulerConfig,
)
from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID

from tests.ut.core.test_scheduler_dynamic_batch import DummyMMRegistry


EOS_TOKEN_ID = 50256
MAX_NUM_BATCHED_TOKENS = 10000


def create_requests(
    num_requests: int,
    num_tokens: int = 10,
    mm_positions: list[list[PlaceholderRange]] | None = None,
    max_tokens: int = 16,
    stop_token_ids: list[int] | None = None,
    block_size: int = 3,
    hash_fn=sha256,
    ignore_eos: bool = False,
):
    init_none_hash(hash_fn)
    sampling_params = SamplingParams(
        ignore_eos=ignore_eos,
        max_tokens=max_tokens,
        stop_token_ids=stop_token_ids,
    )
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
                    modality="image",
                )
                mm_features.append(mm_feature)
        request = Request(
            request_id=f"{i}",
            prompt_token_ids=[i] * num_tokens,
            sampling_params=sampling_params,
            pooling_params=None,
            mm_features=mm_features if mm_features else None,
            block_hasher=get_request_block_hasher(block_size, hash_fn),
        )
        requests.append(request)
    return requests


def make_output(scheduler):
    req_ids = [req.request_id for req in scheduler.running]
    req_id_to_index = {req.request_id: i for i, req in enumerate(scheduler.running)}
    sampled_token_ids = [[1000]] * len(scheduler.running)
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index=req_id_to_index,
        sampled_token_ids=sampled_token_ids,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


@patch("vllm.config.ModelConfig.__post_init__", MagicMock())
@patch("vllm.config.VllmConfig.__post_init__", MagicMock())
def create_recompute_scheduler(num_speculative_tokens: int | None = None):
    block_size = 16
    scheduler_config = SchedulerConfig(
        max_num_seqs=16,
        is_encoder_decoder=False,
        max_model_len=MAX_NUM_BATCHED_TOKENS,
        long_prefill_token_threshold=0,
        disable_chunked_mm_input=False,
        enable_chunked_prefill=True,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
    )
    scheduler_config.max_num_encoder_input_tokens = 10000
    scheduler_config.encoder_cache_size = 10000
    scheduler_config.chunked_prefill_enabled = True

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
    model_config.hf_text_config.model_type = "test_model"

    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.9,
        swap_space=0,
        cache_dtype="auto",
    )

    speculative_config = None
    if num_speculative_tokens is not None:
        speculative_config = SpeculativeConfig(model="ngram", num_speculative_tokens=num_speculative_tokens)

    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        speculative_config=speculative_config,
        device_config=DeviceConfig("cpu"),
    )

    kv_cache_config = KVCacheConfig(
        num_blocks=10000,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(block_size=block_size, num_kv_heads=1, head_size=1, dtype=torch.float32),
            )
        ],
    )
    kv_cache_config.hash_block_size = block_size
    cache_config.num_gpu_blocks = 10000

    scheduler = RecomputeScheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=block_size,
        mm_registry=DummyMMRegistry(),
        log_stats=True,
        structured_output_manager=MagicMock(spec=StructuredOutputManager),
    )
    scheduler.structured_output_manager.should_advance = MagicMock(return_value=False)
    return scheduler


def make_scheduler(**overrides):
    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)
    defaults = {
        "requests": {},
        "waiting": MagicMock(),
        "log_stats": False,
        "is_kv_producer": False,
        "is_hybrid_model": False,
        "is_mtp_kv_consumer": False,
        "num_spec_tokens": 0,
        "_update_request_as_session": MagicMock(),
        "finish_requests": MagicMock(),
        "connector": object(),
        "kv_cache_manager": MagicMock(),
        "finished_recving_kv_req_ids": set(),
        "failed_recving_kv_req_ids": set(),
        "block_size": 4,
    }
    defaults.update(overrides)
    for key, value in defaults.items():
        setattr(scheduler, key, value)
    return scheduler


class CapturingConfig:

    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_recompute_scheduler_config_initialize_from_config_sets_scheduler_class():
    scheduler_config = SchedulerConfig(
        max_num_seqs=4,
        is_encoder_decoder=False,
        max_model_len=128,
        max_num_batched_tokens=4096,
        enable_chunked_prefill=True,
    )
    scheduler_config.async_scheduling = True
    vllm_config = SimpleNamespace(
        scheduler_config=scheduler_config,
        model_config=SimpleNamespace(max_model_len=1024, is_encoder_decoder=True),
    )

    config = RecomputeSchedulerConfig.initialize_from_config.__func__(CapturingConfig, vllm_config)

    assert config.kwargs["scheduler_cls"] == "vllm_ascend.core.recompute_scheduler.AsyncRecomputeScheduler"
    assert config.kwargs["max_model_len"] == 1024
    assert config.kwargs["is_encoder_decoder"] is True
    assert config.kwargs["max_num_batched_tokens"] == 4096


def test_recompute_scheduler_config_initialize_from_config_uses_sync_scheduler_when_disabled():
    scheduler_config = SchedulerConfig(
        max_num_seqs=4,
        is_encoder_decoder=False,
        max_model_len=128,
        max_num_batched_tokens=256,
        enable_chunked_prefill=True,
    )
    scheduler_config.async_scheduling = False
    vllm_config = SimpleNamespace(
        scheduler_config=scheduler_config,
        model_config=SimpleNamespace(max_model_len=2048, is_encoder_decoder=False),
    )

    config = RecomputeSchedulerConfig.initialize_from_config.__func__(CapturingConfig, vllm_config)

    assert config.kwargs["scheduler_cls"] == "vllm_ascend.core.recompute_scheduler.RecomputeScheduler"
    assert config.kwargs["max_model_len"] == 2048
    assert config.kwargs["is_encoder_decoder"] is False
    assert config.kwargs["max_num_batched_tokens"] == 256


@patch("vllm_ascend.core.recompute_scheduler.Scheduler.__init__")
def test_recompute_scheduler_init_sets_runtime_flags(mock_super_init):
    fake_config = SimpleNamespace(
        speculative_config=object(),
        kv_transfer_config=SimpleNamespace(is_kv_consumer=True, is_kv_producer=True),
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type="qwen3_next-large")),
    )

    mock_super_init.return_value = None

    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)
    scheduler.vllm_config = fake_config
    RecomputeScheduler.__init__(scheduler)

    assert scheduler.is_mtp_kv_consumer is True
    assert scheduler.is_kv_producer is True
    assert scheduler.is_hybrid_model is True


@patch("vllm_ascend.core.recompute_scheduler.AsyncScheduler.__init__")
def test_async_recompute_scheduler_init_calls_parent(mock_async_init):
    AsyncRecomputeScheduler(SimpleNamespace(), SimpleNamespace(), MagicMock())

    mock_async_init.assert_called_once()


def test_add_request_initializes_streaming_queue_tokens_and_placeholder_spec_tokens():
    scheduler = make_scheduler(
        log_stats=True,
        is_kv_producer=True,
        is_hybrid_model=True,
        is_mtp_kv_consumer=True,
        num_spec_tokens=3,
    )
    request = SimpleNamespace(
        request_id="req-1",
        resumable=True,
        prompt_token_ids=[1, 2, 3],
        _all_token_ids=[1, 2, 3],
        num_prompt_tokens=3,
        num_tokens=3,
        spec_token_ids=[],
        record_event=MagicMock(),
    )

    scheduler.add_request(request)

    assert isinstance(request.streaming_queue, deque)
    assert request.prompt_token_ids == [1, 2]
    assert request._all_token_ids == [1, 2]
    assert request.num_prompt_tokens == 2
    assert request.spec_token_ids == [PLACEHOLDER_TOKEN_ID] * 3
    scheduler.waiting.add_request.assert_called_once_with(request)
    assert scheduler.requests[request.request_id] is request
    request.record_event.assert_called_once_with(EngineCoreEventType.QUEUED)


@patch("vllm_ascend.core.recompute_scheduler.StreamingUpdate.from_request", return_value="update")
def test_add_request_appends_streaming_update_for_existing_request(_mock_update):
    existing = SimpleNamespace(status=RequestStatus.RUNNING, streaming_queue=MagicMock())
    scheduler = make_scheduler(requests={"req-1": existing})
    request = SimpleNamespace(request_id="req-1")

    scheduler.add_request(request)

    existing.streaming_queue.append.assert_called_once_with("update")


@patch("vllm_ascend.core.recompute_scheduler.StreamingUpdate.from_request", return_value="update")
def test_add_request_raises_for_duplicate_request_without_stream_queue(_mock_update):
    existing = SimpleNamespace(status=RequestStatus.RUNNING, streaming_queue=None)
    scheduler = make_scheduler(requests={"req-1": existing})

    with pytest.raises(AssertionError, match="duplicate request id"):
        scheduler.add_request(SimpleNamespace(request_id="req-1"))


@patch("vllm_ascend.core.recompute_scheduler.StreamingUpdate.from_request", return_value="update")
def test_add_request_updates_waiting_streaming_session(_mock_update):
    existing = SimpleNamespace(status=RequestStatus.WAITING_FOR_STREAMING_REQ, streaming_queue=deque())
    scheduler = make_scheduler(requests={"req-1": existing})
    request = SimpleNamespace(request_id="req-1")

    scheduler.add_request(request)

    scheduler._update_request_as_session.assert_called_once_with(existing, "update")


@patch("vllm_ascend.core.recompute_scheduler.StreamingUpdate.from_request", return_value=None)
def test_add_request_finishes_streaming_session_when_update_is_none(_mock_update):
    existing = SimpleNamespace(status=RequestStatus.WAITING_FOR_STREAMING_REQ, streaming_queue=deque())
    scheduler = make_scheduler(requests={"req-1": existing})

    scheduler.add_request(SimpleNamespace(request_id="req-1"))

    scheduler.finish_requests.assert_called_once_with("req-1", RequestStatus.FINISHED_ABORTED)


def test_update_waiting_for_remote_kv_returns_false_if_request_not_finished():
    scheduler = make_scheduler(finished_recving_kv_req_ids=set())
    request = SimpleNamespace(request_id="req-1")

    assert scheduler._update_waiting_for_remote_kv(request) is False


def test_update_waiting_for_remote_kv_failed_request_caches_existing_tokens():
    scheduler = make_scheduler(
        finished_recving_kv_req_ids={"req-1"},
        failed_recving_kv_req_ids={"req-1"},
    )
    request = SimpleNamespace(request_id="req-1", num_computed_tokens=5)

    assert scheduler._update_waiting_for_remote_kv(request) is True
    scheduler.kv_cache_manager.cache_blocks.assert_called_once_with(request, 5)
    assert scheduler.failed_recving_kv_req_ids == set()
    assert scheduler.finished_recving_kv_req_ids == set()


def test_update_waiting_for_remote_kv_failed_request_frees_blocks_when_no_valid_tokens():
    scheduler = make_scheduler(
        finished_recving_kv_req_ids={"req-1"},
        failed_recving_kv_req_ids={"req-1"},
    )
    request = SimpleNamespace(request_id="req-1", num_computed_tokens=0)

    assert scheduler._update_waiting_for_remote_kv(request) is True
    scheduler.kv_cache_manager.free.assert_called_once_with(request)


def test_update_waiting_for_remote_kv_uses_partial_single_block_tokens_without_decrement():
    scheduler = make_scheduler(finished_recving_kv_req_ids={"req-1"}, block_size=4)
    scheduler.kv_cache_manager.get_block_ids.return_value = [[1, 2]]
    request = SimpleNamespace(request_id="req-1", num_tokens=10, num_computed_tokens=0)

    assert scheduler._update_waiting_for_remote_kv(request) is True
    scheduler.kv_cache_manager.cache_blocks.assert_called_once_with(request, 8)
    assert request.num_computed_tokens == 8


def test_update_waiting_for_remote_kv_decrements_when_all_request_tokens_are_ready():
    scheduler = make_scheduler(finished_recving_kv_req_ids={"req-1"}, block_size=4)
    scheduler.kv_cache_manager.get_block_ids.return_value = [[1], [2]]
    request = SimpleNamespace(request_id="req-1", num_tokens=8, num_computed_tokens=0)

    assert scheduler._update_waiting_for_remote_kv(request) is True
    scheduler.kv_cache_manager.cache_blocks.assert_called_once_with(request, 7)
    assert request.num_computed_tokens == 7


def test_recompute_schedule():
    scheduler = create_recompute_scheduler()
    requests = create_requests(num_requests=10)

    for request in requests:
        scheduler.add_request(request)

    output = scheduler.schedule()

    assert len(output.scheduled_new_reqs) == len(requests)
    assert output.scheduled_cached_reqs.num_reqs == 0
    assert len(output.finished_req_ids) == 0
    assert output.recomputed_reqs == []
    for req_id, num_tokens in output.num_scheduled_tokens.items():
        assert num_tokens == len(requests[int(req_id)].prompt_token_ids)
    assert len(scheduler.waiting) == 0
    assert len(scheduler.running) == len(requests)


def test_recompute_schedule_multimodal_requests():
    scheduler = create_recompute_scheduler()
    mm_positions = [[PlaceholderRange(offset=i, length=10)] for i in range(10)]
    requests = create_requests(num_requests=10, mm_positions=mm_positions)

    for request in requests:
        scheduler.add_request(request)

    output = scheduler.schedule()

    assert len(output.scheduled_new_reqs) == len(requests)
    assert output.scheduled_cached_reqs.num_reqs == 0
    assert len(output.scheduled_encoder_inputs) == len(requests)
    for req_id, encoder_input in output.scheduled_encoder_inputs.items():
        assert len(encoder_input) == 1
        assert req_id in output.num_scheduled_tokens


def test_recompute_stop_via_update_from_output():
    scheduler = create_recompute_scheduler(num_speculative_tokens=1)
    requests = create_requests(num_requests=2, max_tokens=10)
    for req in requests:
        req.num_computed_tokens = req.num_tokens
        scheduler.requests[req.request_id] = req
        scheduler.running.append(req)
        req.status = RequestStatus.RUNNING

    scheduler_output = RecomputeSchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={requests[0].request_id: 1, requests[1].request_id: 2},
        total_num_scheduled_tokens=3,
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={requests[0].request_id: [], requests[1].request_id: [10]},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        recomputed_reqs=[],
    )
    model_output = ModelRunnerOutput(
        req_ids=[req.request_id for req in requests],
        req_id_to_index={req.request_id: i for i, req in enumerate(requests)},
        sampled_token_ids=[[EOS_TOKEN_ID], [10, 11]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    scheduler.update_from_output(scheduler_output, model_output)

    assert len(scheduler.running) == 1
    assert scheduler.running[0].request_id == requests[1].request_id
    assert requests[0].status == RequestStatus.FINISHED_STOPPED
    assert requests[0].request_id in scheduler.finished_req_ids
    assert list(requests[0].output_token_ids) == [EOS_TOKEN_ID]
    assert list(requests[1].output_token_ids) == [10, 11]

    scheduler = create_recompute_scheduler(num_speculative_tokens=2)
    requests = create_requests(num_requests=2, max_tokens=10, stop_token_ids=[42, 43])
    for req in requests:
        req.num_computed_tokens = req.num_tokens
        scheduler.requests[req.request_id] = req
        scheduler.running.append(req)
        req.status = RequestStatus.RUNNING

    scheduler_output = RecomputeSchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={requests[0].request_id: 3, requests[1].request_id: 2},
        total_num_scheduled_tokens=5,
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={requests[0].request_id: [10, 42], requests[1].request_id: [13]},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        recomputed_reqs=[],
    )
    model_output = ModelRunnerOutput(
        req_ids=[req.request_id for req in requests],
        req_id_to_index={req.request_id: i for i, req in enumerate(requests)},
        sampled_token_ids=[[10, 42, 12], [13, 14]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    scheduler.update_from_output(scheduler_output, model_output)

    assert len(scheduler.running) == 1
    assert requests[0].status == RequestStatus.FINISHED_STOPPED
    assert requests[0].stop_reason == 42

    scheduler = create_recompute_scheduler(num_speculative_tokens=2)
    requests = create_requests(num_requests=2, max_tokens=2)
    for req in requests:
        req.num_computed_tokens = req.num_tokens
        scheduler.requests[req.request_id] = req
        scheduler.running.append(req)
        req.status = RequestStatus.RUNNING

    scheduler_output = RecomputeSchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={requests[0].request_id: 3, requests[1].request_id: 1},
        total_num_scheduled_tokens=4,
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={requests[0].request_id: [10, 11], requests[1].request_id: []},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        recomputed_reqs=[],
    )
    model_output = ModelRunnerOutput(
        req_ids=[req.request_id for req in requests],
        req_id_to_index={req.request_id: i for i, req in enumerate(requests)},
        sampled_token_ids=[[10, 11, 12], [13]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    scheduler.update_from_output(scheduler_output, model_output)

    assert len(scheduler.running) == 1
    assert requests[0].status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert list(requests[0].output_token_ids) == [10, 11]


def test_recompute_schedule_spec_decoding_stats():
    spec_tokens = [[1, 2, 3], [1], []]
    output_tokens = [[1, 2, 3, 4], [1, 5], [5]]

    scheduler = create_recompute_scheduler(num_speculative_tokens=3)
    requests = create_requests(num_requests=len(spec_tokens), num_tokens=1)
    req_ids = []
    req_to_index = {}
    for i, request in enumerate(requests):
        scheduler.add_request(request)
        req_ids.append(request.request_id)
        req_to_index[request.request_id] = i

    output = scheduler.schedule()
    model_runner_output = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index=req_to_index,
        sampled_token_ids=[[0] for _ in range(len(requests))],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    draft_token_ids = DraftTokenIds(req_ids, spec_tokens)

    engine_core_outputs = scheduler.update_from_output(output, model_runner_output)
    scheduler.update_draft_token_ids(draft_token_ids)
    assert not engine_core_outputs or engine_core_outputs[0].scheduler_stats.spec_decoding_stats is None

    output = scheduler.schedule()
    model_runner_output = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index=req_to_index,
        sampled_token_ids=output_tokens,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    engine_core_outputs = scheduler.update_from_output(output, model_runner_output)

    stats = engine_core_outputs[0].scheduler_stats.spec_decoding_stats
    assert stats is not None
    assert stats.num_drafts == 2
    assert stats.num_draft_tokens == 4
    assert stats.num_accepted_tokens == 4
    assert stats.num_accepted_tokens_per_pos == [2, 1, 1]


def test_recompute_memory_leak():
    scheduler = create_recompute_scheduler()
    requests = create_requests(num_requests=5, num_tokens=10, max_tokens=10)

    for request in requests:
        scheduler.add_request(request)
        scheduler_output = scheduler.schedule()
        model_runner_output = make_output(scheduler)
        scheduler.update_from_output(scheduler_output, model_runner_output)

    while True:
        scheduler_output = scheduler.schedule()
        if len(scheduler.running) == 0:
            break
        model_runner_output = make_output(scheduler)
        scheduler.update_from_output(scheduler_output, model_runner_output)

    assert len(scheduler.requests) == 0
    assert len(scheduler.waiting) == 0
    assert len(scheduler.running) == 0


def test_recompute_schedule_recomputes_running_request_for_kv_consumer():
    scheduler = create_recompute_scheduler()
    scheduler.vllm_config.kv_transfer_config = SimpleNamespace(is_kv_producer=False)

    request = create_requests(num_requests=1)[0]
    request.status = RequestStatus.RUNNING
    scheduler.requests[request.request_id] = request
    scheduler.running.append(request)
    scheduler.kv_cache_manager.allocate_slots = MagicMock(return_value=None)
    scheduler.kv_cache_manager.free = MagicMock()

    output = scheduler.schedule()

    assert len(output.recomputed_reqs) == 1
    assert output.recomputed_reqs[0].request_id == request.request_id
    assert scheduler.running == []
    scheduler.kv_cache_manager.free.assert_called_once_with(request)


def test_recompute_schedule_skips_streaming_and_resumes_remote_kv_request():
    scheduler = create_recompute_scheduler()
    requests = create_requests(num_requests=2)
    for request in requests:
        scheduler.add_request(request)

    requests[0].status = RequestStatus.WAITING_FOR_REMOTE_KVS
    requests[0].num_preemptions = 1
    requests[1].status = RequestStatus.WAITING_FOR_STREAMING_REQ
    requests[1].streaming_queue = deque()
    scheduler._update_waiting_for_remote_kv = MagicMock(return_value=True)

    output = scheduler.schedule()

    assert len(output.scheduled_new_reqs) == 0
    assert output.scheduled_cached_reqs.num_reqs == 1
    assert len(scheduler.running) == 1
    assert scheduler.running[0].request_id == requests[0].request_id
    assert requests[0].status == RequestStatus.RUNNING
    assert len(scheduler.waiting) == 1
    assert scheduler.waiting.peek_request().request_id == requests[1].request_id


def test_recompute_schedule_async_kv_load_keeps_request_waiting_remote():
    scheduler = create_recompute_scheduler()
    request = create_requests(num_requests=1)[0]
    scheduler.add_request(request)

    scheduler.connector = MagicMock()
    scheduler.connector.get_num_new_matched_tokens.return_value = (2, True)
    scheduler.connector.update_state_after_alloc = MagicMock()
    scheduler.connector.build_connector_meta.return_value = None
    scheduler.connector.take_events.return_value = None

    output = scheduler.schedule()

    assert output.total_num_scheduled_tokens == 0
    assert len(scheduler.running) == 0
    assert len(scheduler.waiting) == 1
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    scheduler.connector.update_state_after_alloc.assert_called_once()


def test_recompute_update_from_output_returns_stats_without_outputs():
    scheduler = create_recompute_scheduler()
    expected_stats = object()
    scheduler.make_stats = MagicMock(return_value=expected_stats)
    scheduler.finished_req_ids_dict = {}

    scheduler_output = RecomputeSchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        recomputed_reqs=[],
    )
    model_output = ModelRunnerOutput(
        req_ids=[],
        req_id_to_index={},
        sampled_token_ids=[],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    outputs = scheduler.update_from_output(scheduler_output, model_output)

    assert 0 in outputs
    assert outputs[0].scheduler_stats is expected_stats


def test_recompute_update_from_output_handles_failed_kv_load_request():
    scheduler = create_recompute_scheduler()
    scheduler.recompute_kv_load_failures = False
    request = create_requests(num_requests=1)[0]
    scheduler.finished_req_ids_dict = {request.client_index: set()}
    request.status = RequestStatus.RUNNING
    scheduler.requests[request.request_id] = request
    scheduler._handle_invalid_blocks = MagicMock(return_value={request.request_id})

    scheduler_output = RecomputeSchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={request.request_id: 1},
        total_num_scheduled_tokens=1,
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        recomputed_reqs=[],
    )
    model_output = ModelRunnerOutput(
        req_ids=[request.request_id],
        req_id_to_index={request.request_id: 0},
        sampled_token_ids=[[1000]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
        kv_connector_output=SimpleNamespace(
            invalid_block_ids=[1],
            kv_connector_stats=None,
            finished_sending=(),
            finished_recving=(),
        ),
    )

    outputs = scheduler.update_from_output(scheduler_output, model_output)

    assert request.is_finished()
    assert request.request_id not in scheduler.requests
    assert outputs[request.client_index].outputs[0].request_id == request.request_id


def test_recompute_update_from_output_advances_structured_output_and_stops_preempted_request():
    scheduler = create_recompute_scheduler()
    request = create_requests(num_requests=1, max_tokens=10)[0]
    request.num_computed_tokens = request.num_tokens
    request.status = RequestStatus.PREEMPTED
    request.structured_output_request = SimpleNamespace(
        grammar=MagicMock(accept_tokens=MagicMock(return_value=False)))
    scheduler.structured_output_manager.should_advance = MagicMock(return_value=True)
    scheduler.requests[request.request_id] = request
    scheduler.waiting.add_request(request)

    scheduler_output = RecomputeSchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={request.request_id: 1},
        total_num_scheduled_tokens=1,
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={request.request_id: []},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        recomputed_reqs=[],
    )
    model_output = ModelRunnerOutput(
        req_ids=[request.request_id],
        req_id_to_index={request.request_id: 0},
        sampled_token_ids=[[EOS_TOKEN_ID]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    scheduler.update_from_output(scheduler_output, model_output)

    request.structured_output_request.grammar.accept_tokens.assert_called_once_with(
        request.request_id, [EOS_TOKEN_ID])
    assert request.is_finished()


def test_recompute_schedule_skips_fsm_and_unknown_connector_match_requests():
    scheduler = create_recompute_scheduler()
    requests = create_requests(num_requests=2)
    for request in requests:
        scheduler.add_request(request)

    requests[0].status = RequestStatus.WAITING_FOR_FSM
    requests[0].structured_output_request = SimpleNamespace(grammar=None)
    scheduler.connector = MagicMock()
    scheduler.connector.get_num_new_matched_tokens.return_value = (None, False)
    scheduler.connector.build_connector_meta.return_value = None
    scheduler.connector.take_events.return_value = None

    output = scheduler.schedule()

    assert output.total_num_scheduled_tokens == 0
    assert len(scheduler.running) == 0
    assert len(scheduler.waiting) == 2


def test_recompute_schedule_breaks_when_chunked_prefill_disabled_and_budget_small():
    scheduler = create_recompute_scheduler()
    scheduler.scheduler_config.enable_chunked_prefill = False
    scheduler.max_num_scheduled_tokens = 1
    request = create_requests(num_requests=1, num_tokens=10)[0]
    scheduler.add_request(request)

    output = scheduler.schedule()

    assert output.total_num_scheduled_tokens == 0
    assert len(output.scheduled_new_reqs) == 0
    assert len(scheduler.waiting) == 1


def test_recompute_update_from_output_handles_pooling_logprobs_nans_and_finished_ids():
    scheduler = create_recompute_scheduler()
    request = create_requests(num_requests=1, max_tokens=10)[0]
    request.pooling_params = MagicMock()
    request.num_computed_tokens = request.num_tokens
    request.status = RequestStatus.RUNNING
    request.sampling_params.logprobs = 1
    scheduler.requests[request.request_id] = request
    scheduler.running.append(request)
    scheduler.finished_req_ids_dict = {request.client_index: {"done-before-send"}}

    fake_logprobs = MagicMock()
    fake_logprobs.slice_request.return_value = "slice"
    scheduler_output = RecomputeSchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={request.request_id: 1},
        total_num_scheduled_tokens=1,
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        recomputed_reqs=[],
    )
    model_output = ModelRunnerOutput(
        req_ids=[request.request_id],
        req_id_to_index={request.request_id: 0},
        sampled_token_ids=[[]],
        logprobs=fake_logprobs,
        prompt_logprobs_dict={},
        pooler_output=[torch.tensor([1.0])],
        num_nans_in_logits={request.request_id: 7},
    )

    outputs = scheduler.update_from_output(scheduler_output, model_output)

    assert request.is_finished()
    assert {"done-before-send", request.request_id}.issubset(
        outputs[request.client_index].finished_requests)
    assert outputs[request.client_index].outputs[0].pooling_output is not None
    assert outputs[request.client_index].outputs[0].num_nans_in_logits == 7


def test_recompute_update_from_output_slices_logprobs_for_generated_tokens():
    scheduler = create_recompute_scheduler()
    request = create_requests(num_requests=1, max_tokens=10)[0]
    request.num_computed_tokens = request.num_tokens
    request.status = RequestStatus.RUNNING
    request.sampling_params.logprobs = 1
    scheduler.requests[request.request_id] = request
    scheduler.running.append(request)
    scheduler.finished_req_ids_dict = {}

    fake_logprobs = MagicMock()
    fake_logprobs.slice_request.return_value = "slice"
    scheduler_output = RecomputeSchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={request.request_id: 1},
        total_num_scheduled_tokens=1,
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={request.request_id: []},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        recomputed_reqs=[],
    )
    model_output = ModelRunnerOutput(
        req_ids=[request.request_id],
        req_id_to_index={request.request_id: 0},
        sampled_token_ids=[[123]],
        logprobs=fake_logprobs,
        prompt_logprobs_dict={},
        pooler_output=[],
        num_nans_in_logits={request.request_id: 3},
    )

    outputs = scheduler.update_from_output(scheduler_output, model_output)

    fake_logprobs.slice_request.assert_called_once_with(0, 1)
    assert outputs[request.client_index].outputs[0].new_logprobs == "slice"
    assert outputs[request.client_index].outputs[0].num_nans_in_logits == 3


def test_recompute_schedule_skips_extra_step_for_output_placeholders():
    scheduler = create_recompute_scheduler()
    running_req, waiting_req = create_requests(num_requests=2, max_tokens=2)
    running_req.status = RequestStatus.RUNNING
    running_req.num_output_placeholders = 2
    running_req.num_computed_tokens = running_req.num_prompt_tokens + running_req.max_tokens
    scheduler.requests[running_req.request_id] = running_req
    scheduler.running.append(running_req)
    scheduler.add_request(waiting_req)

    output = scheduler.schedule()

    assert running_req.request_id not in output.num_scheduled_tokens
    assert waiting_req.request_id in output.num_scheduled_tokens


def test_recompute_schedule_skips_waiting_remote_request_when_not_ready():
    scheduler = create_recompute_scheduler()
    request = create_requests(num_requests=1)[0]
    scheduler.add_request(request)
    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    scheduler._update_waiting_for_remote_kv = MagicMock(return_value=False)

    output = scheduler.schedule()

    assert output.total_num_scheduled_tokens == 0
    assert len(scheduler.running) == 0
    assert len(scheduler.waiting) == 1
    assert scheduler.waiting.peek_request().request_id == request.request_id


def test_recompute_schedule_mamba_aligned_zero_tokens_skips_running_request():
    scheduler = create_recompute_scheduler()
    running_req, waiting_req = create_requests(num_requests=2)
    running_req.status = RequestStatus.RUNNING
    scheduler.requests[running_req.request_id] = running_req
    scheduler.running.append(running_req)
    scheduler.add_request(waiting_req)
    scheduler.need_mamba_block_aligned_split = True
    scheduler._mamba_block_aligned_split = MagicMock(
        side_effect=[0, waiting_req.num_tokens])

    output = scheduler.schedule()

    assert running_req.request_id not in output.num_scheduled_tokens
    assert waiting_req.request_id in output.num_scheduled_tokens


def test_recompute_schedule_frees_encoder_cache_when_waiting_allocation_fails():
    scheduler = create_recompute_scheduler()
    request = create_requests(
        num_requests=1,
        mm_positions=[[PlaceholderRange(offset=0, length=10)]],
    )[0]
    scheduler.add_request(request)
    scheduler.kv_cache_manager.allocate_slots = MagicMock(return_value=None)
    scheduler.encoder_cache_manager.free = MagicMock()

    output = scheduler.schedule()

    assert output.total_num_scheduled_tokens == 0
    scheduler.encoder_cache_manager.free.assert_called_once_with(request)


def test_recompute_update_from_output_skips_missing_and_finished_requests():
    scheduler = create_recompute_scheduler()
    request = create_requests(num_requests=1)[0]
    scheduler.requests[request.request_id] = request
    scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
    scheduler.finished_req_ids_dict = {}

    scheduler_output = RecomputeSchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={request.request_id: 1, "missing": 1},
        total_num_scheduled_tokens=2,
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        recomputed_reqs=[],
    )
    model_output = ModelRunnerOutput(
        req_ids=[request.request_id, "missing"],
        req_id_to_index={request.request_id: 0, "missing": 1},
        sampled_token_ids=[[1], [2]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    outputs = scheduler.update_from_output(scheduler_output, model_output)

    assert 0 in outputs
    assert outputs[0].outputs == []


def test_recompute_schedule_waiting_fsm_request_with_grammar_runs():
    scheduler = create_recompute_scheduler()
    request = create_requests(num_requests=1)[0]
    scheduler.add_request(request)
    request.status = RequestStatus.WAITING_FOR_FSM
    request.structured_output_request = SimpleNamespace(grammar=object())

    output = scheduler.schedule()

    assert len(output.scheduled_new_reqs) == 1
    assert output.scheduled_new_reqs[0].req_id == request.request_id
    assert request.status == RequestStatus.RUNNING


def test_recompute_schedule_builds_ec_connector_metadata():
    scheduler = create_recompute_scheduler()
    request = create_requests(num_requests=1)[0]
    scheduler.add_request(request)
    scheduler.ec_connector = MagicMock()
    scheduler.ec_connector.build_connector_meta.return_value = "ec-meta"

    output = scheduler.schedule()

    assert output.ec_connector_metadata == "ec-meta"


def test_recompute_update_from_output_aggregates_kv_connector_stats():
    scheduler = create_recompute_scheduler()
    scheduler.finished_req_ids_dict = {}
    scheduler.make_stats = MagicMock(return_value="stats")
    scheduler.connector = MagicMock()
    scheduler.connector.get_kv_connector_stats.return_value = "connector-stats"
    scheduler.connector.take_events.return_value = None

    kv_connector_stats = MagicMock()
    kv_connector_stats.aggregate.return_value = "agg-stats"
    scheduler_output = RecomputeSchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        recomputed_reqs=[],
    )
    model_output = ModelRunnerOutput(
        req_ids=[],
        req_id_to_index={},
        sampled_token_ids=[],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
        kv_connector_output=SimpleNamespace(
            kv_connector_stats=kv_connector_stats,
            invalid_block_ids=None,
            finished_sending=(),
            finished_recving=(),
        ),
    )

    outputs = scheduler.update_from_output(scheduler_output, model_output)

    kv_connector_stats.aggregate.assert_called_once_with("connector-stats")
    scheduler.make_stats.assert_called_once()
    assert outputs[0].scheduler_stats == "stats"


def test_recompute_schedule_remote_kv_ready_new_request_becomes_waiting_and_runs():
    scheduler = create_recompute_scheduler()
    request = create_requests(num_requests=1)[0]
    scheduler.add_request(request)
    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    request.num_preemptions = 0
    scheduler._update_waiting_for_remote_kv = MagicMock(return_value=True)

    output = scheduler.schedule()

    assert len(output.scheduled_new_reqs) == 1
    assert output.scheduled_new_reqs[0].req_id == request.request_id


def test_recompute_schedule_priority_preemption_rolls_back_scheduled_running_request():
    scheduler = create_recompute_scheduler()
    req0, req1 = create_requests(num_requests=2)
    for index, request in enumerate((req0, req1)):
        request.status = RequestStatus.RUNNING
        request.priority = 10 - index * 10
        request.arrival_time = index
        scheduler.requests[request.request_id] = request
        scheduler.running.append(request)

    block0 = MagicMock()
    block0.get_block_ids.return_value = ([1],)
    block1 = MagicMock()
    block1.get_block_ids.return_value = ([2],)
    scheduler.policy = SchedulingPolicy.PRIORITY
    scheduler.kv_cache_manager.allocate_slots = MagicMock(
        side_effect=[block0, None, block1])

    output = scheduler.schedule()

    assert req0.request_id in output.preempted_req_ids
    assert req0.request_id not in output.num_scheduled_tokens
    assert req1.request_id in output.num_scheduled_tokens
    assert req0.status == RequestStatus.PREEMPTED


def test_recompute_schedule_records_connector_prefix_cache_stats():
    scheduler = create_recompute_scheduler()
    request = create_requests(num_requests=1, num_tokens=10)[0]
    request.num_preemptions = 1
    scheduler.add_request(request)
    scheduler.connector = MagicMock()
    scheduler.connector.get_num_new_matched_tokens.return_value = (2, False)
    scheduler.connector.update_state_after_alloc = MagicMock()
    scheduler.connector.build_connector_meta.return_value = None
    scheduler.connector.take_events.return_value = None
    scheduler.connector_prefix_cache_stats = MagicMock()

    output = scheduler.schedule()

    assert output.num_scheduled_tokens[request.request_id] > 0
    scheduler.connector_prefix_cache_stats.record.assert_called_once_with(
        num_tokens=request.num_tokens,
        num_hits=2,
        preempted=True,
    )


def test_recompute_schedule_waiting_streaming_request_is_skipped():
    scheduler = create_recompute_scheduler()
    request = create_requests(num_requests=1)[0]
    scheduler.add_request(request)
    request.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    request.streaming_queue = deque()

    output = scheduler.schedule()

    assert output.total_num_scheduled_tokens == 0
    assert len(scheduler.waiting) == 1
    assert scheduler.waiting.peek_request().request_id == request.request_id


def test_recompute_schedule_use_v2_model_runner_promotes_resumed_request_to_new():
    scheduler = create_recompute_scheduler()
    request = create_requests(num_requests=1)[0]
    scheduler.add_request(request)
    request.status = RequestStatus.PREEMPTED
    scheduler.use_v2_model_runner = True

    output = scheduler.schedule()

    assert len(output.scheduled_new_reqs) == 1
    assert output.scheduled_new_reqs[0].req_id == request.request_id
    assert output.scheduled_cached_reqs.num_reqs == 0


def test_recompute_schedule_skips_waiting_request_when_lora_budget_full():
    scheduler = create_recompute_scheduler()
    running_req, waiting_req = create_requests(num_requests=2)
    running_req.status = RequestStatus.RUNNING
    running_req.lora_request = SimpleNamespace(lora_int_id=1)
    waiting_req.lora_request = SimpleNamespace(lora_int_id=2)
    scheduler.requests[running_req.request_id] = running_req
    scheduler.running.append(running_req)
    scheduler.add_request(waiting_req)
    scheduler.lora_config = SimpleNamespace(max_loras=1)

    output = scheduler.schedule()

    assert running_req.request_id in output.num_scheduled_tokens
    assert waiting_req.request_id not in output.num_scheduled_tokens
    assert len(scheduler.waiting) == 1


def test_recompute_schedule_running_request_trims_spec_tokens_and_updates_ec_connector():
    scheduler = create_recompute_scheduler()
    scheduler.scheduler_config.long_prefill_token_threshold = 2
    request = create_requests(
        num_requests=1,
        num_tokens=10,
        mm_positions=[[PlaceholderRange(offset=0, length=10)]],
    )[0]
    request.status = RequestStatus.RUNNING
    request.spec_token_ids = [11, 12, 13]
    request.num_computed_tokens = request.num_tokens
    request.lora_request = SimpleNamespace(lora_int_id=4)
    scheduler.requests[request.request_id] = request
    scheduler.running.append(request)
    scheduler.ec_connector = MagicMock()
    scheduler._try_schedule_encoder_inputs = MagicMock(
        return_value=([0], 2, scheduler.max_num_encoder_input_tokens - 10, [0]))

    output = scheduler.schedule()

    assert output.num_scheduled_tokens[request.request_id] == 2
    assert output.scheduled_spec_decode_tokens[request.request_id] == [11, 12]
    assert output.scheduled_encoder_inputs[request.request_id] == [0]
    scheduler.ec_connector.update_state_after_alloc.assert_called_once_with(
        request, 0)
    assert request.spec_token_ids == []


def test_recompute_schedule_resumed_mtp_request_tracks_spec_tokens_and_external_encoder_load():
    scheduler = create_recompute_scheduler()
    scheduler.is_mtp_kv_consumer = True
    scheduler.num_spec_tokens = 2
    request = create_requests(
        num_requests=1,
        num_tokens=5,
        mm_positions=[[PlaceholderRange(offset=0, length=10)]],
    )[0]
    scheduler.add_request(request)
    request.status = RequestStatus.PREEMPTED
    request.num_computed_tokens = request.num_tokens
    request.spec_token_ids = [21, 22, 23]
    scheduler.ec_connector = MagicMock()
    scheduler._try_schedule_encoder_inputs = MagicMock(
        return_value=([0], 3, scheduler.max_num_encoder_input_tokens - 10, [0]))

    output = scheduler.schedule()

    assert len(output.scheduled_new_reqs) == 0
    assert output.scheduled_cached_reqs.num_reqs == 1
    assert output.scheduled_spec_decode_tokens[request.request_id] == [21, 22, 23]
    scheduler.ec_connector.update_state_after_alloc.assert_called_once_with(
        request, 0)
    assert request.status == RequestStatus.RUNNING
    assert request.spec_token_ids == []


def test_recompute_update_from_output_returns_recomputed_output_and_perf_stats():
    scheduler = create_recompute_scheduler()
    scheduler.finished_req_ids_dict = {}
    scheduler.make_stats = MagicMock(return_value="stats")
    scheduler.perf_metrics = MagicMock()
    scheduler.perf_metrics.is_enabled.return_value = True
    scheduler.perf_metrics.get_step_perf_stats_per_gpu.return_value = "perf"
    scheduler.connector = MagicMock()
    scheduler.connector.get_kv_connector_stats.return_value = None
    scheduler.connector.take_events.return_value = ["connector-event"]
    scheduler.kv_cache_manager.take_events = MagicMock(return_value=None)
    scheduler.kv_event_publisher.publish = MagicMock()

    scheduler_output = RecomputeSchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        recomputed_reqs=[
            SimpleNamespace(request_id="recomp", output_token_ids=[1], client_index=5)
        ],
    )
    model_output = ModelRunnerOutput(
        req_ids=[],
        req_id_to_index={},
        sampled_token_ids=[],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    outputs = scheduler.update_from_output(scheduler_output, model_output)

    assert outputs[5].outputs[0].request_id == "recomp"
    assert outputs[5].outputs[0].stop_reason == "recomputed"
    assert outputs[5].scheduler_stats == "stats"
    published_batch = scheduler.kv_event_publisher.publish.call_args[0][0]
    assert published_batch.events == ["connector-event"]


def test_recompute_update_from_output_adds_finished_requests_for_absent_client():
    scheduler = create_recompute_scheduler()
    scheduler.make_stats = MagicMock(return_value=None)
    scheduler.finished_req_ids_dict = {9: {"finished-only"}}

    scheduler_output = RecomputeSchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        recomputed_reqs=[],
    )
    model_output = ModelRunnerOutput(
        req_ids=[],
        req_id_to_index={},
        sampled_token_ids=[],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    outputs = scheduler.update_from_output(scheduler_output, model_output)

    assert outputs[9].finished_requests == {"finished-only"}


def test_recompute_schedule_breaks_when_running_queue_is_full():
    scheduler = create_recompute_scheduler()
    running_req, waiting_req = create_requests(num_requests=2)
    running_req.status = RequestStatus.RUNNING
    scheduler.requests[running_req.request_id] = running_req
    scheduler.running.append(running_req)
    scheduler.add_request(waiting_req)
    scheduler.max_num_running_reqs = 1

    output = scheduler.schedule()

    assert waiting_req.request_id not in output.num_scheduled_tokens
    assert len(scheduler.waiting) == 1


def test_recompute_schedule_waiting_request_applies_long_prefill_threshold():
    scheduler = create_recompute_scheduler()
    scheduler.scheduler_config.long_prefill_token_threshold = 2
    request = create_requests(num_requests=1, num_tokens=10)[0]
    scheduler.add_request(request)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens[request.request_id] == 2


def test_recompute_schedule_waiting_encoder_request_breaks_when_encoder_schedule_zeroes_tokens():
    scheduler = create_recompute_scheduler()
    request = create_requests(
        num_requests=1,
        num_tokens=10,
        mm_positions=[[PlaceholderRange(offset=0, length=10)]],
    )[0]
    scheduler.add_request(request)
    scheduler._try_schedule_encoder_inputs = MagicMock(
        return_value=([0], 0, scheduler.max_num_encoder_input_tokens, []))

    output = scheduler.schedule()

    assert output.total_num_scheduled_tokens == 0
    assert len(scheduler.waiting) == 1


def test_recompute_schedule_waiting_mtp_request_clears_spec_tokens_without_recording_them():
    scheduler = create_recompute_scheduler()
    scheduler.is_mtp_kv_consumer = True
    request = create_requests(num_requests=1, num_tokens=5)[0]
    scheduler.add_request(request)
    request.status = RequestStatus.PREEMPTED
    request.num_computed_tokens = request.num_tokens
    request.num_output_placeholders = 4
    request.spec_token_ids = [41, 42, 43]

    output = scheduler.schedule()

    assert request.request_id not in output.scheduled_spec_decode_tokens
    assert request.spec_token_ids == []


def test_recompute_schedule_waiting_mtp_request_truncates_spec_tokens():
    scheduler = create_recompute_scheduler()
    scheduler.is_mtp_kv_consumer = True
    scheduler.scheduler_config.long_prefill_token_threshold = 2
    request = create_requests(num_requests=1, num_tokens=5)[0]
    scheduler.add_request(request)
    request.status = RequestStatus.PREEMPTED
    request.num_computed_tokens = request.num_tokens
    request.spec_token_ids = [51, 52, 53, 54]

    output = scheduler.schedule()

    assert output.scheduled_spec_decode_tokens[request.request_id] == [51, 52]
    assert request.spec_token_ids == []


def test_recompute_schedule_preserves_non_negative_cached_token_count():
    scheduler = create_recompute_scheduler()
    request = create_requests(num_requests=1)[0]
    scheduler.add_request(request)
    request.num_cached_tokens = 0

    scheduler.schedule()

    assert request.num_cached_tokens == 0


def test_recompute_update_from_output_with_no_visible_output_returns_nothing():
    scheduler = create_recompute_scheduler()
    request = create_requests(num_requests=1)[0]
    request.status = RequestStatus.RUNNING
    scheduler.requests[request.request_id] = request
    scheduler.running.append(request)
    scheduler.finished_req_ids_dict = {}
    scheduler.make_stats = MagicMock(return_value=None)

    scheduler_output = RecomputeSchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={request.request_id: 1},
        total_num_scheduled_tokens=1,
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        recomputed_reqs=[],
    )
    model_output = ModelRunnerOutput(
        req_ids=[request.request_id],
        req_id_to_index={request.request_id: 0},
        sampled_token_ids=[[]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[None],
    )

    outputs = scheduler.update_from_output(scheduler_output, model_output)

    assert outputs == {}


def test_recompute_update_from_output_merges_cache_and_connector_events():
    scheduler = create_recompute_scheduler()
    scheduler.finished_req_ids_dict = {}
    scheduler.make_stats = MagicMock(return_value=None)
    scheduler.connector = MagicMock()
    scheduler.connector.take_events.return_value = ["connector"]
    scheduler.kv_cache_manager.take_events = MagicMock(return_value=["kv"])
    scheduler.kv_event_publisher.publish = MagicMock()

    scheduler_output = RecomputeSchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        recomputed_reqs=[],
    )
    model_output = ModelRunnerOutput(
        req_ids=[],
        req_id_to_index={},
        sampled_token_ids=[],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    scheduler.update_from_output(scheduler_output, model_output)

    published_batch = scheduler.kv_event_publisher.publish.call_args[0][0]
    assert published_batch.events == ["kv", "connector"]
