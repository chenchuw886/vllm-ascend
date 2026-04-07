from contextlib import nullcontext
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata, KVConnectorRole

from vllm_ascend.distributed.kv_transfer.kv_pool.cpu_offload import cpu_offload_connector as connector_module


def make_vllm_config(*, enable_prefix_caching=True, use_mla=False, cache_dtype="auto", kv_transfer_config=None):
    if kv_transfer_config is None:
        kv_transfer_config = SimpleNamespace(get_from_extra_config=lambda key, default: default)
    return SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=enable_prefix_caching, block_size=4, cache_dtype=cache_dtype),
        model_config=SimpleNamespace(
            use_mla=use_mla,
            dtype="model-dtype",
            hf_text_config=SimpleNamespace(kv_lora_rank=8, qk_rope_head_dim=16),
            hf_config=SimpleNamespace(),
        ),
        parallel_config=SimpleNamespace(data_parallel_rank=0),
        kv_transfer_config=kv_transfer_config,
    )


def make_sched_output():
    return SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(req_id="new-1", num_computed_tokens=2, block_ids=[[10, 11]]), SimpleNamespace(req_id="new-2", num_computed_tokens=1, block_ids=[None])],
        scheduled_cached_reqs=SimpleNamespace(req_ids=["cached-1"], num_computed_tokens=[5], new_block_ids=[[20, 21]]),
        num_scheduled_tokens={"new-1": 3, "new-2": 4, "cached-1": 2},
    )


def test_req_meta_update_merges_fields():
    req_meta = connector_module.ReqMeta([1], [2], 3, 4, 5, 6)
    req_meta.update(connector_module.ReqMeta([7], [8], 9, 10, 11, 12))

    assert req_meta == connector_module.ReqMeta([1, 7], [2, 8], 9, 10, 11, 12)


def test_connector_init_and_delegation(monkeypatch):
    scheduler = SimpleNamespace(
        get_num_new_matched_tokens=lambda request, num: (7, True),
        update_state_after_alloc=lambda request: "updated",
        build_connector_meta=lambda output: "metadata",
        request_finished=lambda request: "finished",
    )
    worker = SimpleNamespace(
        bind_connector_metadata=lambda metadata: setattr(worker, "bound", metadata),
        clear_connector_metadata=lambda: setattr(worker, "cleared", True),
        register_kv_caches=lambda kv: setattr(worker, "kv", kv),
        start_load_kv=lambda: setattr(worker, "started", True),
        wait_for_layer_load=lambda: setattr(worker, "waited", True),
        get_finished=lambda: {"done"},
    )

    monkeypatch.setattr(connector_module, "CPUOffloadingConnectorScheduler", lambda _cfg: scheduler)
    monkeypatch.setattr(connector_module, "CPUOffloadingConnectorWorker", lambda _cfg: worker)

    disabled = connector_module.CPUOffloadingConnector(make_vllm_config(enable_prefix_caching=False), KVConnectorRole.SCHEDULER)
    assert disabled.connector_scheduler is None
    assert disabled.connector_worker is None
    assert disabled.get_num_new_matched_tokens(SimpleNamespace(), 1) == (0, False)
    assert isinstance(disabled.build_connector_meta(SimpleNamespace()), KVConnectorMetadata)

    scheduler_connector = connector_module.CPUOffloadingConnector(make_vllm_config(), KVConnectorRole.SCHEDULER)
    assert scheduler_connector.connector_scheduler is scheduler
    assert scheduler_connector.connector_worker is None
    assert scheduler_connector.get_num_new_matched_tokens(SimpleNamespace(), 1) == (7, True)
    assert scheduler_connector.update_state_after_alloc(SimpleNamespace(), None, 0) == "updated"
    assert scheduler_connector.build_connector_meta(SimpleNamespace()) == "metadata"
    assert scheduler_connector.request_finished(SimpleNamespace(), []) == (True, None)

    worker_connector = connector_module.CPUOffloadingConnector(make_vllm_config(), KVConnectorRole.WORKER)
    metadata = connector_module.CPUOffloadingConnectorMetadata(requests={}, finished_req_ids=set())
    worker_connector.bind_connector_metadata(metadata)
    worker_connector.clear_connector_metadata()
    worker_connector.register_kv_caches({"layer": "cache"})
    worker_connector.start_load_kv(SimpleNamespace())
    worker_connector.wait_for_layer_load("layer")
    assert worker.bound is metadata
    assert worker.cleared is True
    assert worker.kv == {"layer": "cache"}
    assert worker.started is True
    assert worker.waited is True
    assert worker_connector.get_finished(set()) == ({"done"}, None)


def test_connector_bind_connector_metadata_validates_type(monkeypatch):
    worker = SimpleNamespace(bind_connector_metadata=lambda metadata: metadata)
    monkeypatch.setattr(connector_module, "CPUOffloadingConnectorWorker", lambda _cfg: worker)
    connector = connector_module.CPUOffloadingConnector(make_vllm_config(), KVConnectorRole.WORKER)

    with pytest.raises(AssertionError):
        connector.bind_connector_metadata(KVConnectorMetadata())


def test_connector_noop_paths_without_worker_or_scheduler():
    connector = connector_module.CPUOffloadingConnector(make_vllm_config(enable_prefix_caching=False), KVConnectorRole.SCHEDULER)
    metadata = connector_module.CPUOffloadingConnectorMetadata(requests={}, finished_req_ids=set())

    assert connector.bind_connector_metadata(metadata) is None
    assert connector.register_kv_caches({"layer": "cache"}) is None
    assert connector.start_load_kv(SimpleNamespace()) is None
    assert connector.wait_for_layer_load("layer") is None
    assert connector.save_kv_layer("layer", None, None) is None
    assert connector.wait_for_save() is None
    assert connector.update_state_after_alloc(SimpleNamespace(), None, 0) is None
    assert connector.request_finished(SimpleNamespace(), []) == (True, None)


def test_connector_init_with_unhandled_role_leaves_no_delegate_objects():
    connector = connector_module.CPUOffloadingConnector(make_vllm_config(), object())

    assert not hasattr(connector, "connector_scheduler")
    assert not hasattr(connector, "connector_worker")


def test_scheduler_init_uses_threshold_from_config(monkeypatch):
    rpc_client = SimpleNamespace(call=lambda func, *args: None)
    monkeypatch.setattr(connector_module.MetadataServer, "ZMQRPCClient", lambda: rpc_client)

    scheduler = connector_module.CPUOffloadingConnectorScheduler(
        make_vllm_config(kv_transfer_config=SimpleNamespace(get_from_extra_config=lambda key, default: 6))
    )

    assert scheduler.swap_in_threshold == 6
    assert scheduler.zmq_rpc_client is rpc_client


def test_scheduler_init_uses_default_threshold_when_no_config(monkeypatch):
    rpc_client = SimpleNamespace(call=lambda func, *args: None)
    monkeypatch.setattr(connector_module.MetadataServer, "ZMQRPCClient", lambda: rpc_client)

    scheduler = connector_module.CPUOffloadingConnectorScheduler(
        SimpleNamespace(
            cache_config=SimpleNamespace(block_size=4),
            model_config=SimpleNamespace(use_mla=False),
            kv_transfer_config=None,
        )
    )

    assert scheduler.swap_in_threshold == 0


def test_scheduler_get_num_new_matched_tokens_above_and_below_threshold(monkeypatch):
    responses = iter([(10, True), (4, False)])

    def rpc_call(func, *args):
        if func == "post_init":
            return None
        return next(responses)

    rpc_client = SimpleNamespace(call=rpc_call)
    monkeypatch.setattr(connector_module.MetadataServer, "ZMQRPCClient", lambda: rpc_client)
    scheduler = connector_module.CPUOffloadingConnectorScheduler(
        make_vllm_config(kv_transfer_config=SimpleNamespace(get_from_extra_config=lambda key, default: 3))
    )
    request = SimpleNamespace(request_id="req-1", get_hash_new_full_blocks="keep")

    assert scheduler.get_num_new_matched_tokens(request, 5) == (5, True)
    assert scheduler.num_gpu_computed_tokens["req-1"] == 5
    assert scheduler.num_cpu_computed_tokens["req-1"] == 10
    assert request.get_hash_new_full_blocks == "keep"

    request2 = SimpleNamespace(request_id="req-2", get_hash_new_full_blocks="keep-2")
    assert scheduler.get_num_new_matched_tokens(request2, 3) == (0, False)


def test_scheduler_update_build_meta_and_request_finished(monkeypatch):
    rpc_calls = []

    def rpc_call(func, *args):
        rpc_calls.append((func, args))
        if func == "allocate_slots":
            return {"new-1": [100], "cached-1": [200, 201]}
        return None

    rpc_client = SimpleNamespace(call=rpc_call)
    monkeypatch.setattr(connector_module.MetadataServer, "ZMQRPCClient", lambda: rpc_client)
    scheduler = connector_module.CPUOffloadingConnectorScheduler(make_vllm_config())
    scheduler.num_gpu_computed_tokens = {"new-1": 2, "new-2": 1, "cached-1": 5, "stale": 9}
    scheduler.num_cpu_computed_tokens = {"new-1": 6, "new-2": 1, "cached-1": 5, "stale": 9}
    scheduler.allocated_req_ids = {"new-2"}
    scheduler.finished_req_ids = ["finished-1"]
    scheduler.update_state_after_alloc(SimpleNamespace(request_id="cached-1"))

    metadata = scheduler.build_connector_meta(make_sched_output())

    assert isinstance(metadata, connector_module.CPUOffloadingConnectorMetadata)
    assert metadata.finished_req_ids == {"finished-1"}
    assert metadata.requests["new-1"] == connector_module.ReqMeta([10, 11], [100], 3, 2, 2, 6)
    assert metadata.requests["new-2"] == connector_module.ReqMeta([], [], 4, 1, 1, 1)
    assert metadata.requests["cached-1"] == connector_module.ReqMeta([20, 21], [200, 201], 2, 5, 5, 5)
    allocate_call = next(call for call in rpc_calls if call[0] == "allocate_slots")
    assert allocate_call[1][0] == {"new-1": 5, "new-2": 5, "cached-1": 7}
    assert allocate_call[1][1] == {"stale"}
    assert scheduler.num_gpu_computed_tokens == {}
    assert scheduler.num_cpu_computed_tokens == {}
    assert scheduler.allocated_req_ids == set()
    assert scheduler.finished_req_ids == []

    request = SimpleNamespace(request_id="req-finished", get_hash_new_full_blocks="keep")
    scheduler.request_finished(request)
    assert scheduler.finished_req_ids == ["req-finished"]
    assert rpc_calls[-1][0] == "record_request_cache_and_free_slots"
    assert request.get_hash_new_full_blocks == "keep"


def test_worker_init_metadata_server_and_wait(monkeypatch):
    call_log = []

    def rpc_call(func, *args):
        call_log.append(func)
        if func == "ready":
            if call_log.count("ready") == 1:
                raise RuntimeError("not-ready")
            return True
        return None

    thread_targets = []

    class FakeThread:
        def __init__(self, target=None, args=()):
            self.target = target
            self.args = args
            self.daemon = False
            self.started = False
            thread_targets.append(self)

        def start(self):
            self.started = True

    monkeypatch.setattr(connector_module, "get_pp_group", lambda: SimpleNamespace(rank_in_group=0))
    monkeypatch.setattr(connector_module, "get_tp_group", lambda: SimpleNamespace(rank_in_group=0, world_size=2))
    monkeypatch.setattr(connector_module.torch.npu, "Stream", lambda: SimpleNamespace(synchronize=MagicMock()))
    monkeypatch.setattr(connector_module.MetadataServer, "ZMQRPCClient", lambda: SimpleNamespace(call=rpc_call))
    monkeypatch.setattr(connector_module.threading, "Thread", FakeThread)
    monkeypatch.setattr(connector_module.time, "sleep", lambda _secs: None)
    monkeypatch.setattr(connector_module, "VllmConfig", lambda: SimpleNamespace())

    worker = connector_module.CPUOffloadingConnectorWorker(make_vllm_config())

    assert worker.pp_rank == 0
    assert worker.tp_rank == 0
    assert worker.tp_world_size == 2
    assert call_log.count("ready") == 2
    assert len(thread_targets) == 2
    assert thread_targets[0].target == worker._save_listener
    assert thread_targets[1].target == connector_module.MetadataServerProc.run_metadata_server
    assert thread_targets[0].started is True
    assert thread_targets[1].started is True


def test_worker_init_without_metadata_server_on_nonzero_rank(monkeypatch):
    ready_calls = []
    thread_targets = []

    class FakeThread:
        def __init__(self, target=None, args=()):
            self.target = target
            self.args = args
            self.daemon = False
            self.started = False
            thread_targets.append(self)

        def start(self):
            self.started = True

    def rpc_call(func, *args):
        if func == "ready":
            ready_calls.append(func)
            return len(ready_calls) > 1
        return None

    monkeypatch.setattr(connector_module, "get_pp_group", lambda: SimpleNamespace(rank_in_group=1))
    monkeypatch.setattr(connector_module, "get_tp_group", lambda: SimpleNamespace(rank_in_group=1, world_size=2))
    monkeypatch.setattr(connector_module.torch.npu, "Stream", lambda: SimpleNamespace(synchronize=MagicMock()))
    monkeypatch.setattr(connector_module.MetadataServer, "ZMQRPCClient", lambda: SimpleNamespace(call=rpc_call))
    monkeypatch.setattr(connector_module.threading, "Thread", FakeThread)
    monkeypatch.setattr(connector_module, "VllmConfig", lambda: SimpleNamespace())

    worker = connector_module.CPUOffloadingConnectorWorker(make_vllm_config())

    assert worker.pp_rank == 1
    assert worker.tp_rank == 1
    assert ready_calls == ["ready", "ready"]
    assert len(thread_targets) == 1
    assert thread_targets[0].target == worker._save_listener


def test_worker_bind_clear_and_register(monkeypatch):
    worker = object.__new__(connector_module.CPUOffloadingConnectorWorker)
    worker.requests = {"existing": connector_module.ReqMeta([1], [10, 11], 2, 8, 4, 8)}
    worker.block_size = 4
    worker.load_block_mapping = []
    worker.save_input_queue = SimpleNamespace(items=[], put=lambda item: worker.save_input_queue.items.append(item))
    worker.vllm_config = make_vllm_config(use_mla=False)
    worker.pp_rank = 1
    worker.tp_rank = 2
    worker.zmq_rpc_client = SimpleNamespace(call=lambda *args: {"layer-a": ["cpu-a"], "layer-b": ["cpu-b"]})

    metadata = connector_module.CPUOffloadingConnectorMetadata(
        requests={
            "existing": connector_module.ReqMeta([3, 4], [12, 13], 1, 12, 4, 12),
            "new": connector_module.ReqMeta([5, 6, 7], [20, 21, 22], 2, 12, 4, 12),
        },
        finished_req_ids={"existing", "missing"},
    )

    worker.bind_connector_metadata(metadata)

    assert worker.requests["existing"].gpu_block_ids == [1, 3, 4]
    assert worker.requests["new"].gpu_block_ids == [5, 6, 7]
    assert worker.load_block_mapping == [(11, 3), (12, 4), (21, 6), (22, 7)]
    assert worker.save_input_queue.items == [("existing", worker.requests["existing"])]

    worker.clear_connector_metadata()
    assert worker.load_block_mapping == []

    monkeypatch.setattr(connector_module, "get_kv_cache_spec", lambda _cfg: {"layer-a": "spec-a", "layer-b": "spec-b"})
    worker.register_kv_caches({"layer-a": ["gpu-a"], "layer-b": ["gpu-b"]})
    assert worker.gpu_kv_caches == {"layer-a": ["gpu-a"], "layer-b": ["gpu-b"]}
    assert worker.cpu_kv_caches == [["cpu-a"], ["cpu-b"]]


def test_worker_register_kv_caches_with_mla(monkeypatch):
    captured = {}
    worker = object.__new__(connector_module.CPUOffloadingConnectorWorker)
    worker.vllm_config = make_vllm_config(use_mla=True)
    worker.pp_rank = 0
    worker.tp_rank = 0
    worker.zmq_rpc_client = SimpleNamespace(
        call=lambda func, pp_rank, tp_rank, spec, mla_config: captured.update({"mla_config": mla_config}) or {"x": [1]}
    )
    monkeypatch.setattr(connector_module, "get_kv_cache_spec", lambda _cfg: {"layer": "spec"})

    worker.register_kv_caches({"layer": ["gpu"]})

    assert captured["mla_config"] == connector_module.MLAConfig(8, 16)
    assert worker.cpu_kv_caches == [[1]]


def test_worker_load_flow_and_load_kv_layer(monkeypatch):
    copy_calls = []

    class FakeTarget:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, idx):
            return self

        def copy_(self, source, non_blocking=False):
            copy_calls.append((self.name, source.name, non_blocking))

    worker = object.__new__(connector_module.CPUOffloadingConnectorWorker)
    worker.gpu_kv_caches = {"layer-1": [FakeTarget("gpu-1a"), FakeTarget("gpu-1b")], "layer-2": [FakeTarget("gpu-2a")]} 
    worker.cpu_kv_caches = [[FakeTarget("cpu-1a"), FakeTarget("cpu-1b")], [FakeTarget("cpu-2a")]]
    worker.load_block_mapping = [(1, 2)]
    worker.load_stream = SimpleNamespace(synchronize=MagicMock())
    worker.current_layer = 0
    monkeypatch.setattr(connector_module.torch.npu, "stream", lambda _stream: nullcontext())

    worker.start_load_kv()
    assert worker.current_layer == 0
    assert copy_calls[:2] == [("gpu-1a", "cpu-1a", True), ("gpu-1b", "cpu-1b", True)]

    worker.wait_for_layer_load()
    worker.load_stream.synchronize.assert_called_once()
    assert worker.current_layer == 1
    assert copy_calls[2:] == [("gpu-2a", "cpu-2a", True)]

    worker.current_layer = 2
    worker.load_kv_layer(2)


def test_worker_get_finished_single_rank(monkeypatch):
    queue_items = iter(["req-a"])
    worker = object.__new__(connector_module.CPUOffloadingConnectorWorker)
    worker.requests = {"req-a": object()}
    worker.save_output_queue = SimpleNamespace(get_nowait=lambda: next(queue_items))
    worker.tp_world_size = 1

    def raise_empty():
        raise connector_module.queue.Empty

    worker.save_output_queue.get_nowait = MagicMock(side_effect=["req-a", connector_module.queue.Empty()])

    assert worker.get_finished() == {"req-a"}
    assert "req-a" not in worker.requests


def test_worker_get_finished_rank_zero_and_nonzero(monkeypatch):
    started_threads = []

    class FakeThread:
        def __init__(self, target=None, args=()):
            self.target = target
            self.args = args
            self.daemon = False
            self.started = False
            started_threads.append(self)

        def start(self):
            self.started = True

    worker = object.__new__(connector_module.CPUOffloadingConnectorWorker)
    worker.requests = {"req-a": 1, "req-b": 2, "req-c": 3}
    worker.save_output_queue = MagicMock()
    worker.save_output_queue.get_nowait = MagicMock(side_effect=["req-a", connector_module.queue.Empty()])
    worker.tp_world_size = 2
    worker.tp_rank = 0
    worker.tp_group = SimpleNamespace(recv_object=lambda src: ["req-b"])
    worker.done_sending_count = defaultdict(int, {"req-a": 1})
    monkeypatch.setattr(connector_module.threading, "Thread", FakeThread)

    result = worker.get_finished()

    assert result == {"req-a"}
    assert started_threads[0].target == worker._sending_finished
    assert started_threads[0].args == ({"req-a"},)
    assert started_threads[0].started is True

    other = object.__new__(connector_module.CPUOffloadingConnectorWorker)
    other.requests = {"req-z": 1}
    other.save_output_queue = MagicMock()
    other.save_output_queue.get_nowait = MagicMock(side_effect=["req-z", connector_module.queue.Empty()])
    other.tp_world_size = 2
    other.tp_rank = 1
    sent = []
    other.tp_group = SimpleNamespace(send_object=lambda obj, dst: sent.append((obj, dst)))

    assert other.get_finished() == {"req-z"}
    assert sent == [({"req-z"}, 0)]


def test_worker_sending_finished_and_save_listener(monkeypatch):
    rpc_calls = []
    worker = object.__new__(connector_module.CPUOffloadingConnectorWorker)
    worker.zmq_rpc_client = SimpleNamespace(call=lambda func, req_id: rpc_calls.append((func, req_id)))

    worker._sending_finished({"req-a", "req-b"})
    assert set(rpc_calls) == {("cache_and_free_slots", "req-a"), ("cache_and_free_slots", "req-b")}

    class FakeQueue:
        def __init__(self, item):
            self.item = item
            self.count = 0

        def get(self):
            if self.count == 0:
                self.count += 1
                return self.item
            raise RuntimeError("stop")

    class FakeTensor:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, idx):
            return self

        def copy_(self, other, non_blocking=False):
            save_copies.append((self.name, other.name, non_blocking))

    save_copies = []
    saver = object.__new__(connector_module.CPUOffloadingConnectorWorker)
    saver.save_input_queue = FakeQueue(("req-save", connector_module.ReqMeta([5, 6, 7], [10, 11, 12], 4, 12, 4, 8)))
    saver.save_output_queue = SimpleNamespace(items=[], put=lambda item: saver.save_output_queue.items.append(item))
    saver.block_size = 4
    saver.use_mla = False
    saver.tp_rank = 0
    saver.tp_world_size = 2
    saver.save_stream = SimpleNamespace(synchronize=MagicMock())
    saver.cpu_kv_caches = [[FakeTensor("cpu-1")]]
    saver.gpu_kv_caches = {"layer": [FakeTensor("gpu-1")]}
    monkeypatch.setattr(connector_module.torch.npu, "stream", lambda _stream: nullcontext())

    with pytest.raises(RuntimeError, match="stop"):
        saver._save_listener()

    assert save_copies == [("cpu-1", "gpu-1", True)]
    saver.save_stream.synchronize.assert_called_once()
    assert saver.save_output_queue.items == ["req-save"]


def test_worker_save_listener_mla_branch(monkeypatch):
    class FakeQueue:
        def __init__(self, item):
            self.item = item
            self.count = 0

        def get(self):
            if self.count == 0:
                self.count += 1
                return self.item
            raise RuntimeError("stop")

    class FakeTensor:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, idx):
            return self

        def copy_(self, other, non_blocking=False):
            mla_copies.append((self.name, other.name, non_blocking))

    mla_copies = []
    saver = object.__new__(connector_module.CPUOffloadingConnectorWorker)
    saver.save_input_queue = FakeQueue(("req-mla", connector_module.ReqMeta([5, 6, 7, 8], [10, 11, 12, 13], 8, 8, 0, 0)))
    saver.save_output_queue = SimpleNamespace(items=[], put=lambda item: saver.save_output_queue.items.append(item))
    saver.block_size = 4
    saver.use_mla = True
    saver.tp_rank = 1
    saver.tp_world_size = 2
    saver.save_stream = SimpleNamespace(synchronize=MagicMock())
    saver.cpu_kv_caches = [[FakeTensor("cpu-1")]]
    saver.gpu_kv_caches = {"layer": [FakeTensor("gpu-1")]}
    monkeypatch.setattr(connector_module.torch.npu, "stream", lambda _stream: nullcontext())

    with pytest.raises(RuntimeError, match="stop"):
        saver._save_listener()

    assert mla_copies == [("cpu-1", "gpu-1", True), ("cpu-1", "gpu-1", True)]


def test_get_kv_cache_spec_covers_ec_transfer_and_layer_types(monkeypatch):
    monkeypatch.setattr(connector_module, "has_ec_transfer", lambda: True)
    monkeypatch.setattr(connector_module, "get_ec_transfer", lambda: SimpleNamespace(is_producer=True))
    assert connector_module.get_kv_cache_spec(make_vllm_config()) == {}

    monkeypatch.setattr(connector_module, "has_ec_transfer", lambda: False)

    class FakeAttention:
        def __init__(self, spec):
            self._spec = spec

        def get_kv_cache_spec(self, _cfg):
            return self._spec

    class FakeMLAAttention(FakeAttention):
        head_size = 32

    class FakeMamba:
        def __init__(self, spec):
            self._spec = spec

        def get_kv_cache_spec(self, _cfg):
            return self._spec

    monkeypatch.setattr(connector_module, "Attention", FakeAttention)
    monkeypatch.setattr(connector_module, "MLAAttention", FakeMLAAttention)
    monkeypatch.setattr(connector_module, "MambaBase", FakeMamba)
    monkeypatch.setattr(connector_module, "FullAttentionSpec", lambda **kwargs: ("full", kwargs))
    monkeypatch.setattr(connector_module, "STR_DTYPE_TO_TORCH_DTYPE", {"fp8": "mapped-dtype"})

    config = make_vllm_config(cache_dtype="auto")
    config.model_config.hf_config.index_topk = 1
    layers = {
        "attn": FakeAttention("attn-spec"),
        "mla-sparse": FakeMLAAttention("ignored"),
        "mla-spec": FakeMLAAttention("mla-spec"),
        "mamba": FakeMamba("mamba-spec"),
    }
    monkeypatch.setattr(connector_module, "get_layers_from_vllm_config", lambda _cfg, _base: layers)

    with pytest.raises(NotImplementedError, match="Prefix caching is not supported for Mamba yet"):
        connector_module.get_kv_cache_spec(config)

    config.cache_config.enable_prefix_caching = False
    config.model_config.hf_config = SimpleNamespace()
    config.cache_config.cache_dtype = "fp8"
    result = connector_module.get_kv_cache_spec(config)

    assert result["attn"] == "attn-spec"
    assert result["mla-sparse"] == "ignored"
    assert result["mla-spec"] == "mla-spec"
    assert result["mamba"] == "mamba-spec"


def test_get_kv_cache_spec_sparse_mla_branch(monkeypatch):
    class FakeMLAAttention:
        head_size = 64

        def get_kv_cache_spec(self, _cfg):
            return "unused"

    monkeypatch.setattr(connector_module, "has_ec_transfer", lambda: False)
    monkeypatch.setattr(connector_module, "Attention", type("FakeAttention", (), {}))
    monkeypatch.setattr(connector_module, "MLAAttention", FakeMLAAttention)
    monkeypatch.setattr(connector_module, "MambaBase", type("FakeMamba", (), {}))
    monkeypatch.setattr(connector_module, "FullAttentionSpec", lambda **kwargs: kwargs)
    monkeypatch.setattr(connector_module, "get_layers_from_vllm_config", lambda _cfg, _base: {"mla": FakeMLAAttention()})

    config = make_vllm_config(cache_dtype="auto")
    config.model_config.hf_config.index_topk = 1

    result = connector_module.get_kv_cache_spec(config)

    assert result == {"mla": {"block_size": 4, "num_kv_heads": 1, "head_size": 64, "dtype": "model-dtype"}}


def test_get_kv_cache_spec_skips_falsey_specs(monkeypatch):
    class FakeAttention:
        def get_kv_cache_spec(self, _cfg):
            return None

    class FakeMLAAttention:
        head_size = 8

        def get_kv_cache_spec(self, _cfg):
            return None

    class FakeMamba:
        def get_kv_cache_spec(self, _cfg):
            return None

    monkeypatch.setattr(connector_module, "has_ec_transfer", lambda: False)
    monkeypatch.setattr(connector_module, "Attention", FakeAttention)
    monkeypatch.setattr(connector_module, "MLAAttention", FakeMLAAttention)
    monkeypatch.setattr(connector_module, "MambaBase", FakeMamba)
    monkeypatch.setattr(
        connector_module,
        "get_layers_from_vllm_config",
        lambda _cfg, _base: {"attn": FakeAttention(), "mla": FakeMLAAttention(), "mamba": FakeMamba()},
    )

    config = make_vllm_config()
    config.cache_config.enable_prefix_caching = False

    assert connector_module.get_kv_cache_spec(config) == {}


def test_get_kv_cache_spec_hits_standalone_mla_and_unknown_layer_branches(monkeypatch):
    class FakeAttention:
        def get_kv_cache_spec(self, _cfg):
            return None

    class FakeMLAAttention:
        head_size = 8

        def __init__(self, spec):
            self._spec = spec

        def get_kv_cache_spec(self, _cfg):
            return self._spec

    class FakeMamba:
        def get_kv_cache_spec(self, _cfg):
            return None

    class UnknownLayer:
        pass

    monkeypatch.setattr(connector_module, "has_ec_transfer", lambda: False)
    monkeypatch.setattr(connector_module, "Attention", FakeAttention)
    monkeypatch.setattr(connector_module, "MLAAttention", FakeMLAAttention)
    monkeypatch.setattr(connector_module, "MambaBase", FakeMamba)
    monkeypatch.setattr(
        connector_module,
        "get_layers_from_vllm_config",
        lambda _cfg, _base: {"mla-hit": FakeMLAAttention("mla-spec"), "unknown": UnknownLayer()},
    )

    assert connector_module.get_kv_cache_spec(make_vllm_config()) == {"mla-hit": "mla-spec"}
