import types
from types import SimpleNamespace

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import pool_worker as worker_module


class FakeThread:

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.started = False
        self.added = []
        self.stored = []
        self.deleted = []
        self.finished = set()
        self.kv_events = ["event"]
        self.stored_requests = {}

    def start(self):
        self.started = True

    def add_request(self, request):
        self.added.append(request)

    def add_stored_request(self, req_id):
        self.stored.append(req_id)
        self.stored_requests[req_id] = self.stored_requests.get(req_id, 0) + 1

    def delete_finished_stored_request(self, req_id):
        self.deleted.append(req_id)
        self.stored_requests.pop(req_id, None)

    def get_and_clear_finished_requests(self):
        result = set(self.finished)
        self.finished.clear()
        return result

    def get_kv_events(self):
        return list(self.kv_events)


class FakeEvent:

    def __init__(self):
        self.wait_called = False
        self.clear_called = False
        self.record_called = False
        self.set_called = False

    def wait(self, timeout=None):
        self.wait_called = True
        return True

    def clear(self):
        self.clear_called = True

    def record(self):
        self.record_called = True

    def set(self):
        self.set_called = True


class FakeTensor:

    def __init__(self, shape, ptr, element_size=2):
        self.shape = shape
        self._ptr = ptr
        self._element_size = element_size

    def data_ptr(self):
        return self._ptr

    def element_size(self):
        return self._element_size

    def __getitem__(self, _index):
        return self


class FakeKey:

    def __init__(self, value):
        self.value = value

    def to_string(self):
        return self.value

    def split_layers(self, num_layers):
        return [FakeKey(f"{self.value}-layer-{index}") for index in range(num_layers)]


class FakeTokenDatabase:

    def __init__(self, metadata=None, block_size=None, partitions=None):
        self.metadata = metadata
        self.block_size = block_size
        self.partitions = partitions
        self.base_addrs = None
        self.block_len = None
        self.process_result = []
        self.prepare_calls = []

    def set_kv_caches_base_addr(self, addrs):
        self.base_addrs = list(addrs)

    def set_block_len(self, block_len):
        self.block_len = list(block_len)

    def process_tokens(self, token_len, block_hashes, mask_num=0):
        return list(self.process_result)

    def prepare_value(self, start, end, block_ids):
        self.prepare_calls.append((start, end, list(block_ids)))
        return [start + 10], [end - start], block_ids[start // 2 if block_ids else 0]


def make_model_config(*, use_mla=False, num_hidden_layers=6, total_kv_heads=4, num_layers=3, model="/tmp/model"):
    return SimpleNamespace(
        use_mla=use_mla,
        hf_text_config=SimpleNamespace(num_hidden_layers=num_hidden_layers),
        get_num_layers=lambda parallel_config: num_layers,
        get_total_num_kv_heads=lambda: total_kv_heads,
        model=model,
    )


def make_vllm_config(*, kv_role="kv_both", extra=None, use_mla=False, rank=0, tp=2, pp=2, block_size=4, enable_events=False):
    extra = extra or {}
    return SimpleNamespace(
        model_config=make_model_config(use_mla=use_mla),
        parallel_config=SimpleNamespace(rank=rank, pipeline_parallel_size=pp, data_parallel_rank=1),
        kv_transfer_config=SimpleNamespace(kv_role=kv_role, kv_connector_extra_config=extra),
        cache_config=SimpleNamespace(block_size=block_size),
        kv_events_config=SimpleNamespace(enable_kv_cache_events=enable_events) if enable_events else None,
    )


def patch_common_constructor(monkeypatch, *, tp_rank=1, tp_size=2, pcp_size=1, pcp_rank=0, dcp_size=1, dcp_rank=0, backend_name="mooncake"):
    fake_database_instances = []

    def fake_database(metadata, block_size, partitions):
        db = FakeTokenDatabase(metadata, block_size, partitions)
        fake_database_instances.append(db)
        return db

    monkeypatch.setattr(worker_module, "ChunkedTokenDatabase", fake_database)
    monkeypatch.setattr(worker_module, "get_tensor_model_parallel_rank", lambda: tp_rank)
    monkeypatch.setattr(worker_module, "get_tensor_model_parallel_world_size", lambda: tp_size)
    monkeypatch.setattr(worker_module, "get_pcp_group", lambda: SimpleNamespace(world_size=pcp_size, rank_in_group=pcp_rank))
    monkeypatch.setattr(worker_module, "get_decode_context_model_parallel_world_size", lambda: dcp_size)
    monkeypatch.setattr(worker_module, "get_decode_context_model_parallel_rank", lambda: dcp_rank)

    fake_backend_cls = lambda parallel_config: SimpleNamespace(parallel_config=parallel_config)
    monkeypatch.setattr(worker_module.importlib, "import_module", lambda path: types.SimpleNamespace(**({"MooncakeBackend": fake_backend_cls} if backend_name == "mooncake" else {"MemcacheBackend": fake_backend_cls})))
    return fake_database_instances


def test_constructor_sets_basic_fields_and_auto_partitions(monkeypatch):
    fake_databases = patch_common_constructor(monkeypatch, tp_rank=1, tp_size=4, pcp_size=2, pcp_rank=1, dcp_size=2, dcp_rank=1)
    config = make_vllm_config(kv_role="kv_consumer", extra={"consumer_is_to_put": True, "prefill_pp_size": 3}, use_mla=False, rank=5, pp=2, block_size=4, enable_events=True)

    worker = worker_module.KVPoolWorker(config, use_layerwize=True)

    assert worker.use_mla is False
    assert worker.use_sparse is False
    assert worker.tp_rank == 1
    assert worker.tp_size == 4
    assert worker.pp_rank == 1
    assert worker.block_size == 16
    assert worker.num_kv_head == 4
    assert worker.put_step == 1
    assert worker.head_or_tp_rank == 1
    assert worker.metadata.model_name == "model"
    assert fake_databases[0].partitions == [2, 2, 2]
    assert worker.enable_kv_events is True
    assert worker.kv_send_thread is None
    assert worker.kv_recv_thread is None


def test_constructor_distributes_remaining_partition_layers_and_requires_backend_fields(monkeypatch):
    fake_databases = patch_common_constructor(monkeypatch)
    config = make_vllm_config(kv_role="kv_consumer", extra={"consumer_is_to_put": True, "prefill_pp_size": 4})
    config.model_config = make_model_config(num_hidden_layers=10)

    worker = worker_module.KVPoolWorker(config, use_layerwize=False)
    assert fake_databases[0].partitions == [2, 3, 3, 2]

    patch_common_constructor(monkeypatch)
    original_backend_map = worker_module.backend_map.copy()
    worker_module.backend_map["mooncake"] = {"path": None, "name": "MooncakeBackend"}
    with pytest.raises(AssertionError):
        worker_module.KVPoolWorker(make_vllm_config(), use_layerwize=False)
    worker_module.backend_map.clear()
    worker_module.backend_map.update(original_backend_map)


def test_constructor_handles_mla_head_replica_and_partition_validation(monkeypatch):
    patch_common_constructor(monkeypatch, tp_rank=3, tp_size=8)
    config = make_vllm_config(kv_role="kv_consumer", extra={"consumer_is_to_put": True, "prefill_pp_size": 2, "prefill_pp_layer_partition": "3,3"}, use_mla=True)

    worker = worker_module.KVPoolWorker(config, use_layerwize=False)
    assert worker.use_mla is True
    assert worker.num_kv_head == 1
    assert worker.put_step == 8
    assert worker.head_or_tp_rank == 0

    patch_common_constructor(monkeypatch)
    bad_config = make_vllm_config(kv_role="kv_consumer", extra={"consumer_is_to_put": True, "prefill_pp_size": 2, "prefill_pp_layer_partition": "a,b"})
    with pytest.raises(ValueError, match="Invalid partition string"):
        worker_module.KVPoolWorker(bad_config, use_layerwize=False)

    patch_common_constructor(monkeypatch)
    bad_len = make_vllm_config(kv_role="kv_consumer", extra={"consumer_is_to_put": True, "prefill_pp_size": 3, "prefill_pp_layer_partition": "3,3"})
    with pytest.raises(ValueError, match="prefill_pp_size"):
        worker_module.KVPoolWorker(bad_len, use_layerwize=False)

    patch_common_constructor(monkeypatch)
    bad_sum = make_vllm_config(kv_role="kv_consumer", extra={"consumer_is_to_put": True, "prefill_pp_size": 2, "prefill_pp_layer_partition": "2,3"})
    with pytest.raises(ValueError, match="num_hidden_layers"):
        worker_module.KVPoolWorker(bad_sum, use_layerwize=False)


def test_constructor_rejects_unknown_backend(monkeypatch):
    patch_common_constructor(monkeypatch)
    config = make_vllm_config(extra={"backend": "unknown"})

    with pytest.raises(AssertionError):
        worker_module.KVPoolWorker(config, use_layerwize=False)


def test_register_kv_caches_non_layerwise_and_layerwise(monkeypatch):
    register_calls = []
    send_threads = []
    recv_threads = []
    events = []

    monkeypatch.setattr(worker_module.threading, "Event", lambda: events.append(FakeEvent()) or events[-1])
    monkeypatch.setattr(worker_module, "KVCacheStoreSendingThread", lambda *args: send_threads.append(FakeThread(*args)) or send_threads[-1])
    monkeypatch.setattr(worker_module, "KVCacheStoreRecvingThread", lambda *args: recv_threads.append(FakeThread(*args)) or recv_threads[-1])
    monkeypatch.setattr(worker_module, "KVCacheStoreLayerSendingThread", lambda *args: send_threads.append(FakeThread(*args)) or send_threads[-1])
    monkeypatch.setattr(worker_module, "KVCacheStoreLayerRecvingThread", lambda *args: recv_threads.append(FakeThread(*args)) or recv_threads[-1])
    monkeypatch.setattr(worker_module.logger, "info", lambda *args: None)

    worker = object.__new__(worker_module.KVPoolWorker)
    worker.use_mla = False
    worker.use_sparse = False
    worker.kv_role = "kv_both"
    worker.consumer_is_to_put = False
    worker.use_layerwise = False
    worker.load_async = True
    worker.block_size = 4
    worker.tp_rank = 1
    worker.dcp_size = 1
    worker.put_step = 2
    worker.num_layers = 3
    worker.enable_kv_events = True
    worker.m_store = SimpleNamespace(register_buffer=lambda ptrs, lengths: register_calls.append((ptrs, lengths)))
    worker.token_database = FakeTokenDatabase()

    kv_caches = {
        "layer": [FakeTensor((2, 4, 2, 8), 100), FakeTensor((2, 4, 2, 8), 200)],
    }
    worker.register_kv_caches(kv_caches)

    assert worker.num_blocks == 2
    assert worker.block_len == [128]
    assert worker.kv_caches_base_addr == [100, 200]
    assert register_calls == [([100, 200], [256, 256])]
    assert worker.token_database.base_addrs == [100, 200]
    assert worker.token_database.block_len == [128]
    assert len(send_threads) == 1 and send_threads[0].started is True
    assert len(recv_threads) == 1 and recv_threads[0].started is True
    assert events[-1].wait_called is True

    send_threads.clear()
    recv_threads.clear()
    events.clear()
    register_calls.clear()

    worker2 = object.__new__(worker_module.KVPoolWorker)
    worker2.use_mla = True
    worker2.use_sparse = False
    worker2.kv_role = "kv_producer"
    worker2.consumer_is_to_put = False
    worker2.use_layerwise = True
    worker2.load_async = False
    worker2.block_size = 4
    worker2.tp_rank = 0
    worker2.dcp_size = 1
    worker2.put_step = 1
    worker2.num_layers = 2
    worker2.enable_kv_events = False
    worker2.m_store = SimpleNamespace(register_buffer=lambda ptrs, lengths: register_calls.append((ptrs, lengths)))
    worker2.token_database = FakeTokenDatabase()

    mla_caches = {"layer": [FakeTensor((2, 4, 2, 8), 300), FakeTensor((2, 4, 2, 4), 400)]}
    worker2.register_kv_caches(mla_caches)

    assert worker2.block_len == [128, 64]
    assert len(send_threads) == 1 and send_threads[0].started is True
    assert len(recv_threads) == 1 and recv_threads[0].started is True
    assert events[-1].wait_called is True


def test_register_kv_caches_role_branches_without_send_thread(monkeypatch):
    send_threads = []
    recv_threads = []
    events = []
    monkeypatch.setattr(worker_module.threading, "Event", lambda: events.append(FakeEvent()) or events[-1])
    monkeypatch.setattr(worker_module, "KVCacheStoreSendingThread", lambda *args: send_threads.append(FakeThread(*args)) or send_threads[-1])
    monkeypatch.setattr(worker_module, "KVCacheStoreRecvingThread", lambda *args: recv_threads.append(FakeThread(*args)) or recv_threads[-1])
    monkeypatch.setattr(worker_module, "KVCacheStoreLayerSendingThread", lambda *args: send_threads.append(FakeThread(*args)) or send_threads[-1])
    monkeypatch.setattr(worker_module, "KVCacheStoreLayerRecvingThread", lambda *args: recv_threads.append(FakeThread(*args)) or recv_threads[-1])
    monkeypatch.setattr(worker_module.logger, "info", lambda *args: None)

    worker = object.__new__(worker_module.KVPoolWorker)
    worker.use_mla = False
    worker.use_sparse = False
    worker.kv_role = "kv_consumer"
    worker.consumer_is_to_put = False
    worker.use_layerwise = True
    worker.load_async = False
    worker.block_size = 4
    worker.tp_rank = 0
    worker.dcp_size = 1
    worker.put_step = 1
    worker.num_layers = 2
    worker.enable_kv_events = False
    worker.m_store = SimpleNamespace(register_buffer=lambda ptrs, lengths: None)
    worker.token_database = FakeTokenDatabase()
    worker.register_kv_caches({"layer": [FakeTensor((2, 4, 2, 8), 1)]})
    assert send_threads == []
    assert len(recv_threads) == 1

    send_threads.clear()
    recv_threads.clear()
    worker2 = object.__new__(worker_module.KVPoolWorker)
    worker2.use_mla = False
    worker2.use_sparse = False
    worker2.kv_role = "kv_consumer"
    worker2.consumer_is_to_put = False
    worker2.use_layerwise = False
    worker2.load_async = False
    worker2.block_size = 4
    worker2.tp_rank = 0
    worker2.dcp_size = 1
    worker2.put_step = 1
    worker2.num_layers = 2
    worker2.enable_kv_events = False
    worker2.m_store = SimpleNamespace(register_buffer=lambda ptrs, lengths: None)
    worker2.token_database = FakeTokenDatabase()
    worker2.register_kv_caches({"layer": [FakeTensor((2, 4, 2, 8), 2)]})
    assert send_threads == []
    assert recv_threads == []


def test_start_load_kv_covers_layerwise_async_and_sync_paths(monkeypatch):
    sync_get_calls = []
    worker = object.__new__(worker_module.KVPoolWorker)
    worker.current_layer = 99
    worker.block_size = 4
    worker.use_layerwise = False
    worker.load_async = False
    worker.tp_rank = 1
    worker.m_store = SimpleNamespace(get=lambda keys, addrs, sizes: sync_get_calls.append((keys, addrs, sizes)))
    worker.token_database = FakeTokenDatabase()
    worker.token_database.process_result = [(0, 2, FakeKey("k1")), (2, 4, FakeKey("k2"))]
    worker.kv_recv_thread = FakeThread()

    request_skip = SimpleNamespace(load_spec=None)
    request_sync = SimpleNamespace(
        req_id="req-sync",
        load_spec=SimpleNamespace(can_load=True, kvpool_cached_tokens=3, vllm_cached_tokens=2, token_len=0),
        token_len_chunk=4,
        block_hashes=["a", "b"],
        block_ids=[10, 11],
    )
    metadata = SimpleNamespace(requests=[request_skip, request_sync])
    worker.start_load_kv(metadata)

    assert worker.current_layer == 0
    assert request_sync.load_spec.token_len == 4
    assert sync_get_calls == [(["k2", "k1"], [[12], [10]], [[2], [2]])]

    async_worker = object.__new__(worker_module.KVPoolWorker)
    async_worker.block_size = 4
    async_worker.use_layerwise = False
    async_worker.load_async = True
    async_worker.current_layer = 0
    async_worker.kv_recv_thread = FakeThread()
    async_worker.token_database = FakeTokenDatabase()
    async_request = SimpleNamespace(
        req_id="req-async",
        load_spec=SimpleNamespace(can_load=True, kvpool_cached_tokens=4, vllm_cached_tokens=0, token_len=0),
        token_len_chunk=4,
        block_hashes=["a"],
        block_ids=[1],
    )
    async_worker.start_load_kv(SimpleNamespace(requests=[async_request]))
    assert async_worker.kv_recv_thread.added == [async_request]

    layer_calls = []
    layer_worker = object.__new__(worker_module.KVPoolWorker)
    layer_worker.block_size = 4
    layer_worker.use_layerwise = True
    layer_worker.load_async = False
    layer_worker.current_layer = 0
    layer_worker.retrieve_layer = lambda request: iter([None, "final-mask"])
    layer_request = SimpleNamespace(
        req_id="req-layer",
        load_spec=SimpleNamespace(can_load=True, kvpool_cached_tokens=4, vllm_cached_tokens=0, token_len=0),
        token_len_chunk=4,
        block_hashes=["a"],
        block_ids=[1],
    )
    layer_worker.start_load_kv(SimpleNamespace(requests=[layer_request]))
    assert len(layer_worker.layerwise_retrievers) == 1


def test_wait_for_layer_load_save_and_wait_for_save(monkeypatch):
    debug_logs = []
    monkeypatch.setattr(worker_module.logger, "debug", lambda *args: debug_logs.append(args))
    monkeypatch.setattr(worker_module.torch.npu, "Event", FakeEvent)

    worker = object.__new__(worker_module.KVPoolWorker)
    worker.num_layers = 2
    worker.current_layer = 1
    worker.layerwise_retrievers = [iter([SimpleNamespace(sum=lambda: SimpleNamespace(item=lambda: 3))])]
    worker.wait_for_layer_load()
    assert debug_logs and "Retrieved" in debug_logs[0][0]

    worker_nonfinal = object.__new__(worker_module.KVPoolWorker)
    worker_nonfinal.num_layers = 3
    worker_nonfinal.current_layer = 0
    worker_nonfinal.layerwise_retrievers = [iter([SimpleNamespace(sum=lambda: SimpleNamespace(item=lambda: 9))])]
    worker_nonfinal.wait_for_layer_load()

    saved_requests = []
    worker2 = object.__new__(worker_module.KVPoolWorker)
    worker2.current_layer = 0
    worker2.layerwise_storers = []
    worker2.store_layer = lambda request, current_event: iter([saved_requests.append((request.req_id, current_event.record_called)), None])
    savable = SimpleNamespace(req_id="save-1", can_save=True)
    skip = SimpleNamespace(req_id="skip", can_save=False)
    worker2.save_kv_layer(SimpleNamespace(requests=[skip, savable]))
    assert worker2.current_layer == 1
    assert saved_requests == [("save-1", True)]

    send_thread = FakeThread()
    worker3 = object.__new__(worker_module.KVPoolWorker)
    worker3.kv_send_thread = send_thread
    request = SimpleNamespace(req_id="req", can_save=True, current_event=None)
    worker3.wait_for_save(SimpleNamespace(requests=[request]))
    assert send_thread.stored == ["req"]
    assert send_thread.added == [request]
    assert request.current_event.record_called is True

    worker4 = object.__new__(worker_module.KVPoolWorker)
    worker4.current_layer = 0
    worker4.layerwise_storers = []
    worker4.store_layer = lambda request, current_event: iter(())
    worker4.save_kv_layer(SimpleNamespace(requests=[SimpleNamespace(req_id="skip", can_save=False)]))
    assert worker4.current_layer == 1

    worker4.layerwise_storers = [iter([None])]
    worker4.current_layer = 1
    worker4.save_kv_layer(SimpleNamespace(requests=[]))
    assert worker4.current_layer == 2

    worker_exception = object.__new__(worker_module.KVPoolWorker)
    worker_exception.current_layer = 1
    def raising_storer():
        raise RuntimeError("boom")
        yield

    worker_exception.layerwise_storers = [raising_storer()]
    with pytest.raises(RuntimeError, match="boom"):
        worker_exception.save_kv_layer(SimpleNamespace(requests=[]))

    worker5 = object.__new__(worker_module.KVPoolWorker)
    worker5.kv_send_thread = FakeThread()
    skip_request = SimpleNamespace(req_id="skip", can_save=None, current_event=None)
    worker5.wait_for_save(SimpleNamespace(requests=[skip_request]))
    assert worker5.kv_send_thread.stored == []
    assert worker5.kv_send_thread.added == []


def test_retrieve_and_store_layer_generators(monkeypatch):
    info_logs = []
    debug_logs = []
    monkeypatch.setattr(worker_module.logger, "info", lambda *args: info_logs.append(args))
    monkeypatch.setattr(worker_module.logger, "debug", lambda *args: debug_logs.append(args))

    recv_thread = FakeThread()
    send_thread = FakeThread()
    get_event = FakeEvent()

    worker = object.__new__(worker_module.KVPoolWorker)
    worker.block_size = 4
    worker.num_layers = 2
    worker.token_database = FakeTokenDatabase()
    worker.token_database.process_result = [(0, 2, FakeKey("k1")), (2, 4, FakeKey("k2"))]
    worker.get_event = get_event
    worker.kv_recv_thread = recv_thread

    request = SimpleNamespace(
        req_id="req",
        token_len_chunk=4,
        load_spec=SimpleNamespace(vllm_cached_tokens=0),
        block_hashes=["a", "b"],
        block_ids=[10, 11],
    )
    generator = worker.retrieve_layer(request)
    assert next(generator) is None
    assert next(generator) is None
    ret_mask = next(generator)
    assert ret_mask.tolist() == [True, True, True, True]
    assert recv_thread.added[0].layer_id == 0
    assert recv_thread.added[1].layer_id == 1

    get_event.wait = lambda timeout=None: False
    worker_retry = object.__new__(worker_module.KVPoolWorker)
    worker_retry.block_size = 4
    worker_retry.num_layers = 2
    worker_retry.token_database = FakeTokenDatabase()
    worker_retry.token_database.process_result = [(0, 2, FakeKey("k1"))]
    worker_retry.get_event = get_event
    worker_retry.kv_recv_thread = FakeThread()
    retry_gen = worker_retry.retrieve_layer(request)
    assert next(retry_gen) is None
    assert next(retry_gen) is None
    assert info_logs and "Layerwise get failed" in info_logs[0][0]

    worker_empty = object.__new__(worker_module.KVPoolWorker)
    worker_empty.block_size = 4
    worker_empty.num_layers = 2
    worker_empty.token_database = FakeTokenDatabase()
    worker_empty.token_database.process_result = []
    generator_empty = worker_empty.retrieve_layer(request)
    assert next(generator_empty) is None
    assert next(generator_empty) is None
    assert generator_empty.send(None).tolist() == [False, False, False, False]

    worker_store = object.__new__(worker_module.KVPoolWorker)
    worker_store.num_layers = 2
    worker_store.token_database = FakeTokenDatabase()
    worker_store.token_database.process_result = [(0, 2, FakeKey("k1"))]
    worker_store.kv_send_thread = send_thread
    store_request = SimpleNamespace(req_id="store", token_len_chunk=2, block_hashes=["a"], block_ids=[1], is_last_chunk=True)
    store_gen = worker_store.store_layer(store_request, current_event="evt")
    assert next(store_gen) is None
    assert next(store_gen) is None
    assert next(store_gen, "done") == "done"
    assert send_thread.added[0].layer_id == 0
    assert send_thread.added[1].layer_id == 1

    worker_store_empty = object.__new__(worker_module.KVPoolWorker)
    worker_store_empty.num_layers = 2
    worker_store_empty.token_database = FakeTokenDatabase()
    worker_store_empty.token_database.process_result = []
    empty_gen = worker_store_empty.store_layer(store_request, current_event=None)
    assert next(empty_gen) is None
    assert next(empty_gen) is None
    assert next(empty_gen, "done") == "done"


def test_get_finished_lookup_and_helpers(monkeypatch):
    debug_logs = []
    monkeypatch.setattr(worker_module.logger, "debug", lambda *args: debug_logs.append(args))

    worker = object.__new__(worker_module.KVPoolWorker)
    worker.kv_role = "kv_both"
    worker.consumer_is_to_put = False
    worker.load_async = True
    worker.tp_rank = 1
    worker.kv_recv_thread = FakeThread()
    worker.kv_recv_thread.finished = {"recv-done"}
    worker.get_and_clear_finished_requests = lambda finished_req_ids, meta: {"send-done"}
    done_sending, done_recving = worker.get_finished({"done"}, SimpleNamespace())
    assert done_sending == {"send-done"}
    assert done_recving == {"recv-done"}
    assert debug_logs and "Number of completed" in debug_logs[0][0]

    worker2 = object.__new__(worker_module.KVPoolWorker)
    worker2.kv_send_thread = FakeThread()
    worker2.kv_send_thread.stored_requests = {"a": 0, "b": 1, "c": 0}
    worker2.finished_store_req = {"a", "c"}
    meta = SimpleNamespace(preempted_req_ids={"preempted"})
    worker2.kv_send_thread.stored_requests["preempted"] = 2
    result = worker2.get_and_clear_finished_requests({"b", "c", "d"}, meta)
    assert result == {"a", "c"}
    assert worker2.kv_send_thread.deleted == ["preempted", "a", "c"]
    assert worker2.finished_store_req == {"b"}

    worker3 = object.__new__(worker_module.KVPoolWorker)
    worker3.num_layers = 2
    worker3.tp_size = 2
    worker3.num_kv_head = 2
    worker3.pp_size = 2
    worker3.m_store = SimpleNamespace(exists=lambda keys: [1, 0])
    worker3.token_database = FakeTokenDatabase()
    worker3.token_database.process_result = [(0, 2, FakeKey("key@head_or_tp_rank:0@pp_rank:0")), (2, 4, FakeKey("key2@head_or_tp_rank:0@pp_rank:0"))]
    assert worker3.lookup(4, [b"a", b"b"], False) == 2

    worker3.m_store = SimpleNamespace(exists=lambda keys: [1, 1])
    assert worker3.lookup(4, [b"a", b"b"], False) == 4

    worker3.m_store = SimpleNamespace(exists=lambda keys: (_ for _ in ()).throw(RuntimeError("boom")))
    error_logs = []
    monkeypatch.setattr(worker_module.logger, "error", lambda *args: error_logs.append(args))
    assert worker3.lookup(4, [b"a"], False) == 0
    assert error_logs

    worker4 = object.__new__(worker_module.KVPoolWorker)
    worker4.num_layers = 2
    worker4.tp_size = 2
    worker4.num_kv_head = 2
    worker4.pp_size = 2
    worker4.m_store = SimpleNamespace(exists=lambda keys: [1, 1, 1, 1, 1, 1, 0, 1])
    worker4.token_database = FakeTokenDatabase()
    worker4.token_database.process_result = [(0, 2, FakeKey("x@head_or_tp_rank:0@pp_rank:0")), (2, 4, FakeKey("y@head_or_tp_rank:0@pp_rank:0"))]
    assert worker4.lookup_scheduler(4, [b"a", b"b"], True) == 2
    assert worker4.check_all_layers_exists([1, 1, 1, 0], 2) == [1, 0]
    assert worker4.find_min_first_non_one_index([[1, 1], [1, 0]]) == 1
    assert worker4.find_min_first_non_one_index([[1, 1], [1, 1]]) == -1

    worker5 = object.__new__(worker_module.KVPoolWorker)
    worker5.enable_kv_events = True
    worker5.kv_send_thread = FakeThread()
    assert worker5.get_kv_events() == ["event"]
    worker5.enable_kv_events = False
    assert worker5.get_kv_events() == []


def test_get_finished_and_lookup_scheduler_remaining_branches(monkeypatch):
    worker = object.__new__(worker_module.KVPoolWorker)
    worker.kv_role = "kv_consumer"
    worker.consumer_is_to_put = False
    worker.load_async = False
    worker.tp_rank = 0
    assert worker.get_finished(set(), SimpleNamespace()) == (set(), set())

    worker_lookup = object.__new__(worker_module.KVPoolWorker)
    worker_lookup.num_layers = 2
    worker_lookup.token_database = FakeTokenDatabase()
    worker_lookup.token_database.process_result = [(0, 2, FakeKey("k1")), (2, 4, FakeKey("k2"))]
    worker_lookup.m_store = SimpleNamespace(exists=lambda keys: [1, 1, 0, 0])
    assert worker_lookup.lookup(4, [b"a", b"b"], True) == 2

    error_logs = []
    monkeypatch.setattr(worker_module.logger, "error", lambda *args: error_logs.append(args))
    worker_lookup.m_store = SimpleNamespace(exists=lambda keys: (_ for _ in ()).throw(RuntimeError("boom")))
    assert worker_lookup.lookup_scheduler(4, [b"a"], False) == 0
    assert error_logs

    worker_sched = object.__new__(worker_module.KVPoolWorker)
    worker_sched.num_layers = 2
    worker_sched.tp_size = 2
    worker_sched.num_kv_head = 2
    worker_sched.pp_size = 2
    worker_sched.token_database = FakeTokenDatabase()
    worker_sched.token_database.process_result = [(0, 2, FakeKey("a@head_or_tp_rank:0@pp_rank:0")), (2, 4, FakeKey("b@head_or_tp_rank:0@pp_rank:0"))]
    worker_sched.m_store = SimpleNamespace(exists=lambda keys: [1, 1, 1, 1, 1, 1, 1, 1])
    assert worker_sched.lookup_scheduler(4, [b"a", b"b"], False) == 4

    worker_finish = object.__new__(worker_module.KVPoolWorker)
    worker_finish.kv_send_thread = FakeThread()
    worker_finish.kv_send_thread.stored_requests = {"instant": 0}
    worker_finish.finished_store_req = set()
    result = worker_finish.get_and_clear_finished_requests({"instant"}, SimpleNamespace(preempted_req_ids=set()))
    assert result == {"instant"}
    assert worker_finish.kv_send_thread.deleted == ["instant"]
