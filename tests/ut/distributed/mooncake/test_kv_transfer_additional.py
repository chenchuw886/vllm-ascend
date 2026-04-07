from types import SimpleNamespace

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import kv_transfer as transfer_module


class FakeStore:

    def __init__(self):
        self.exists_result = [1]
        self.put_calls = []
        self.get_calls = []
        self.set_device_calls = 0

    def set_device(self):
        self.set_device_calls += 1

    def exists(self, keys):
        return self.exists_result

    def put(self, keys, addrs, sizes):
        self.put_calls.append((keys, addrs, sizes))

    def get(self, keys, addrs, sizes):
        self.get_calls.append((keys, addrs, sizes))


class FakeKey:

    def __init__(self, value):
        self.value = value

    def to_string(self):
        return self.value


class FakeQueue:

    def __init__(self, items=None):
        self.items = list(items or [])
        self.put_items = []
        self.task_done_calls = 0

    def put(self, item):
        self.put_items.append(item)

    def get(self):
        if not self.items:
            raise KeyboardInterrupt()
        item = self.items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def task_done(self):
        self.task_done_calls += 1


class FakeTokenDatabase:

    def __init__(self):
        self.decode_calls = []
        self.prepare_calls = []
        self.prepare_layer_calls = []
        self.process_result = []

    def process_tokens(self, token_len, block_hashes, mask_num=0):
        return list(self.process_result)

    def prepare_value(self, start, end, block_ids):
        self.prepare_calls.append((start, end, list(block_ids)))
        return [start + 100], [end - start], block_ids[start // 2 if block_ids else 0]

    def decode_adaptor_prefill_pp(self, keys, addrs, sizes):
        self.decode_calls.append((list(keys), list(addrs), list(sizes)))
        return [f"decoded:{k}" for k in keys], addrs, sizes

    def prepare_value_layer(self, start, end, block_ids, layer_id):
        self.prepare_layer_calls.append((start, end, list(block_ids), layer_id))
        return [start + layer_id], [end - start]


class ConcreteTransferThread(transfer_module.KVTransferThread):

    def _handle_request(self, req_meta):
        self.handled.append(req_meta)
        self.request_queue.task_done()


def make_req_meta(req_id="req", *, token_len_chunk=4, block_ids=None, block_hashes=None, load_spec=None, current_event=None):
    return SimpleNamespace(
        req_id=req_id,
        token_len_chunk=token_len_chunk,
        block_ids=block_ids or [10, 11],
        block_hashes=block_hashes or ["h1", "h2"],
        current_event=current_event,
        token_ids=[1, 2, 3, 4],
        original_block_size=2,
        load_spec=load_spec,
    )


def make_layer_req(req_id="layer", *, starts=None, ends=None, keys=None, layer_id=0, is_last_chunk=False, current_event=None):
    return SimpleNamespace(
        req_id=req_id,
        starts=[0, 2] if starts is None else starts,
        ends=[2, 4] if ends is None else ends,
        keys=[FakeKey("k1"), FakeKey("k2")] if keys is None else keys,
        layer_id=layer_id,
        current_event=current_event,
        is_last_chunk=is_last_chunk,
        block_ids=[7, 8],
    )


def test_transfer_thread_basic_helpers_and_lookup_paths(monkeypatch):
    store = FakeStore()
    token_db = FakeTokenDatabase()
    thread = ConcreteTransferThread(store, token_db, 2, 1, 1, SimpleNamespace(set=lambda: None), name="base")
    thread.handled = []
    thread.request_queue = FakeQueue()

    assert thread.add_request("payload") is None
    assert thread.request_queue.put_items == ["payload"]

    thread.set_finished_request("req-1")
    thread.set_finished_request("req-2")
    assert thread.get_and_clear_finished_requests() == {"req-1", "req-2"}
    assert thread.get_and_clear_finished_requests() == set()

    store.exists_result = [1, 1, 0]
    assert thread.lookup(["a", "b", "c"]) == 2
    store.exists_result = [1, 1]
    assert thread.lookup(["a", "b"]) == 2

    logged = []
    monkeypatch.setattr(transfer_module.logger, "error", lambda *args: logged.append(args))
    store.exists = lambda keys: (_ for _ in ()).throw(RuntimeError("lookup boom"))
    assert thread.lookup(["x"]) == 0
    assert logged and "Remote connection failed in contains" in logged[0][0]

    thread.update_kv_event(["evt1", "evt2"])
    assert thread.get_kv_events() == ["evt1", "evt2"]
    assert thread.get_kv_events() == []
    assert transfer_module.KVTransferThread._handle_request(thread, "noop") is None


def test_transfer_thread_run_handles_none_payload_real_payload_and_exception(monkeypatch):
    store = FakeStore()
    token_db = FakeTokenDatabase()
    ready_event = SimpleNamespace(set=lambda: setattr(ready_event, "set_called", True))
    thread = ConcreteTransferThread(store, token_db, 2, 0, 1, ready_event, name="runner")
    thread.handled = []
    thread.request_queue = FakeQueue([None, "real", RuntimeError("boom")])
    warnings = []
    errors = []
    monkeypatch.setattr(transfer_module.logger, "warning", lambda *args: warnings.append(args))
    monkeypatch.setattr(transfer_module.logger, "error", lambda *args: errors.append(args))

    with pytest.raises(KeyboardInterrupt):
        thread.run()

    assert store.set_device_calls == 1
    assert ready_event.set_called is True
    assert warnings and "Received a None request!" in warnings[0][0]
    assert thread.handled == ["real"]
    assert errors and "Error in KVCacheTransferThread" in errors[0][0]


def test_sending_thread_stored_request_counters():
    thread = transfer_module.KVCacheStoreSendingThread(FakeStore(), FakeTokenDatabase(), 2, 0, 1, 1, "kv_both", SimpleNamespace(set=lambda: None))

    thread.add_stored_request("req")
    thread.add_stored_request("req")
    assert thread.stored_requests["req"] == 2
    thread.dec_stored_request("req")
    assert thread.stored_requests["req"] == 1
    thread.delete_finished_stored_request("req")
    assert "req" not in thread.stored_requests
    thread.dec_stored_request("missing")
    thread.delete_finished_stored_request("missing")


def test_sending_thread_handle_request_covers_missing_req_empty_keys_and_all_skipped(monkeypatch):
    store = FakeStore()
    token_db = FakeTokenDatabase()
    token_db.process_result = [(0, 2, FakeKey("k1"))]
    queue = FakeQueue()
    thread = transfer_module.KVCacheStoreSendingThread(store, token_db, 2, 0, 1, 1, "kv_both", SimpleNamespace(set=lambda: None))
    thread.request_queue = queue

    thread._handle_request(make_req_meta(req_id="missing"))
    assert queue.task_done_calls == 1

    queue.task_done_calls = 0
    thread.add_stored_request("empty")
    token_db.process_result = []
    thread._handle_request(make_req_meta(req_id="empty"))
    assert thread.stored_requests["empty"] == 0
    assert queue.task_done_calls == 0

    thread.add_stored_request("skip-all")
    token_db.process_result = [(0, 2, FakeKey("k1"))]
    monkeypatch.setattr(thread, "lookup", lambda keys: len(keys))
    thread._handle_request(make_req_meta(req_id="skip-all"))
    assert thread.stored_requests["skip-all"] == 0


def test_sending_thread_handle_request_main_path_with_events_and_consumer_decode(monkeypatch):
    store = FakeStore()
    token_db = FakeTokenDatabase()
    token_db.process_result = [(0, 2, FakeKey("k1")), (2, 4, FakeKey("k2"))]
    queue = FakeQueue()
    event = SimpleNamespace(synchronize=lambda: setattr(event, "sync_called", True))
    thread = transfer_module.KVCacheStoreSendingThread(store, token_db, 2, 0, 1, 1, "kv_consumer", SimpleNamespace(set=lambda: None), enable_kv_event=True)
    thread.request_queue = queue
    thread.add_stored_request("req-main")
    monkeypatch.setattr(thread, "lookup", lambda keys: 1)
    monkeypatch.setattr(transfer_module, "maybe_convert_block_hash", lambda value: f"hash:{value}")
    monkeypatch.setattr(transfer_module, "BlockStored", lambda **kwargs: SimpleNamespace(**kwargs))

    req_meta = make_req_meta(req_id="req-main", block_hashes=["b1", "b2"], current_event=event)
    thread._handle_request(req_meta)

    assert event.sync_called is True
    assert token_db.decode_calls
    assert store.put_calls[0][0] == ["decoded:k2"]
    assert thread.get_kv_events()[0].block_hashes == ["hash:b2"]
    assert queue.task_done_calls == 1
    assert thread.stored_requests["req-main"] == 0


def test_sending_thread_handle_request_main_path_without_events_or_decode(monkeypatch):
    store = FakeStore()
    token_db = FakeTokenDatabase()
    token_db.process_result = [(0, 2, FakeKey("k1")), (2, 4, FakeKey("k2"))]
    queue = FakeQueue()
    thread = transfer_module.KVCacheStoreSendingThread(store, token_db, 2, 0, 2, 1, "kv_both", SimpleNamespace(set=lambda: None), enable_kv_event=False)
    thread.request_queue = queue
    thread.add_stored_request("req-plain")
    monkeypatch.setattr(thread, "lookup", lambda keys: 0)
    monkeypatch.setattr(transfer_module, "maybe_convert_block_hash", lambda value: value)

    req_meta = make_req_meta(req_id="req-plain", current_event=None)
    thread._handle_request(req_meta)

    assert token_db.decode_calls == []
    assert store.put_calls == [(["k1", "k2"], [[100], [102]], [[2], [2]])]
    assert thread.get_kv_events() == []
    assert queue.task_done_calls == 1


def test_recving_thread_handle_request_rotates_lists_and_marks_finished():
    store = FakeStore()
    token_db = FakeTokenDatabase()
    token_db.process_result = [(0, 2, FakeKey("k1")), (2, 4, FakeKey("k2"))]
    queue = FakeQueue()
    load_spec = SimpleNamespace(token_len=4, vllm_cached_tokens=2)
    thread = transfer_module.KVCacheStoreRecvingThread(store, token_db, 2, 1, 1, SimpleNamespace(set=lambda: None))
    thread.request_queue = queue

    thread._handle_request(make_req_meta(req_id="recv", load_spec=load_spec, block_hashes=["b1", "b2"]))

    assert store.get_calls == [(["k2", "k1"], [[102], [100]], [[2], [2]])]
    assert thread.get_and_clear_finished_requests() == {"recv"}
    assert queue.task_done_calls == 1


def test_layer_sending_thread_covers_no_keys_skip_all_and_main_path(monkeypatch):
    store = FakeStore()
    token_db = FakeTokenDatabase()
    queue = FakeQueue()
    event = SimpleNamespace(synchronize=lambda: setattr(event, "sync_called", True))
    infos = []
    monkeypatch.setattr(transfer_module.logger, "info", lambda *args: infos.append(args))

    thread = transfer_module.KVCacheStoreLayerSendingThread(store, token_db, 2, 0, 1, 1, SimpleNamespace(set=lambda: None), num_layers=2)
    thread.request_queue = queue

    assert thread.add_request("payload") is None
    assert queue.put_items == ["payload"]

    thread._handle_request(make_layer_req(req_id="done-empty", keys=[], is_last_chunk=True, layer_id=0))
    assert thread.get_and_clear_finished_requests() == {"done-empty"}

    thread._handle_request(make_layer_req(req_id="not-done-empty", keys=[], is_last_chunk=False, layer_id=0))
    assert thread.get_and_clear_finished_requests() == set()

    monkeypatch.setattr(thread, "lookup", lambda keys: len(keys))
    thread._handle_request(make_layer_req(req_id="done-skip", is_last_chunk=True, layer_id=1))
    assert thread.get_and_clear_finished_requests() == {"done-skip"}

    monkeypatch.setattr(thread, "lookup", lambda keys: 1)
    thread._handle_request(make_layer_req(req_id="done-main", layer_id=1, is_last_chunk=True, current_event=event))
    assert event.sync_called is True
    assert store.put_calls
    assert thread.get_and_clear_finished_requests() == {"done-main"}
    assert queue.task_done_calls == 1
    assert infos and "Storing KV cache" in infos[0][0]


def test_layer_sending_thread_main_path_without_sync_or_finish(monkeypatch):
    store = FakeStore()
    token_db = FakeTokenDatabase()
    queue = FakeQueue()
    thread = transfer_module.KVCacheStoreLayerSendingThread(store, token_db, 2, 0, 2, 1, SimpleNamespace(set=lambda: None), num_layers=3)
    thread.request_queue = queue
    monkeypatch.setattr(thread, "lookup", lambda keys: 0)

    thread._handle_request(make_layer_req(req_id="mid-layer", layer_id=1, is_last_chunk=False, current_event=None))

    assert store.put_calls == [(["k1", "k2"], [[1], [3]], [[2], [2]])]
    assert thread.get_and_clear_finished_requests() == set()
    assert queue.task_done_calls == 1


def test_layer_recving_thread_add_request_and_handle_request_sets_event():
    store = FakeStore()
    token_db = FakeTokenDatabase()
    get_event = SimpleNamespace(set=lambda: setattr(get_event, "set_called", True))
    thread = transfer_module.KVCacheStoreLayerRecvingThread(store, token_db, 2, 1, 1, SimpleNamespace(set=lambda: None), get_event)
    queue = FakeQueue()
    thread.request_queue = queue

    assert thread.add_request("payload") is None
    assert queue.put_items == ["payload"]

    thread._handle_request(make_layer_req())
    assert store.get_calls == [(["k2", "k1"], [[2], [0]], [[2], [2]])]
    assert queue.task_done_calls == 1
    assert get_event.set_called is True
