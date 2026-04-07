from types import SimpleNamespace

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata, KVConnectorRole

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import ascend_store_connector as connector_module


class FakeAggregator:

    def __init__(self, num_workers):
        self.num_workers = num_workers
        self.events = []
        self.cleared = False
        self.reset = False

    def add_events(self, events):
        self.events.extend(events)

    def get_common_events(self):
        return self.events[:1]

    def clear_events(self):
        self.events.clear()
        self.cleared = True

    def reset_workers(self):
        self.num_workers = 1
        self.reset = True

    def increment_workers(self, count):
        self.num_workers += count

    def get_all_events(self):
        return list(self.events)

    def get_number_of_workers(self):
        return self.num_workers


class FakeSocket:

    def __init__(self, frames):
        self.frames = list(frames)
        self.sent = []
        self.closed = None

    def recv_multipart(self, copy=False):
        if not self.frames:
            raise RuntimeError("stop")
        return self.frames.pop(0)

    def send(self, response):
        self.sent.append(response)

    def close(self, linger=0):
        self.closed = linger


class FakeThread:

    def __init__(self, target=None, daemon=False):
        self.target = target
        self.daemon = daemon
        self.started = False

    def start(self):
        self.started = True



def make_vllm_config(*, role="kv_both", use_layerwise=False, consumer_is_to_put=False, rank=0, connector_name="AscendStoreConnector"):
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_role=role,
            kv_connector="MooncakeConnectorStoreV1" if connector_name == "MooncakeConnectorStoreV1" else connector_name,
            kv_connector_extra_config={"use_layerwise": use_layerwise, "consumer_is_to_put": consumer_is_to_put},
        ),
        parallel_config=SimpleNamespace(rank=rank),
    )


def test_ascend_store_kv_events_wraps_aggregator(monkeypatch):
    monkeypatch.setattr(connector_module, "KVEventAggregator", FakeAggregator)

    events = connector_module.AscendStoreKVEvents(num_workers=2)
    events.add_events(["a", "b"])
    assert events.get_all_events() == ["a", "b"]
    assert events.get_number_of_workers() == 2

    aggregated = events.aggregate()
    assert aggregated is events
    assert events.get_all_events() == ["a"]

    events.increment_workers(3)
    assert events.get_number_of_workers() == 4
    events.clear_events()
    assert events.get_all_events() == []
    assert "AscendStoreKVEvents" in repr(events)


def test_connector_scheduler_side_methods_and_events(monkeypatch):
    scheduler = SimpleNamespace(
        get_num_new_matched_tokens=lambda request, num: (num + 1, True),
        update_state_after_alloc=lambda request, blocks, tokens: (request.request_id, tokens),
        build_connector_meta=lambda output: output,
        request_finished=lambda request, block_ids: (bool(block_ids), {"req": request.request_id}),
    )
    monkeypatch.setattr(connector_module, "KVPoolScheduler", lambda cfg, use_layerwise: scheduler)

    connector = connector_module.AscendStoreConnector(make_vllm_config(), KVConnectorRole.SCHEDULER)
    req = SimpleNamespace(request_id="req-1")
    assert connector.get_num_new_matched_tokens(req, 2) == (3, True)
    assert connector.update_state_after_alloc(req, None, 4) == ("req-1", 4)
    assert connector.build_connector_meta("meta") == "meta"
    assert connector.request_finished(req, [1]) == (True, {"req": "req-1"})

    assert list(connector.take_events()) == []
    connector.update_connector_output(SimpleNamespace(kv_cache_events=None))
    connector.update_connector_output(SimpleNamespace(kv_cache_events=KVConnectorMetadata()))
    assert connector._kv_cache_events is None

    monkeypatch.setattr(connector_module, "KVEventAggregator", FakeAggregator)
    first = connector_module.AscendStoreKVEvents(num_workers=1)
    first.add_events(["a"])
    second = connector_module.AscendStoreKVEvents(num_workers=2)
    second.add_events(["b"])
    connector.update_connector_output(SimpleNamespace(kv_cache_events=first))
    connector.update_connector_output(SimpleNamespace(kv_cache_events=second))
    assert list(connector.take_events()) == ["a"]
    assert connector._kv_cache_events is None


def test_connector_worker_side_methods_cover_noops_and_event_wrapping(monkeypatch):
    worker = SimpleNamespace(
        register_kv_caches=lambda kv: setattr(worker, "registered", kv),
        start_load_kv=lambda meta: setattr(worker, "start_meta", meta),
        wait_for_layer_load=lambda: setattr(worker, "waited", True),
        save_kv_layer=lambda meta: setattr(worker, "save_meta", meta),
        wait_for_save=lambda meta: setattr(worker, "wait_meta", meta),
        get_finished=lambda finished_ids, meta: ({"sent"}, {"recv"}),
        get_kv_events=lambda: ["evt"],
    )
    monkeypatch.setattr(connector_module, "KVPoolWorker", lambda cfg, use_layerwise: worker)

    metadata = SimpleNamespace(tag="metadata")
    monkeypatch.setattr(connector_module.AscendStoreConnector, "_get_connector_metadata", lambda self: metadata)
    connector = connector_module.AscendStoreConnector(make_vllm_config(use_layerwise=True, rank=1), KVConnectorRole.WORKER)

    connector.register_kv_caches({"layer": "cache"})
    connector.start_load_kv(None)
    connector.wait_for_layer_load("layer")
    connector.save_kv_layer("layer", None, None)
    connector.wait_for_save()
    assert worker.registered == {"layer": "cache"}
    assert worker.start_meta is metadata
    assert worker.waited is True
    assert worker.save_meta is metadata
    assert connector.get_finished({"done"}) == ({"sent"}, {"recv"})

    events = connector.get_kv_connector_kv_cache_events()
    assert isinstance(events, connector_module.AscendStoreKVEvents)
    assert events.get_all_events() == ["evt"]

    worker.get_kv_events = lambda: []
    assert connector.get_kv_connector_kv_cache_events() is None

    connector.use_layerwise = False
    assert connector.wait_for_layer_load("layer") is None
    assert connector.save_kv_layer("layer", None, None) is None

    connector.wait_for_save()
    assert worker.wait_meta is metadata

    connector.kv_role = "kv_consumer"
    connector.use_layerwise = True
    connector.consumer_is_to_put = False
    assert connector.save_kv_layer("layer", None, None) is None
    assert connector.wait_for_save() is None


def test_connector_worker_rank_zero_builds_lookup_server_and_warns_legacy(monkeypatch):
    worker = SimpleNamespace()
    created_lookup = []
    warnings = []
    monkeypatch.setattr(connector_module, "KVPoolWorker", lambda cfg, use_layerwise: worker)
    monkeypatch.setattr(connector_module, "LookupKeyServer", lambda pool_worker, cfg, use_layerwise: created_lookup.append((pool_worker, use_layerwise)) or "lookup")
    monkeypatch.setattr(connector_module.logger, "warning", lambda *args: warnings.append(args))

    connector = connector_module.AscendStoreConnector(
        make_vllm_config(rank=0, connector_name="MooncakeConnectorStoreV1"),
        KVConnectorRole.WORKER,
    )

    assert connector.lookup_server == "lookup"
    assert created_lookup == [(worker, False)]
    assert warnings

    connector_nonzero = connector_module.AscendStoreConnector(make_vllm_config(rank=1), KVConnectorRole.WORKER)
    assert not hasattr(connector_nonzero, "lookup_server")


def test_lookup_key_server_processes_request_and_close(monkeypatch):
    frames = [[(12).to_bytes(4, "big"), b"frame1", b"frame2"]]
    fake_socket = FakeSocket(frames)
    fake_thread = FakeThread()

    class FakeDecoder:
        def decode(self, frames):
            assert frames == [b"frame1", b"frame2"]
            return ["hash-a", "hash-b"]

    def fake_make_zmq_socket(*args, **kwargs):
        return fake_socket

    def fake_thread_factory(target=None, daemon=False):
        fake_thread.target = target
        fake_thread.daemon = daemon
        return fake_thread

    pool_worker = SimpleNamespace(lookup_scheduler=lambda token_len, hashes_str, use_layerwise: token_len + len(hashes_str))
    monkeypatch.setattr(connector_module, "MsgpackDecoder", lambda *args, **kwargs: FakeDecoder())
    monkeypatch.setattr(connector_module.zmq, "Context", lambda: "ctx")
    monkeypatch.setattr(connector_module, "make_zmq_socket", fake_make_zmq_socket)
    monkeypatch.setattr(connector_module, "get_zmq_rpc_path_lookup", lambda _cfg: "ipc://lookup")
    monkeypatch.setattr(connector_module.threading, "Thread", fake_thread_factory)

    server = connector_module.LookupKeyServer(pool_worker, make_vllm_config(), use_layerwise=True)
    assert fake_thread.started is True
    with pytest.raises(RuntimeError, match="stop"):
        fake_thread.target()
    assert int.from_bytes(fake_socket.sent[0], "big") == 14

    server.close()
    assert fake_socket.closed == 0


def test_lookup_key_server_process_loop_exits_cleanly_when_running_is_false(monkeypatch):
    fake_socket = FakeSocket([])
    fake_thread = FakeThread()

    monkeypatch.setattr(connector_module, "MsgpackDecoder", lambda *args, **kwargs: SimpleNamespace(decode=lambda frames: []))
    monkeypatch.setattr(connector_module.zmq, "Context", lambda: "ctx")
    monkeypatch.setattr(connector_module, "make_zmq_socket", lambda *args, **kwargs: fake_socket)
    monkeypatch.setattr(connector_module, "get_zmq_rpc_path_lookup", lambda _cfg: "ipc://lookup")
    monkeypatch.setattr(
        connector_module.threading,
        "Thread",
        lambda target=None, daemon=False: setattr(fake_thread, "target", target) or setattr(fake_thread, "daemon", daemon) or fake_thread,
    )

    server = connector_module.LookupKeyServer(SimpleNamespace(lookup_scheduler=lambda *args: 0), make_vllm_config(), use_layerwise=False)
    server.running = False

    assert fake_thread.target() is None
    assert fake_socket.sent == []
