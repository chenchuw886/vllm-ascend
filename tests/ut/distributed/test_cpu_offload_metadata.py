import pickle
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.kv_pool.cpu_offload import metadata as metadata_module


class FakeTensor:

    def __init__(self):
        self.reshape_calls = []
        self.split_calls = []

    def reshape(self, shape):
        self.reshape_calls.append(shape)
        return self

    def split(self, sizes, dim=-1):
        self.split_calls.append((tuple(sizes), dim))
        return ("key", "value")


class FakeSharedMemory:

    def __init__(self, name="fake", create=False, size=0):
        self.name = name
        self.create = create
        self.size = size
        self.buf = bytearray(size or 8)
        self.closed = False
        self.unlinked = False

    def close(self):
        self.closed = True

    def unlink(self):
        self.unlinked = True


class FakeMLAAttentionSpec:
    pass


def make_vllm_config(kv_transfer_config):
    return SimpleNamespace(
        kv_transfer_config=kv_transfer_config,
        parallel_config=SimpleNamespace(world_size=4, pipeline_parallel_size=2),
        cache_config=SimpleNamespace(enable_prefix_caching=True),
    )


def make_kv_transfer_config(connector_name, extra=None):
    extra = extra or {}
    return SimpleNamespace(
        kv_connector=connector_name,
        kv_connector_extra_config=extra,
        get_from_extra_config=lambda key, default: extra.get(key, default),
    )


def make_server(**kwargs):
    server = metadata_module.MetadataServer.__new__(metadata_module.MetadataServer)
    server.world_size = kwargs.get("world_size", 4)
    server.pipeline_parallel_size = kwargs.get("pipeline_parallel_size", 2)
    server.available_memory = kwargs.get("available_memory", 800)
    server.shared_memory = kwargs.get("shared_memory", {})
    server.num_cpu_blocks = kwargs.get("num_cpu_blocks", -1)
    server.functions = kwargs.get("functions", {})
    server.socket = kwargs.get("socket", MagicMock())
    server.ctx = kwargs.get("ctx", MagicMock())
    if "layer" in kwargs:
        server.layer = kwargs["layer"]
    return server


def test_get_cpu_offload_connector_returns_direct_match():
    kv_transfer_config = make_kv_transfer_config("CPUOffloadingConnector")

    result = metadata_module.get_cpu_offload_connector(make_vllm_config(kv_transfer_config))

    assert result is kv_transfer_config


def test_get_cpu_offload_connector_searches_multi_connector(monkeypatch):
    multi_connector = make_kv_transfer_config(
        "MultiConnector",
        extra={
            "connectors": [
                {"kv_connector": "OtherConnector"},
                {"kv_connector": "CPUOffloadingConnector", "cpu_swap_space_gb": 32},
            ]
        },
    )

    monkeypatch.setattr(metadata_module, "KVTransferConfig", lambda **kwargs: SimpleNamespace(**kwargs))

    result = metadata_module.get_cpu_offload_connector(make_vllm_config(multi_connector))

    assert result.kv_connector == "CPUOffloadingConnector"
    assert result.cpu_swap_space_gb == 32


def test_get_cpu_offload_connector_returns_none_when_missing():
    assert metadata_module.get_cpu_offload_connector(make_vllm_config(None)) is None
    assert metadata_module.get_cpu_offload_connector(make_vllm_config(make_kv_transfer_config("OtherConnector"))) is None


def test_get_cpu_offload_connector_returns_none_when_multi_connector_has_no_match(monkeypatch):
    multi_connector = make_kv_transfer_config(
        "MultiConnector",
        extra={"connectors": [{"kv_connector": "OtherConnector"}]},
    )
    monkeypatch.setattr(metadata_module, "KVTransferConfig", lambda **kwargs: SimpleNamespace(**kwargs))

    assert metadata_module.get_cpu_offload_connector(make_vllm_config(multi_connector)) is None


def test_zmq_rpc_client_init_builds_default_identity(monkeypatch):
    fake_context = MagicMock()
    fake_socket = MagicMock()

    monkeypatch.setattr(metadata_module.os, "getpid", lambda: 321)
    monkeypatch.setattr(metadata_module.zmq, "Context", lambda: fake_context)
    monkeypatch.setattr(metadata_module, "make_zmq_socket", lambda *args, **kwargs: fake_socket)

    client = metadata_module.MetadataServer.ZMQRPCClient()

    assert client.ctx is fake_context
    assert client.socket is fake_socket


def test_zmq_rpc_client_init_uses_explicit_identity(monkeypatch):
    fake_context = MagicMock()
    captured = {}

    def fake_make_zmq_socket(*args, **kwargs):
        captured.update(kwargs)
        return MagicMock()

    monkeypatch.setattr(metadata_module.zmq, "Context", lambda: fake_context)
    monkeypatch.setattr(metadata_module, "make_zmq_socket", fake_make_zmq_socket)

    metadata_module.MetadataServer.ZMQRPCClient(identity="worker-explicit")

    assert captured["identity"] == b"worker-explicit"


def test_zmq_rpc_client_call_returns_regular_result(monkeypatch):
    client = metadata_module.MetadataServer.ZMQRPCClient.__new__(metadata_module.MetadataServer.ZMQRPCClient)
    client.socket = MagicMock()
    client.ctx = MagicMock()
    client.socket.recv.side_effect = [b"", pickle.dumps((123, None))]

    result = client.call("ready")

    assert result == 123
    assert client.socket.send.call_count == 2


def test_zmq_rpc_client_call_raises_remote_error(monkeypatch):
    client = metadata_module.MetadataServer.ZMQRPCClient.__new__(metadata_module.MetadataServer.ZMQRPCClient)
    client.socket = MagicMock()
    client.ctx = MagicMock()
    client.socket.recv.side_effect = [b"", pickle.dumps((None, RuntimeError("rpc failed")))]
    logged = []
    monkeypatch.setattr(metadata_module.logger, "exception", lambda message: logged.append(message))

    with pytest.raises(RuntimeError, match="rpc failed"):
        client.call("ready")

    assert any("call metadata sever error" in message for message in logged)


def test_zmq_rpc_client_call_builds_cpu_kv_cache_tensors(monkeypatch):
    shm = FakeSharedMemory(size=16)
    client = metadata_module.MetadataServer.ZMQRPCClient.__new__(metadata_module.MetadataServer.ZMQRPCClient)
    client.socket = MagicMock()
    client.ctx = MagicMock()
    client.socket.recv.side_effect = [
        b"",
        pickle.dumps((({"layer": shm}, (2, 2), torch.float32, metadata_module.MLAConfig(nope_dim=1, rope_dim=1)), None)),
    ]
    fake_tensor = FakeTensor()
    monkeypatch.setattr(metadata_module.torch, "frombuffer", lambda *_args, **_kwargs: fake_tensor)

    result = client.call("init_cpu_kv_caches")

    assert result == {"layer": ("key", "value")}
    assert set(client.shared_memory_dict.keys()) == {"layer"}
    assert isinstance(client.shared_memory_dict["layer"], FakeSharedMemory)
    assert client.shared_memory_dict["layer"].size == 16
    assert fake_tensor.reshape_calls == [(2, 2)]
    assert fake_tensor.split_calls == [((1, 1), -1)]


def test_zmq_rpc_client_call_builds_non_mla_cpu_kv_cache_tensors(monkeypatch):
    client = metadata_module.MetadataServer.ZMQRPCClient.__new__(metadata_module.MetadataServer.ZMQRPCClient)
    client.socket = MagicMock()
    client.ctx = MagicMock()
    client.socket.recv.side_effect = [
        b"",
        pickle.dumps((({"layer": FakeSharedMemory(size=8)}, (1, 2), torch.float32, None), None)),
    ]
    fake_tensor = FakeTensor()
    monkeypatch.setattr(metadata_module.torch, "frombuffer", lambda *_args, **_kwargs: fake_tensor)

    result = client.call("init_cpu_kv_caches")

    assert result == {"layer": fake_tensor}
    assert fake_tensor.reshape_calls == [(1, 2)]
    assert fake_tensor.split_calls == []


def test_zmq_rpc_client_del_closes_resources():
    client = metadata_module.MetadataServer.ZMQRPCClient.__new__(metadata_module.MetadataServer.ZMQRPCClient)
    client.socket = MagicMock()
    client.ctx = MagicMock()
    client.shared_memory_dict = {"layer": FakeSharedMemory()}

    client.__del__()

    client.socket.close.assert_called_once()
    client.ctx.term.assert_called_once()
    assert client.shared_memory_dict["layer"].closed is True


def test_metadata_server_init_sets_up_socket_and_defaults(monkeypatch):
    fake_context = MagicMock()
    fake_socket = MagicMock()
    kv_transfer_config = make_kv_transfer_config("CPUOffloadingConnector", extra={"cpu_swap_space_gb": 2})

    monkeypatch.setattr(metadata_module, "get_cpu_offload_connector", lambda _cfg: kv_transfer_config)
    monkeypatch.setattr(metadata_module.zmq, "Context", lambda: fake_context)
    monkeypatch.setattr(metadata_module, "make_zmq_socket", lambda *args, **kwargs: fake_socket)

    server = metadata_module.MetadataServer(make_vllm_config(kv_transfer_config))

    assert server.available_memory == 2 * 1024 * 1024 * 1024
    assert server.socket is fake_socket
    assert set(server.functions) == {"init_cpu_kv_caches", "post_init", "ready"}
    assert server.num_cpu_blocks == -1


def test_safe_create_shared_memory_replaces_existing_segment(monkeypatch):
    existing = FakeSharedMemory(name="cache")
    created = FakeSharedMemory(name="cache", create=True, size=16)
    calls = []

    def fake_shared_memory(name, create=False, size=0):
        calls.append((name, create, size))
        if create:
            return created
        return existing

    monkeypatch.setattr(metadata_module, "SharedMemory", fake_shared_memory)

    result = metadata_module.MetadataServer._safe_create_shared_memory("cache", 16)

    assert result is created
    assert existing.closed is True
    assert existing.unlinked is True
    assert calls == [("cache", False, 0), ("cache", True, 16)]


def test_safe_create_shared_memory_handles_missing_existing_segment(monkeypatch):
    created = FakeSharedMemory(name="new-cache", create=True, size=8)

    def fake_shared_memory(name, create=False, size=0):
        if create:
            return created
        raise FileNotFoundError()

    monkeypatch.setattr(metadata_module, "SharedMemory", fake_shared_memory)

    result = metadata_module.MetadataServer._safe_create_shared_memory("new-cache", 8)

    assert result is created


def test_metadata_server_ready_returns_true():
    assert metadata_module.MetadataServer.ready(make_server()) is True


def test_init_cpu_kv_caches_builds_non_mla_shared_memory(monkeypatch):
    layer = SimpleNamespace(page_size_bytes=10, block_size=4, num_kv_heads=2, head_size=8, dtype=torch.float32)
    created = []
    server = make_server(available_memory=800, world_size=4, pipeline_parallel_size=2)

    monkeypatch.setattr(metadata_module.MetadataServer, "_safe_create_shared_memory", lambda name, size: created.append((name, size)) or FakeSharedMemory(name=name, size=size))
    monkeypatch.setattr(metadata_module, "get_dtype_size", lambda dtype: 4)
    monkeypatch.setattr(metadata_module, "MLAAttentionSpec", FakeMLAAttentionSpec)

    result = server.init_cpu_kv_caches(1, 2, {"layer1": layer, "layer2": layer}, mla_config=None)

    assert result[1] == (2, 10, 4, 2, 8)
    assert result[2] == torch.float32
    assert result[3] is None
    assert server.num_cpu_blocks == 10
    assert server.layer is layer
    assert len(created) == 2
    assert server.init_cpu_kv_caches(1, 2, {"layer1": layer, "layer2": layer}, mla_config=None) is result


def test_init_cpu_kv_caches_builds_mla_shared_memory(monkeypatch):
    class FakeMLALayer(FakeMLAAttentionSpec):
        pass

    layer = FakeMLALayer()
    layer.page_size_bytes = 10
    layer.block_size = 4
    layer.num_kv_heads = 2
    layer.head_size = 8
    layer.dtype = torch.float16
    server = make_server(available_memory=400, world_size=4, pipeline_parallel_size=2)
    mla_config = metadata_module.MLAConfig(nope_dim=5, rope_dim=3)

    monkeypatch.setattr(metadata_module.MetadataServer, "_safe_create_shared_memory", lambda name, size: FakeSharedMemory(name=name, size=size))
    monkeypatch.setattr(metadata_module, "get_dtype_size", lambda dtype: 2)
    monkeypatch.setattr(metadata_module, "MLAAttentionSpec", FakeMLAAttentionSpec)

    result = server.init_cpu_kv_caches(3, 7, {"layer": layer}, mla_config=mla_config)

    assert (3, 0) in server.shared_memory
    assert result[1] == (20, 4, 2, 8)
    assert result[2] == torch.float16
    assert result[3] is mla_config
    assert server.num_cpu_blocks == 20


def test_init_cpu_kv_caches_keeps_smaller_num_cpu_blocks(monkeypatch):
    layer = SimpleNamespace(page_size_bytes=10, block_size=4, num_kv_heads=2, head_size=8, dtype=torch.float32)
    server = make_server(available_memory=1600, world_size=4, pipeline_parallel_size=2, num_cpu_blocks=5)

    monkeypatch.setattr(metadata_module.MetadataServer, "_safe_create_shared_memory", lambda name, size: FakeSharedMemory(name=name, size=size))
    monkeypatch.setattr(metadata_module, "get_dtype_size", lambda dtype: 4)
    monkeypatch.setattr(metadata_module, "MLAAttentionSpec", FakeMLAAttentionSpec)

    result = server.init_cpu_kv_caches(0, 1, {"layer": layer}, mla_config=None)

    assert result[1] == (2, 40, 4, 2, 8)
    assert server.num_cpu_blocks == 5


def test_init_cpu_kv_caches_validates_specs_and_mla_dimensions(monkeypatch):
    server = make_server(available_memory=400)
    layer = SimpleNamespace(page_size_bytes=10, block_size=4, num_kv_heads=2, head_size=8, dtype=torch.float32)
    bad_layer = SimpleNamespace(page_size_bytes=11, block_size=4, num_kv_heads=2, head_size=8, dtype=torch.float32)

    with pytest.raises(AssertionError):
        server.init_cpu_kv_caches(0, 0, {"a": layer, "b": bad_layer}, mla_config=None)

    class FakeMLALayer(FakeMLAAttentionSpec):
        pass

    mla_layer = FakeMLALayer()
    mla_layer.page_size_bytes = 10
    mla_layer.block_size = 4
    mla_layer.num_kv_heads = 2
    mla_layer.head_size = 8
    mla_layer.dtype = torch.float32
    monkeypatch.setattr(metadata_module, "MLAAttentionSpec", FakeMLAAttentionSpec)
    monkeypatch.setattr(metadata_module.MetadataServer, "_safe_create_shared_memory", lambda name, size: FakeSharedMemory(name=name, size=size))
    monkeypatch.setattr(metadata_module, "get_dtype_size", lambda dtype: 4)

    with pytest.raises(AssertionError):
        server.init_cpu_kv_caches(0, 0, {"a": mla_layer}, mla_config=metadata_module.MLAConfig(nope_dim=1, rope_dim=1))


def test_post_init_creates_cpu_cache_manager_and_is_idempotent(monkeypatch):
    manager = SimpleNamespace(
        get_matched_num_and_touch=MagicMock(),
        allocate_slots=MagicMock(),
        record_request_cache_and_free_slots=MagicMock(),
        cache_and_free_slots=MagicMock(),
    )
    server = make_server(num_cpu_blocks=5, layer="layer", functions={})

    monkeypatch.setattr(metadata_module, "CPUKVCacheManager", lambda layer, num_blocks: manager)

    server.post_init()

    assert server.cpu_block_manager is manager
    assert set(server.functions) == {
        "get_matched_num_and_touch",
        "allocate_slots",
        "record_request_cache_and_free_slots",
        "cache_and_free_slots",
    }

    server.post_init()
    assert server.cpu_block_manager is manager


def test_post_init_requires_initialized_num_blocks(monkeypatch):
    server = make_server(num_cpu_blocks=-1, layer="layer")

    with pytest.raises(AssertionError):
        server.post_init()


def test_serve_step_handles_success_invalid_unknown_and_execution_error(monkeypatch):
    def run_case(raw_message, functions):
        socket = MagicMock()
        socket.recv.side_effect = [b"client", b"", raw_message]
        server = make_server(socket=socket, functions=functions)
        server.serve_step()
        response = pickle.loads(socket.send.call_args_list[-1].args[0])
        return response

    success = run_case(pickle.dumps(("ready", (), {})), {"ready": lambda: True})
    assert success == (True, None)

    invalid = run_case(b"not-pickle", {})
    assert invalid[0] is None
    assert "Invalid request" in str(invalid[1])

    unknown = run_case(pickle.dumps(("missing", (), {})), {})
    assert unknown[0] is None
    assert isinstance(unknown[1], NameError)

    logged = []
    monkeypatch.setattr(metadata_module.logger, "exception", lambda message: logged.append(message))
    failure = run_case(pickle.dumps(("boom", (), {})), {"boom": lambda: (_ for _ in ()).throw(ValueError("bad"))})
    assert failure[0] is None
    assert isinstance(failure[1], ValueError)
    assert any("metadata execute error" in message for message in logged)


def test_shutdown_closes_socket_ctx_and_shared_memory(monkeypatch):
    shm = FakeSharedMemory()
    socket = MagicMock()
    ctx = MagicMock()
    server = make_server(socket=socket, ctx=ctx, shared_memory={(0, 0): ({"layer": shm}, (), torch.float32, None)})
    removed = []

    monkeypatch.setattr(metadata_module.os.path, "exists", lambda _path: True)
    monkeypatch.setattr(metadata_module.os, "remove", lambda path: removed.append(path))

    server.shutdown()

    socket.close.assert_called_once()
    ctx.term.assert_called_once()
    assert removed == [metadata_module.MetadataServer.METADATA_SERVER_ADDRESS.replace("ipc://", "")]
    assert shm.closed is True
    assert shm.unlinked is True


def test_shutdown_skips_remove_when_socket_path_missing(monkeypatch):
    server = make_server(socket=MagicMock(), ctx=MagicMock(), shared_memory={})
    removed = []

    monkeypatch.setattr(metadata_module.os.path, "exists", lambda _path: False)
    monkeypatch.setattr(metadata_module.os, "remove", lambda path: removed.append(path))

    server.shutdown()

    assert removed == []


def test_run_metadata_server_returns_early_when_disabled(monkeypatch):
    cfg = SimpleNamespace(cache_config=SimpleNamespace(enable_prefix_caching=False), kv_transfer_config=None)

    assert metadata_module.MetadataServerProc.run_metadata_server(cfg) is None


def test_run_metadata_server_handles_system_exit_and_shutdown(monkeypatch):
    server = SimpleNamespace(serve_step=MagicMock(side_effect=SystemExit()), shutdown=MagicMock())
    cfg = make_vllm_config(make_kv_transfer_config("CPUOffloadingConnector"))
    captured = {}

    def fake_metadata_server(_cfg):
        import inspect

        captured["handler"] = inspect.currentframe().f_back.f_locals["_signal_handler"]
        return server

    monkeypatch.setattr(metadata_module, "get_cpu_offload_connector", lambda _cfg: object())
    monkeypatch.setattr(metadata_module, "MetadataServer", fake_metadata_server)

    with pytest.raises(SystemExit):
        metadata_module.MetadataServerProc.run_metadata_server(cfg)

    server.shutdown.assert_called_once()
    with pytest.raises(SystemExit):
        captured["handler"](0, None)
    assert captured["handler"](0, None) is None


def test_run_metadata_server_handles_exception_and_shutdown(monkeypatch):
    server = SimpleNamespace(serve_step=MagicMock(side_effect=RuntimeError("serve failed")), shutdown=MagicMock())
    cfg = make_vllm_config(make_kv_transfer_config("CPUOffloadingConnector"))
    logged = []

    monkeypatch.setattr(metadata_module, "get_cpu_offload_connector", lambda _cfg: object())
    monkeypatch.setattr(metadata_module, "MetadataServer", lambda _cfg: server)
    monkeypatch.setattr(metadata_module.logger, "exception", lambda message: logged.append(message))

    with pytest.raises(RuntimeError, match="serve failed"):
        metadata_module.MetadataServerProc.run_metadata_server(cfg)

    server.shutdown.assert_called_once()
    assert any("Metadata server error" in message for message in logged)