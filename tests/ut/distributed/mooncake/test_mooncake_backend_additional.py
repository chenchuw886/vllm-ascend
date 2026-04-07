import builtins
import json
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import mooncake_backend as mooncake_module


class FakeMooncakeStore:

    def __init__(self, setup_result=0):
        self.setup_result = setup_result
        self.setup_calls = []
        self.exists_result = [1]
        self.put_result = [0]
        self.get_result = [0]

    def setup(self, *args):
        self.setup_calls.append(args)
        return self.setup_result

    def batch_is_exist(self, keys):
        return self.exists_result

    def batch_put_from_multi_buffers(self, keys, addrs, sizes):
        return self.put_result

    def batch_get_into_multi_buffers(self, keys, addrs, sizes):
        return self.get_result


class FakeTransferEngine:

    def __init__(self, rpc_port=9999, engine="engine-handle"):
        self._rpc_port = rpc_port
        self._engine = engine

    def get_rpc_port(self):
        return self._rpc_port

    def get_engine(self):
        return self._engine


@pytest.fixture
def fake_store_module(monkeypatch):
    store = FakeMooncakeStore()
    module = types.ModuleType("mooncake.store")
    module.MooncakeDistributedStore = lambda: store
    monkeypatch.setitem(sys.modules, "mooncake.store", module)
    return store


def test_mooncake_backend_import_error(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "mooncake.store":
            raise ImportError("missing mooncake")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="Please install mooncake"):
        mooncake_module.MooncakeBackend(SimpleNamespace(rank=0))


def test_mooncake_store_config_from_file_and_env(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "mooncake.json"
        config_path.write_text(
            json.dumps(
                {
                    "metadata_server": "meta:1",
                    "global_segment_size": "2KB",
                    "local_buffer_size": "3KB",
                    "protocol": "ascend",
                    "device_name": "npu0",
                    "master_server_address": "master:2",
                }
            )
        )

        config = mooncake_module.MooncakeStoreConfig.from_file(str(config_path))
        assert config == mooncake_module.MooncakeStoreConfig(
            metadata_server="meta:1",
            global_segment_size=2 * 1024,
            local_buffer_size=3 * 1024,
            protocol="ascend",
            device_name="npu0",
            master_server_address="master:2",
        )

        monkeypatch.setenv("MOONCAKE_CONFIG_PATH", str(config_path))
        loaded = mooncake_module.MooncakeStoreConfig.load_from_env()
        assert loaded == config

        monkeypatch.delenv("MOONCAKE_CONFIG_PATH", raising=False)
        with pytest.raises(ValueError, match="MOONCAKE_CONFIG_PATH"):
            mooncake_module.MooncakeStoreConfig.load_from_env()


def test_mooncake_backend_init_ascend_without_fabric_mem(fake_store_module, monkeypatch):
    config = mooncake_module.MooncakeStoreConfig(
        metadata_server="meta:1",
        global_segment_size=100,
        local_buffer_size=200,
        protocol="ascend",
        device_name="dev0",
        master_server_address="master:2",
    )
    monkeypatch.setattr(mooncake_module.MooncakeStoreConfig, "load_from_env", staticmethod(lambda: config))
    monkeypatch.setattr(mooncake_module, "get_ip", lambda: "127.0.0.1")
    monkeypatch.setattr(mooncake_module.os, "getenv", lambda key, default=None: "0" if key == "ASCEND_ENABLE_USE_FABRIC_MEM" else default)
    monkeypatch.setattr(
        mooncake_module.global_te,
        "get_transfer_engine",
        lambda hostname, device_name=None: FakeTransferEngine(rpc_port=4321, engine="te-engine"),
    )

    backend = mooncake_module.MooncakeBackend(SimpleNamespace(rank=4))

    assert backend.rank == 4
    assert backend.local_seg == "127.0.0.1:4321"
    assert fake_store_module.setup_calls == [
        ("127.0.0.1:4321", "meta:1", 100, 200, "ascend", "dev0", "master:2", "te-engine")
    ]


def test_mooncake_backend_init_ascend_with_fabric_mem(fake_store_module, monkeypatch):
    config = mooncake_module.MooncakeStoreConfig(
        metadata_server="meta:1",
        global_segment_size=100,
        local_buffer_size=200,
        protocol="ascend",
        device_name="dev0",
        master_server_address="master:2",
    )
    monkeypatch.setattr(mooncake_module.MooncakeStoreConfig, "load_from_env", staticmethod(lambda: config))
    monkeypatch.setattr(mooncake_module, "get_ip", lambda: "10.0.0.8")
    monkeypatch.setattr(mooncake_module.os, "getenv", lambda key, default=None: "1" if key == "ASCEND_ENABLE_USE_FABRIC_MEM" else default)

    backend = mooncake_module.MooncakeBackend(SimpleNamespace(rank=1))

    assert backend.local_seg == "10.0.0.8"
    assert fake_store_module.setup_calls == [
        ("10.0.0.8", "meta:1", 100, 0, "ascend", "dev0", "master:2")
    ]


def test_mooncake_backend_init_logs_and_raises_on_setup_failure(fake_store_module, monkeypatch):
    fake_store_module.setup_result = -1
    logged = []
    config = mooncake_module.MooncakeStoreConfig(
        metadata_server="meta:1",
        global_segment_size=100,
        local_buffer_size=200,
        protocol="ascend",
        device_name="dev0",
        master_server_address="master:2",
    )
    monkeypatch.setattr(mooncake_module.MooncakeStoreConfig, "load_from_env", staticmethod(lambda: config))
    monkeypatch.setattr(mooncake_module, "get_ip", lambda: "127.0.0.1")
    monkeypatch.setattr(mooncake_module.os, "getenv", lambda key, default=None: "1" if key == "ASCEND_ENABLE_USE_FABRIC_MEM" else default)
    monkeypatch.setattr(mooncake_module.logger, "error", lambda *args: logged.append(args))

    with pytest.raises(RuntimeError, match="Initialize mooncake failed"):
        mooncake_module.MooncakeBackend(SimpleNamespace(rank=0))

    assert logged and logged[0][0] == "Initialize mooncake failed."


def test_mooncake_backend_non_ascend_protocol_currently_raises_unbound_local(fake_store_module, monkeypatch):
    config = mooncake_module.MooncakeStoreConfig(
        metadata_server="meta:1",
        global_segment_size=100,
        local_buffer_size=200,
        protocol="tcp",
        device_name="dev0",
        master_server_address="master:2",
    )
    monkeypatch.setattr(mooncake_module.MooncakeStoreConfig, "load_from_env", staticmethod(lambda: config))

    with pytest.raises(UnboundLocalError):
        mooncake_module.MooncakeBackend(SimpleNamespace(rank=0))


def test_mooncake_backend_set_device_register_buffer_exists_put_get(fake_store_module, monkeypatch):
    backend = object.__new__(mooncake_module.MooncakeBackend)
    backend.rank = 6
    backend.store = fake_store_module
    set_device_calls = []
    register_calls = []
    logged = []

    monkeypatch.setattr(mooncake_module.torch.npu, "set_device", lambda device: set_device_calls.append(device))
    monkeypatch.setattr(mooncake_module.os, "getenv", lambda key, default=None: "0" if key == "ASCEND_ENABLE_USE_FABRIC_MEM" else default)
    monkeypatch.setattr(mooncake_module.global_te, "register_buffer", lambda ptrs, lengths: register_calls.append((ptrs, lengths)))
    monkeypatch.setattr(mooncake_module.logger, "error", lambda *args: logged.append(args))

    backend.set_device()
    assert str(set_device_calls[0]) == "npu:6"

    backend.register_buffer([1, 2], [3, 4])
    assert register_calls == [([1, 2], [3, 4])]

    monkeypatch.setattr(mooncake_module.os, "getenv", lambda key, default=None: "1" if key == "ASCEND_ENABLE_USE_FABRIC_MEM" else default)
    backend.register_buffer([5], [6])
    assert register_calls == [([1, 2], [3, 4])]

    assert backend.exists(["a"]) == [1]

    backend.put(["k"], [[1]], [[2]])
    backend.get(["k"], [[1]], [[2]])
    assert logged == []

    fake_store_module.put_result = [-1]
    fake_store_module.get_result = [-1]
    backend.put(["k2"], [[3]], [[4]])
    backend.get(["k2"], [[3]], [[4]])
    assert any("Failed to put key" in args[0] for args in logged)
    assert any("Failed to get key" in args[0] for args in logged)

    fake_store_module.batch_put_from_multi_buffers = lambda *args: (_ for _ in ()).throw(RuntimeError("put boom"))
    fake_store_module.batch_get_into_multi_buffers = lambda *args: (_ for _ in ()).throw(RuntimeError("get boom"))
    backend.put(["k3"], [[5]], [[6]])
    backend.get(["k3"], [[5]], [[6]])
    assert any("error" in args[0] for args in logged)


def test_mooncake_parse_empty_string_and_convert_overflow():
    with pytest.raises(ValueError, match="cannot be empty"):
        mooncake_module._parse_global_segment_size("   ")

    with pytest.raises(ValueError, match="Storage size too large"):
        mooncake_module._convert_to_bytes("1e309", 1, "1e309")
