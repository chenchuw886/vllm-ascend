import builtins
import sys
import types
from types import SimpleNamespace

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import backend as backend_module
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import memcache_backend as memcache_module


class FakeStore:

    def __init__(self):
        self.init_calls = []
        self.register_calls = []
        self.exist_result = [1, 0]
        self.get_result = [0]
        self.put_result = [0]

    def init(self, rank):
        self.init_calls.append(rank)
        return 0

    def register_buffer(self, ptr, size):
        self.register_calls.append((ptr, size))

    def batch_is_exist(self, keys):
        return self.exist_result

    def batch_get_into_layers(self, key, addr, size, direction):
        return self.get_result

    def batch_put_from_layers(self, key, addr, size, direction):
        return self.put_result


@pytest.fixture
def fake_memcache_module(monkeypatch):
    module = types.ModuleType("memcache_hybrid")
    store = FakeStore()
    module.DistributedObjectStore = lambda: store
    monkeypatch.setitem(sys.modules, "memcache_hybrid", module)
    return store


def test_backend_abstract_method_bodies_are_callable():
    class ConcreteBackend(backend_module.Backend):
        def __init__(self, parallel_config):
            return super().__init__(parallel_config)

        def set_device(self):
            return super().set_device()

        def register_buffer(self, ptrs, lengths):
            return super().register_buffer(ptrs, lengths)

        def exists(self, keys):
            return super().exists(keys)

        def put(self, keys, addrs, sizes):
            return super().put(keys, addrs, sizes)

        def get(self, keys, addrs, sizes):
            return super().get(keys, addrs, sizes)

    backend = ConcreteBackend(SimpleNamespace())

    assert backend.set_device() is None
    assert backend.register_buffer([1], [2]) is None
    assert backend.exists(["k"]) is None
    assert backend.put(["k"], [[1]], [[2]]) is None
    assert backend.get(["k"], [[1]], [[2]]) is None


def test_memcache_backend_init_import_error(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "memcache_hybrid":
            raise ImportError("missing memcache")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="Please install memcache"):
        memcache_module.MemcacheBackend(SimpleNamespace(rank=0))


def test_memcache_backend_init_non_a2_branch(fake_memcache_module, monkeypatch):
    monkeypatch.setattr(memcache_module, "get_ascend_device_type", lambda: memcache_module.AscendDeviceType.A3)

    backend = memcache_module.MemcacheBackend(SimpleNamespace(rank=3))

    assert backend.rank == 3
    assert backend.store is fake_memcache_module
    assert fake_memcache_module.init_calls == [3]


def test_memcache_backend_init_a2_branch_runs_all_gather(fake_memcache_module, monkeypatch):
    gathered = []
    fake_distributed = types.ModuleType("vllm.distributed")
    fake_distributed.get_world_group = lambda: SimpleNamespace(device_group="world-group")
    monkeypatch.setitem(sys.modules, "vllm.distributed", fake_distributed)
    monkeypatch.setattr(memcache_module, "get_ascend_device_type", lambda: memcache_module.AscendDeviceType.A2)
    monkeypatch.setattr(memcache_module.torch, "zeros", lambda *_args, **_kwargs: "tmp-tensor")
    monkeypatch.setattr(memcache_module.torch, "empty_like", lambda tensor: f"empty:{tensor}")
    monkeypatch.setattr(memcache_module.torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        memcache_module.torch.distributed,
        "all_gather",
        lambda output_tensor_list, tmp_tensor, group=None: gathered.append((list(output_tensor_list), tmp_tensor, group)),
    )

    backend = memcache_module.MemcacheBackend(SimpleNamespace(rank=1))

    assert backend.rank == 1
    assert fake_memcache_module.init_calls == [1]
    assert gathered == [(["empty:tmp-tensor", "empty:tmp-tensor"], "tmp-tensor", "world-group")]


def test_memcache_backend_init_logs_value_error(fake_memcache_module, monkeypatch):
    logged = []
    monkeypatch.setattr(memcache_module, "get_ascend_device_type", lambda: (_ for _ in ()).throw(ValueError("bad cfg")))
    monkeypatch.setattr(memcache_module.logger, "error", lambda *args: logged.append(args))

    with pytest.raises(ValueError, match="bad cfg"):
        memcache_module.MemcacheBackend(SimpleNamespace(rank=0))

    assert logged and "Configuration loading failed" in logged[0][0]


def test_memcache_backend_init_logs_generic_exception(fake_memcache_module, monkeypatch):
    logged = []
    monkeypatch.setattr(memcache_module, "get_ascend_device_type", lambda: memcache_module.AscendDeviceType.A3)
    fake_memcache_module.init = lambda rank: (_ for _ in ()).throw(RuntimeError("init failed"))
    monkeypatch.setattr(memcache_module.logger, "error", lambda *args: logged.append(args))

    with pytest.raises(RuntimeError, match="init failed"):
        memcache_module.MemcacheBackend(SimpleNamespace(rank=0))

    assert logged and "An error occurred while loading the configuration" in logged[0][0]


def test_memcache_backend_set_device_and_register_buffer(fake_memcache_module, monkeypatch):
    set_device_calls = []
    monkeypatch.setattr(memcache_module, "get_ascend_device_type", lambda: memcache_module.AscendDeviceType.A2)
    monkeypatch.setattr(memcache_module.torch, "zeros", lambda *_args, **_kwargs: "tmp")
    monkeypatch.setattr(memcache_module.torch, "empty_like", lambda tensor: tensor)
    monkeypatch.setattr(memcache_module.torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(memcache_module.torch.distributed, "all_gather", lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "vllm.distributed", types.SimpleNamespace(get_world_group=lambda: SimpleNamespace(device_group="wg")))
    monkeypatch.setattr(memcache_module.torch.npu, "set_device", lambda device: set_device_calls.append(device))

    backend = memcache_module.MemcacheBackend(SimpleNamespace(rank=2))
    backend.set_device()
    assert str(set_device_calls[0]) == "npu:2"

    backend.register_buffer([10, 20], [30, 40])
    assert fake_memcache_module.register_calls == [(10, 30), (20, 40)]

    fake_memcache_module.register_calls.clear()
    monkeypatch.setattr(memcache_module, "get_ascend_device_type", lambda: memcache_module.AscendDeviceType.A3)
    backend.register_buffer([1], [2])
    assert fake_memcache_module.register_calls == []


def test_memcache_backend_exists_get_and_put(fake_memcache_module, monkeypatch):
    backend = object.__new__(memcache_module.MemcacheBackend)
    backend.store = fake_memcache_module
    logged = []
    monkeypatch.setattr(memcache_module.logger, "error", lambda *args: logged.append(args))

    assert backend.exists(["a", "b"]) == [1, 0]

    backend.get(["k"], [[1]], [[2]])
    backend.put(["k"], [[1]], [[2]])
    assert logged == []

    fake_memcache_module.get_result = [1]
    fake_memcache_module.put_result = [1]
    backend.get(["k"], [[1]], [[2]])
    backend.put(["k"], [[1]], [[2]])
    assert any("Failed to get key" in args[0] for args in logged)

    fake_memcache_module.batch_get_into_layers = lambda *args: (_ for _ in ()).throw(RuntimeError("get boom"))
    fake_memcache_module.batch_put_from_layers = lambda *args: (_ for _ in ()).throw(RuntimeError("put boom"))
    backend.get(["k2"], [[3]], [[4]])
    backend.put(["k2"], [[3]], [[4]])
    assert any("Failed to get key" in args[0] for args in logged)
    assert any("Failed to put key" in args[0] for args in logged)
