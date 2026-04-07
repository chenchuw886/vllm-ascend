from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch.distributed import ReduceOp
from vllm.distributed.utils import StatelessProcessGroup

from vllm_ascend.distributed.device_communicators import pyhccl as pyhccl_module


class FakeUniqueId:

    def __init__(self, values=None):
        self.internal = bytearray(values or [0, 0, 0, 0])


class FakeHcclLib:

    def __init__(self, *_args, **_kwargs):
        self.init_rank_calls = []
        self.all_reduce_calls = []
        self.broadcast_calls = []

    def hcclGetUniqueId(self):
        return FakeUniqueId([1, 2, 3, 4])

    def hcclCommInitRank(self, world_size, unique_id, rank):
        self.init_rank_calls.append((world_size, list(unique_id.internal[:4]), rank))
        return "fake-comm"

    def hcclAllReduce(self, *args):
        self.all_reduce_calls.append(args)

    def hcclBroadcast(self, *args):
        self.broadcast_calls.append(args)


def test_constructor_initializes_non_stateless_group(monkeypatch):
    fake_lib = FakeHcclLib()
    stream = SimpleNamespace(npu_stream=123, synchronize=lambda: None)
    broadcast_calls = []

    def fake_broadcast(tensor, src, group):
        broadcast_calls.append((src, group))
        tensor.copy_(torch.tensor([9, 8, 7, 6], dtype=torch.uint8))

    monkeypatch.setattr(pyhccl_module, "HCCLLibrary", lambda *_: fake_lib)
    monkeypatch.setattr(pyhccl_module.logger, "info", lambda *_: None)
    monkeypatch.setattr(pyhccl_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(pyhccl_module.dist, "get_backend", lambda _group: "nccl")
    monkeypatch.setattr(pyhccl_module.dist, "get_rank", lambda _group: 0)
    monkeypatch.setattr(pyhccl_module.dist, "get_world_size", lambda _group: 2)
    monkeypatch.setattr(pyhccl_module.dist, "get_process_group_ranks", lambda _group: [5, 6])
    monkeypatch.setattr(pyhccl_module.dist, "broadcast", fake_broadcast)
    monkeypatch.setattr(pyhccl_module.torch.npu, "device", lambda _device: nullcontext())
    monkeypatch.setattr(pyhccl_module, "current_stream", lambda: stream)

    group = object()
    communicator = pyhccl_module.PyHcclCommunicator(group=group, device="cpu")

    assert communicator.available is True
    assert communicator.disabled is False
    assert communicator.device == torch.device("cpu")
    assert list(communicator.unique_id.internal[:4]) == [9, 8, 7, 6]
    assert fake_lib.init_rank_calls == [(2, [9, 8, 7, 6], 0)]
    assert len(fake_lib.all_reduce_calls) == 1
    assert broadcast_calls == [(5, group)]


def test_constructor_initializes_stateless_group_and_int_device(monkeypatch):
    fake_lib = FakeHcclLib()
    stream = SimpleNamespace(npu_stream=456, synchronize=lambda: None)
    original_all_reduce = pyhccl_module.PyHcclCommunicator.all_reduce
    original_zeros = pyhccl_module.torch.zeros
    group = StatelessProcessGroup(rank=1, world_size=2, store=None, socket=None)
    group.broadcast_obj = lambda unique_id, src: FakeUniqueId([7, 7, 7, 7])

    monkeypatch.setattr(pyhccl_module, "HCCLLibrary", lambda *_: fake_lib)
    monkeypatch.setattr(pyhccl_module, "hcclUniqueId", lambda: FakeUniqueId([0, 0, 0, 0]))
    monkeypatch.setattr(pyhccl_module.logger, "info", lambda *_: None)
    monkeypatch.setattr(pyhccl_module.torch.npu, "device", lambda _device: nullcontext())
    monkeypatch.setattr(pyhccl_module, "current_stream", lambda: stream)
    monkeypatch.setattr(pyhccl_module.torch, "zeros", lambda *_args, **_kwargs: original_zeros(1))
    monkeypatch.setattr(pyhccl_module.PyHcclCommunicator, "all_reduce", lambda self, tensor, op=ReduceOp.SUM, stream=None: tensor)

    communicator = pyhccl_module.PyHcclCommunicator(group=group, device=0)

    assert communicator.rank == 1
    assert communicator.world_size == 2
    assert communicator.available is True
    assert communicator.disabled is False
    assert communicator.device == torch.device("npu:0")
    assert list(communicator.unique_id.internal[:4]) == [7, 7, 7, 7]
    assert fake_lib.init_rank_calls == [(2, [7, 7, 7, 7], 1)]

    monkeypatch.setattr(pyhccl_module.PyHcclCommunicator, "all_reduce", original_all_reduce)


def test_constructor_accepts_torch_device_instance(monkeypatch):
    fake_lib = FakeHcclLib()
    stream = SimpleNamespace(npu_stream=654, synchronize=lambda: None)
    group = StatelessProcessGroup(rank=0, world_size=2, store=None, socket=None)
    group.broadcast_obj = lambda unique_id, src: unique_id

    monkeypatch.setattr(pyhccl_module, "HCCLLibrary", lambda *_: fake_lib)
    monkeypatch.setattr(pyhccl_module.logger, "info", lambda *_: None)
    monkeypatch.setattr(pyhccl_module.torch.npu, "device", lambda _device: nullcontext())
    monkeypatch.setattr(pyhccl_module, "current_stream", lambda: stream)

    communicator = pyhccl_module.PyHcclCommunicator(group=group, device=torch.device("cpu"))

    assert communicator.device == torch.device("cpu")
    assert communicator.available is True
    assert communicator.disabled is False


def test_all_reduce_returns_none_when_disabled():
    communicator = object.__new__(pyhccl_module.PyHcclCommunicator)
    communicator.disabled = True

    assert communicator.all_reduce(torch.tensor([1.0])) is None


def test_all_reduce_checks_device_and_uses_current_stream(monkeypatch):
    fake_lib = FakeHcclLib()
    stream = SimpleNamespace(npu_stream=321)
    communicator = object.__new__(pyhccl_module.PyHcclCommunicator)
    communicator.disabled = False
    communicator.device = torch.device("cpu")
    communicator.hccl = fake_lib
    communicator.comm = "fake-comm"
    monkeypatch.setattr(pyhccl_module, "current_stream", lambda: stream)

    tensor = torch.tensor([1.0], dtype=torch.float32)
    output = communicator.all_reduce(tensor)

    assert output.shape == tensor.shape
    assert len(fake_lib.all_reduce_calls) == 1

    with pytest.raises(AssertionError, match="created to work on"):
        communicator.device = torch.device("meta")
        communicator.all_reduce(tensor)


def test_all_reduce_uses_explicit_stream(monkeypatch):
    fake_lib = FakeHcclLib()
    communicator = object.__new__(pyhccl_module.PyHcclCommunicator)
    communicator.disabled = False
    communicator.device = torch.device("cpu")
    communicator.hccl = fake_lib
    communicator.comm = "fake-comm"

    tensor = torch.tensor([1], dtype=torch.int64)
    stream = SimpleNamespace(npu_stream=999)
    communicator.all_reduce(tensor, op=ReduceOp.MAX, stream=stream)

    assert len(fake_lib.all_reduce_calls) == 1


def test_broadcast_covers_disabled_src_variants_and_device_check(monkeypatch):
    fake_lib = FakeHcclLib()
    communicator = object.__new__(pyhccl_module.PyHcclCommunicator)
    communicator.hccl = fake_lib
    communicator.comm = "fake-comm"
    communicator.rank = 0
    communicator.device = torch.device("cpu")

    communicator.disabled = True
    assert communicator.broadcast(torch.tensor([1.0]), src=0) is None

    communicator.disabled = False
    tensor = torch.tensor([1.0], dtype=torch.float32)
    monkeypatch.setattr(pyhccl_module, "current_stream", lambda: SimpleNamespace(npu_stream=11))
    communicator.broadcast(tensor, src=0)
    communicator.broadcast(tensor, src=1, stream=SimpleNamespace(npu_stream=22))
    assert len(fake_lib.broadcast_calls) == 2

    communicator.device = torch.device("meta")
    with pytest.raises(AssertionError, match="created to work on"):
        communicator.broadcast(tensor, src=0)