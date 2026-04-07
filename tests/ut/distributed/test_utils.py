from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.distributed import utils as distributed_utils


def test_fc3_all_gather_returns_input_when_forward_context_missing(monkeypatch):
    input_tensor = torch.tensor([[1.0, 2.0]])

    def raise_assertion():
        raise AssertionError("forward context not initialized")

    monkeypatch.setattr(distributed_utils, "get_forward_context", raise_assertion)

    result = distributed_utils.fc3_all_gather_and_maybe_unpad_impl(input_tensor)

    assert result is input_tensor


def test_fc3_all_gather_unpads_by_extra_context(monkeypatch):
    input_tensor = torch.tensor([[1.0], [2.0]])
    gathered_tensor = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    mock_group = SimpleNamespace(all_gather=MagicMock(return_value=gathered_tensor))

    monkeypatch.setattr(distributed_utils, "get_forward_context", lambda: SimpleNamespace(dp_metadata=None))
    monkeypatch.setattr(distributed_utils, "get_fc3_quant_x_group", lambda: mock_group)
    monkeypatch.setattr(distributed_utils, "_EXTRA_CTX", SimpleNamespace(pad_size=2))

    result = distributed_utils.fc3_all_gather_and_maybe_unpad_impl(input_tensor)

    assert torch.equal(result, torch.tensor([[1.0], [2.0]]))
    mock_group.all_gather.assert_called_once_with(input_tensor, 0)


def test_fc3_all_gather_keeps_tensor_when_pad_size_is_zero(monkeypatch):
    input_tensor = torch.tensor([[1.0], [2.0]])
    gathered_tensor = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    mock_group = SimpleNamespace(all_gather=MagicMock(return_value=gathered_tensor))

    monkeypatch.setattr(distributed_utils, "get_forward_context", lambda: SimpleNamespace(dp_metadata=None))
    monkeypatch.setattr(distributed_utils, "get_fc3_quant_x_group", lambda: mock_group)
    monkeypatch.setattr(distributed_utils, "_EXTRA_CTX", SimpleNamespace(pad_size=0))

    result = distributed_utils.fc3_all_gather_and_maybe_unpad_impl(input_tensor)

    assert torch.equal(result, gathered_tensor)


def test_fc3_all_gather_rebuilds_tensor_from_dp_metadata(monkeypatch):
    input_tensor = torch.tensor([[99.0]])
    gathered_tensor = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    mock_group = SimpleNamespace(all_gather=MagicMock(return_value=gathered_tensor))
    dp_metadata = SimpleNamespace(num_tokens_across_dp_cpu=torch.tensor([1, 2]))

    monkeypatch.setattr(distributed_utils, "get_forward_context", lambda: SimpleNamespace(dp_metadata=dp_metadata))
    monkeypatch.setattr(distributed_utils, "get_fc3_quant_x_group", lambda: mock_group)
    monkeypatch.setattr(distributed_utils, "get_dp_group", lambda: SimpleNamespace(world_size=2))
    monkeypatch.setattr(distributed_utils, "_EXTRA_CTX", SimpleNamespace(padded_length=3))

    result = distributed_utils.fc3_all_gather_and_maybe_unpad_impl(input_tensor)

    assert torch.equal(result, torch.tensor([[1.0], [4.0], [5.0]]))


def test_all_gather_async_returns_input_for_single_rank():
    group = SimpleNamespace(world_size=1)
    input_tensor = torch.tensor([1, 2, 3])

    output, handle = distributed_utils.all_gather_async(input_tensor, group)

    assert output is input_tensor
    assert handle is None


def test_all_gather_async_allocates_output_when_missing(monkeypatch):
    input_tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    output_holder = {}

    def fake_all_gather_into_tensor(output, input_, group, async_op):
        output.copy_(torch.cat([input_, input_], dim=0))
        output_holder["output"] = output.clone()
        output_holder["group"] = group
        output_holder["async_op"] = async_op
        return "work-handle"

    monkeypatch.setattr(distributed_utils.dist, "all_gather_into_tensor", fake_all_gather_into_tensor)
    group = SimpleNamespace(world_size=2, device_group="device-group")

    output, handle = distributed_utils.all_gather_async(input_tensor, group)

    assert handle == "work-handle"
    assert output.shape == (4, 2)
    assert torch.equal(output_holder["output"], torch.tensor([[1, 2], [3, 4], [1, 2], [3, 4]], dtype=torch.float32))
    assert output_holder["group"] == "device-group"
    assert output_holder["async_op"] is True


def test_all_gather_async_uses_provided_output(monkeypatch):
    input_tensor = torch.tensor([1, 2], dtype=torch.int64)
    provided_output = torch.empty(4, dtype=torch.int64)

    def fake_all_gather_into_tensor(output, input_, group, async_op):
        output.copy_(torch.tensor([1, 2, 1, 2]))
        return None

    monkeypatch.setattr(distributed_utils.dist, "all_gather_into_tensor", fake_all_gather_into_tensor)
    group = SimpleNamespace(world_size=2, device_group="provided-group")

    output, handle = distributed_utils.all_gather_async(input_tensor, group, output=provided_output, async_op=False)

    assert output is provided_output
    assert handle is None
    assert torch.equal(provided_output, torch.tensor([1, 2, 1, 2]))