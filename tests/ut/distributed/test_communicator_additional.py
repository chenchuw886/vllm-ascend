from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist

from vllm_ascend.distributed.device_communicators.npu_communicator import NPUCommunicator


@patch("vllm.config.get_current_vllm_config", return_value=None)
@patch("torch.npu.current_device", return_value=MagicMock())
@patch("torch.npu.set_device", return_value=MagicMock())
@patch("torch.distributed.get_process_group_ranks", return_value={0: 0, 1: 1})
@patch("torch.distributed.get_group_rank", return_value={0: 0, 1: 1})
@patch("torch.distributed.is_initialized", return_value=True)
@patch("torch.distributed.get_rank", return_value=1)
@patch("torch.distributed.is_initialized", return_value=True)
@patch("torch.distributed.get_backend", return_value="hccl")
@patch("torch.distributed.get_rank", return_value=1)
@patch("torch.distributed.get_world_size", return_value=2)
@patch("torch.distributed.get_process_group_ranks", return_value=[0, 1])
@patch("torch.npu.device")
def test_all_to_all_normalizes_negative_scatter_dim(*_):

    def patched_all_to_all(output_tensor_list, input_tensor_list, group=None, async_op=False):
        output_tensor_list[:] = [
            torch.tensor([[1, 10], [2, 20]]),
            torch.tensor([[3, 30], [4, 40]]),
        ]

    torch.distributed.all_to_all = patched_all_to_all
    input_tensor = torch.tensor([[1, 2, 3, 4], [10, 20, 30, 40]])

    with patch.dict(dist.distributed_c10d._world.pg_map, {dist.group.WORLD: MagicMock()}, clear=False):
        comm = NPUCommunicator(cpu_group=dist.group.WORLD)
        output = comm.all_to_all(input_tensor, scatter_dim=-1, gather_dim=-1)

    assert output.tolist() == [[1, 10, 3, 30], [2, 20, 4, 40]]