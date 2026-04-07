from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.utils import utils as kv_utils


def test_kv_alltoall_and_rearrange_handles_ratio_and_invalid_inputs(monkeypatch):
    key = torch.ones(2, 1, 1)
    value = torch.zeros(2, 1, 1)

    assert kv_utils.kv_alltoall_and_rearrange(1, key, value) == (None, None)

    with pytest.raises(ValueError, match="key or value is None"):
        kv_utils.kv_alltoall_and_rearrange(2, None, value)

    calls = []

    def fake_alltoall_and_rearrange(ratio, tensor):
        calls.append((ratio, tensor))
        return tensor + 1

    monkeypatch.setattr(kv_utils, "alltoall_and_rearrange", fake_alltoall_and_rearrange)

    result_key, result_value = kv_utils.kv_alltoall_and_rearrange(2, key, value)

    assert calls == [(2, key), (2, value)]
    assert torch.equal(result_key, key + 1)
    assert torch.equal(result_value, value + 1)


def test_alltoall_and_rearrange_invokes_collective(monkeypatch):
    input_tensor = torch.arange(24).view(4, 2, 3)
    captured = {}

    def fake_all_to_all_single(output, input_, group):
        captured["group"] = group
        output.copy_(input_)

    monkeypatch.setattr(kv_utils.dist, "all_to_all_single", fake_all_to_all_single)
    monkeypatch.setattr(kv_utils, "get_p_tp_group", lambda: SimpleNamespace(device_group="ptp-group"))

    result = kv_utils.alltoall_and_rearrange(2, input_tensor)

    assert captured["group"] == "ptp-group"
    assert torch.equal(result, kv_utils.rearrange_output(input_tensor, 2, 2))


def test_rearrange_output_reorders_tensor_and_validates_shape():
    base_output = torch.arange(24).view(4, 2, 3)

    result = kv_utils.rearrange_output(base_output, 2, 2)

    assert torch.equal(result, torch.tensor(
        [[[0, 1, 2], [3, 4, 5]], [[12, 13, 14], [15, 16, 17]], [[6, 7, 8], [9, 10, 11]], [[18, 19, 20], [21, 22, 23]]]
    ))

    with pytest.raises(ValueError, match="must be divisible"):
        kv_utils.rearrange_output(torch.arange(18).view(3, 2, 3), 2, 2)


class FakeTensor:

    def __init__(self, ptr, element_size, values):
        self._ptr = ptr
        self._element_size = element_size
        self._values = values

    def data_ptr(self):
        return self._ptr

    def element_size(self):
        return self._element_size

    def __getitem__(self, item):
        return self._values[item]


def test_align_memory_returns_aligned_slice():
    fake_tensor = FakeTensor(ptr=12, element_size=4, values=[10, 20, 30, 40])

    result = kv_utils.align_memory(fake_tensor, alignment=16)

    assert result == [20, 30, 40]


def test_get_transfer_timeout_value_prefers_override(monkeypatch):
    monkeypatch.setenv("ASCEND_TRANSFER_TIMEOUT", "88")

    assert kv_utils.get_transfer_timeout_value() == 88


def test_get_transfer_timeout_value_uses_hccl_defaults(monkeypatch):
    monkeypatch.delenv("ASCEND_TRANSFER_TIMEOUT", raising=False)
    monkeypatch.setenv("HCCL_RDMA_TIMEOUT", "4")
    monkeypatch.setenv("HCCL_RDMA_RETRY_CNT", "3")

    assert kv_utils.get_transfer_timeout_value() == 3000


def test_get_cp_group_covers_small_and_regular_cases():
    assert kv_utils.get_cp_group(2, 4, 1) == [[0, 1]]
    assert kv_utils.get_cp_group(8, 4, 1) == [{0, 2, 4, 6}, {1, 3, 5, 7}]


def test_context_parallel_parameters_check_covers_success_and_failures():
    producer = kv_utils.parallel_info(tp_size=4, pcp_size=2, dcp_size=2, use_mla=False, pd_head_ratio=1)
    decoder = kv_utils.parallel_info(tp_size=4, pcp_size=1, dcp_size=2, use_mla=False, pd_head_ratio=1)

    kv_utils.context_parallel_parameters_check(1, 2, producer, decoder, total_num_kv_heads=4)

    with pytest.raises(AssertionError):
        kv_utils.context_parallel_parameters_check(3, 1, producer, decoder, total_num_kv_heads=4)

    with pytest.raises(AssertionError):
        bad_decoder = kv_utils.parallel_info(tp_size=4, pcp_size=1, dcp_size=3, use_mla=False, pd_head_ratio=1)
        kv_utils.context_parallel_parameters_check(1, 1, producer, bad_decoder, total_num_kv_heads=7)

    mla_producer = kv_utils.parallel_info(tp_size=4, pcp_size=2, dcp_size=2, use_mla=True, pd_head_ratio=1)
    kv_utils.context_parallel_parameters_check(1, 2, mla_producer, decoder, total_num_kv_heads=4)


def test_get_tp_rank_head_mapping_covers_both_layout_modes():
    assert kv_utils.get_tp_rank_head_mapping(4, 2) == {0: [0, 1], 1: [2, 3]}
    assert kv_utils.get_tp_rank_head_mapping(2, 4) == {0: [0], 1: [0], 2: [1], 3: [1]}

    with pytest.raises(ValueError, match="cannot be evenly divided"):
        kv_utils.get_tp_rank_head_mapping(3, 2)

    with pytest.raises(ValueError, match="cannot be evenly divided"):
        kv_utils.get_tp_rank_head_mapping(2, 3)


def test_get_head_group_mapping_filters_selected_groups():
    mapping = kv_utils.get_head_group_mapping(4, 4, 2, [1])

    assert mapping == {1: [2, 3]}

    with pytest.raises(ValueError, match="cannot be divided"):
        kv_utils.get_head_group_mapping(4, 4, 3, [0])


def test_get_local_remote_block_port_mappings_builds_expected_structures(monkeypatch):
    p_info = kv_utils.parallel_info(tp_size=2, pcp_size=1, dcp_size=1, use_mla=False, pd_head_ratio=2)
    d_info = kv_utils.parallel_info(tp_size=2, pcp_size=1, dcp_size=1, use_mla=False, pd_head_ratio=1)
    req_meta = SimpleNamespace(remote_cache_tokens=0)
    info_messages = []

    monkeypatch.setattr(kv_utils.logger, "info", info_messages.append)

    p_rank_block_mapping, d_block_rank_mapping, pd_head_mapping, d_trans_count_mapping = (
        kv_utils.get_local_remote_block_port_mappings(
            to_trans_idx=2,
            p_parallel_info=p_info,
            d_parallel_info=d_info,
            d_hosts=["host-a", "host-b"],
            d_port=7000,
            selected_p_cp_group=[0, 1],
            selected_d_cp_group=[0],
            prompt_len=2,
            block_size=1,
            req_meta=req_meta,
            total_num_kv_heads=2,
            req_id="req-1",
        )
    )

    assert p_rank_block_mapping == [[[ [0, 1] ], [ [0, 1] ]]]
    assert d_block_rank_mapping == {
        0: {0: {"pcp_rank": 0, "dcp_rank": 0, "host": "host-a", "port": 7000, "block_idx": 0}},
        1: {0: {"pcp_rank": 0, "dcp_rank": 0, "host": "host-a", "port": 7000, "block_idx": 1}},
    }
    assert pd_head_mapping == {0: [0], 1: []}
    assert d_trans_count_mapping == {("host-a", 7000): 2, ("host-b", 7001): 2}
    assert any("Head 1 exists in P but not in D mapping" in msg for msg in info_messages)


def test_get_local_remote_block_port_mappings_handles_partial_transfer_counts(monkeypatch):
    monkeypatch.setattr(kv_utils.logger, "info", lambda *_: None)

    _, _, pd_head_mapping, d_trans_count_mapping = kv_utils.get_local_remote_block_port_mappings(
        to_trans_idx=1,
        p_parallel_info=kv_utils.parallel_info(tp_size=2, pcp_size=2, dcp_size=2, use_mla=False, pd_head_ratio=1),
        d_parallel_info=kv_utils.parallel_info(tp_size=2, pcp_size=2, dcp_size=2, use_mla=False, pd_head_ratio=1),
        d_hosts=["host-a", "host-b"],
        d_port=8000,
        selected_p_cp_group=[0],
        selected_d_cp_group=[0],
        prompt_len=1,
        block_size=1,
        req_meta=SimpleNamespace(remote_cache_tokens=1),
        total_num_kv_heads=2,
        req_id="req-3",
    )

    assert pd_head_mapping == {0: [0]}
    assert d_trans_count_mapping == {
        ("host-a", 8000): 0,
        ("host-a", 8001): 1,
        ("host-b", 8002): 0,
        ("host-b", 8003): 0,
    }


def test_get_local_remote_block_port_mappings_skips_unselected_p_groups(monkeypatch):
    monkeypatch.setattr(kv_utils.logger, "info", lambda *_: None)

    p_rank_block_mapping, _, _, _ = kv_utils.get_local_remote_block_port_mappings(
        to_trans_idx=2,
        p_parallel_info=kv_utils.parallel_info(tp_size=2, pcp_size=1, dcp_size=1, use_mla=False, pd_head_ratio=1),
        d_parallel_info=kv_utils.parallel_info(tp_size=2, pcp_size=1, dcp_size=1, use_mla=False, pd_head_ratio=1),
        d_hosts=["host-a", "host-b"],
        d_port=8100,
        selected_p_cp_group=[0],
        selected_d_cp_group=[0],
        prompt_len=2,
        block_size=1,
        req_meta=SimpleNamespace(remote_cache_tokens=0),
        total_num_kv_heads=2,
        req_id="req-4",
    )

    assert p_rank_block_mapping == [[[[0, 1]], [[]]]]


def test_get_transfer_mappings_skips_out_of_range_blocks(monkeypatch):
    debug_messages = []
    monkeypatch.setattr(kv_utils.logger, "debug", debug_messages.append)

    transfer_mappings = kv_utils.get_transfer_mappings(
        p_rank_block_mapping=[[[[0, 1, 2]]]],
        d_block_rank_mapping={
            1: {0: {"host": "host-a", "port": 7000, "block_idx": 0}},
            2: {0: {"host": "host-a", "port": 7000, "block_idx": 1}},
        },
        pd_head_mapping={0: {0}},
        d_trans_count_mapping={("host-a", 7000): 4},
        req_meta=SimpleNamespace(local_block_ids=[[10, 11, 12]], remote_block_ids=[[20, 21]]),
        block_group_idx=0,
        p_parallel_info=kv_utils.parallel_info(tp_size=1, pcp_size=1, dcp_size=1, use_mla=False, pd_head_ratio=1),
        req_id="req-2",
        transed_idx=1,
        to_trans_idx=3,
        tp_rank=0,
        pcp_rank=0,
        dcp_rank=0,
    )

    assert transfer_mappings == {
        ("host-a", 7000): {
            "local_block_ids": [11, 12],
            "remote_block_ids": [20, 21],
            "trans_count": 4,
        }
    }
    assert any("req-2" in msg for msg in debug_messages)