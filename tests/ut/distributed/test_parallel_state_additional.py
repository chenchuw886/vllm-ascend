from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from vllm.config import ParallelConfig

from vllm_ascend.distributed import parallel_state as ps


PARALLEL_GLOBALS = [
    "_MC2",
    "_MLP_TP",
    "_OTP",
    "_LMTP",
    "_EMBED_TP",
    "_FLASHCOMM2_OTP",
    "_FLASHCOMM2_ODP",
    "_FC3_QUANT_X",
    "_SHARD_WEIGHT",
    "_P_TP",
    "_DYNAMIC_EPLB",
]


@pytest.fixture(autouse=True)
def reset_parallel_groups(monkeypatch):
    for attr in PARALLEL_GLOBALS:
        monkeypatch.setattr(ps, attr, None)


def make_parallel_config():
    return ParallelConfig(data_parallel_size=2, tensor_parallel_size=4, pipeline_parallel_size=2)


def make_ascend_config(
    *,
    pd_tp_ratio=2,
    pd_head_ratio=2,
    num_head_replica=0,
    dynamic_eplb=False,
    flashcomm2_oproj_tensor_parallel_size=2,
    layer_sharding=None,
    multistream_overlap_gate=False,
    otp_size=2,
    lmhead_tp_size=2,
    embedding_tp_size=2,
    mlp_tp_size=2,
):
    return SimpleNamespace(
        pd_tp_ratio=pd_tp_ratio,
        pd_head_ratio=pd_head_ratio,
        num_head_replica=num_head_replica,
        flashcomm2_oproj_tensor_parallel_size=flashcomm2_oproj_tensor_parallel_size,
        layer_sharding=layer_sharding,
        multistream_overlap_gate=multistream_overlap_gate,
        eplb_config=SimpleNamespace(dynamic_eplb=dynamic_eplb),
        finegrained_tp_config=SimpleNamespace(
            oproj_tensor_parallel_size=otp_size,
            lmhead_tensor_parallel_size=lmhead_tp_size,
            embedding_tensor_parallel_size=embedding_tp_size,
            mlp_tensor_parallel_size=mlp_tp_size,
        ),
    )


def patch_common_init(monkeypatch, ascend_config, *, is_kv_producer=True, flashcomm2=False, enable_dsa=False):
    world_group = SimpleNamespace(local_rank=0, device_group="world-device")
    tp_group = SimpleNamespace(name="tp-group")
    init_calls = []

    def fake_init_model_parallel_group(group_ranks, local_rank, backend, group_name):
        group = SimpleNamespace(
            group_name=group_name,
            group_ranks=group_ranks,
            local_rank=local_rank,
            backend=backend,
            destroy=MagicMock(),
        )
        init_calls.append(group)
        return group

    monkeypatch.setattr(ps.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(ps.torch.distributed, "get_world_size", lambda: 16)
    monkeypatch.setattr(ps.torch.distributed, "get_backend", lambda _group: "hccl")
    monkeypatch.setattr(ps, "get_world_group", lambda: world_group)
    monkeypatch.setattr(ps, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(ps, "get_ascend_config", lambda: ascend_config)
    monkeypatch.setattr(
        ps,
        "get_current_vllm_config",
        lambda: SimpleNamespace(kv_transfer_config=SimpleNamespace(is_kv_producer=is_kv_producer)),
    )
    monkeypatch.setattr(ps, "flashcomm2_enable", lambda: flashcomm2)
    monkeypatch.setattr(ps, "enable_dsa_cp_with_layer_shard", lambda: enable_dsa)
    monkeypatch.setattr(ps, "init_model_parallel_group", fake_init_model_parallel_group)
    return world_group, tp_group, init_calls


def test_init_ascend_model_parallel_returns_early_when_already_initialized(monkeypatch):
    monkeypatch.setattr(ps, "_MC2", object())
    monkeypatch.setattr(ps.torch.distributed, "is_initialized", lambda: (_ for _ in ()).throw(RuntimeError("no call")))

    ps.init_ascend_model_parallel(make_parallel_config())


@pytest.mark.parametrize(
    ("setter", "getter", "message"),
    [
        ("_MC2", ps.get_mc2_group, "mc2 group is not initialized"),
        ("_MLP_TP", ps.get_mlp_tp_group, "mlp group is not initialized"),
        ("_OTP", ps.get_otp_group, "output tensor parallel group is not initialized"),
        ("_LMTP", ps.get_lmhead_tp_group, "lm head tensor parallel group is not initialized"),
        ("_EMBED_TP", ps.get_embed_tp_group, "emtp group is not initialized"),
        ("_FLASHCOMM2_ODP", ps.get_flashcomm2_odp_group, "output data parallel group for flashcomm2 is not initialized"),
        ("_SHARD_WEIGHT", ps.get_shard_weight_group, "output shard weight parallel group for flashcomm2 is not initialized"),
        ("_P_TP", ps.get_p_tp_group, "distributed prefill tensor parallel group is not initialized"),
        ("_FC3_QUANT_X", ps.get_fc3_quant_x_group, "fc3 quant x group is not initialized"),
        ("_DYNAMIC_EPLB", ps.get_dynamic_eplb_group, "fc3 quant x group is not initialized"),
    ],
)
def test_group_getters_validate_initialization(monkeypatch, setter, getter, message):
    with pytest.raises(AssertionError, match=message):
        getter()

    sentinel = object()
    monkeypatch.setattr(ps, setter, sentinel)

    assert getter() is sentinel


def test_get_flashcomm2_otp_group_returns_value_without_assert(monkeypatch):
    assert ps.get_flashcomm2_otp_group() is None

    sentinel = object()
    monkeypatch.setattr(ps, "_FLASHCOMM2_OTP", sentinel)

    assert ps.get_flashcomm2_otp_group() is sentinel


def test_destroy_ascend_model_parallel_destroys_all_groups(monkeypatch):
    flashcomm2_otp = MagicMock()
    flashcomm2_odp = MagicMock()
    monkeypatch.setattr(ps, "_MC2", MagicMock())
    monkeypatch.setattr(ps, "_MLP_TP", MagicMock())
    monkeypatch.setattr(ps, "_OTP", MagicMock())
    monkeypatch.setattr(ps, "_LMTP", MagicMock())
    monkeypatch.setattr(ps, "_EMBED_TP", MagicMock())
    monkeypatch.setattr(ps, "_P_TP", MagicMock())
    monkeypatch.setattr(ps, "_SHARD_WEIGHT", MagicMock())
    monkeypatch.setattr(ps, "_FC3_QUANT_X", MagicMock())
    monkeypatch.setattr(ps, "_DYNAMIC_EPLB", MagicMock())
    monkeypatch.setattr(ps, "_FLASHCOMM2_OTP", flashcomm2_otp)
    monkeypatch.setattr(ps, "_FLASHCOMM2_ODP", flashcomm2_odp)
    monkeypatch.setattr(ps, "get_ascend_config", lambda: SimpleNamespace(flashcomm2_oproj_tensor_parallel_size=2))

    ps.destroy_ascend_model_parallel()

    for attr in ["_MC2", "_MLP_TP", "_OTP", "_LMTP", "_EMBED_TP", "_P_TP", "_SHARD_WEIGHT", "_FC3_QUANT_X", "_DYNAMIC_EPLB"]:
        assert getattr(ps, attr) is None
    flashcomm2_otp.destroy.assert_called_once()
    flashcomm2_odp.destroy.assert_called_once()
    assert ps._FLASHCOMM2_OTP is None
    assert ps._FLASHCOMM2_ODP is None


def test_destroy_ascend_model_parallel_keeps_flashcomm_groups_when_size_is_one(monkeypatch):
    flashcomm2_otp = MagicMock()
    flashcomm2_odp = MagicMock()
    monkeypatch.setattr(ps, "_FLASHCOMM2_OTP", flashcomm2_otp)
    monkeypatch.setattr(ps, "_FLASHCOMM2_ODP", flashcomm2_odp)
    monkeypatch.setattr(ps, "get_ascend_config", lambda: SimpleNamespace(flashcomm2_oproj_tensor_parallel_size=1))

    ps.destroy_ascend_model_parallel()

    flashcomm2_otp.destroy.assert_not_called()
    flashcomm2_odp.destroy.assert_not_called()
    assert ps._FLASHCOMM2_OTP is flashcomm2_otp
    assert ps._FLASHCOMM2_ODP is flashcomm2_odp


def test_init_parallel_state_creates_dynamic_eplb_and_fc3_but_not_p_tp(monkeypatch):
    ascend_config = make_ascend_config(
        pd_head_ratio=1,
        dynamic_eplb=True,
        multistream_overlap_gate=True,
        otp_size=0,
        lmhead_tp_size=0,
        embedding_tp_size=0,
        mlp_tp_size=0,
    )
    _, _, init_calls = patch_common_init(monkeypatch, ascend_config, flashcomm2=False)

    ps.init_ascend_model_parallel(make_parallel_config())

    assert ps.get_mc2_group().group_name == "mc2"
    assert ps.get_dynamic_eplb_group().group_name == "dynamic_eplb"
    assert ps.get_fc3_quant_x_group().group_name == "fc3_quant_x"
    with pytest.raises(AssertionError, match="distributed prefill tensor parallel group is not initialized"):
        ps.get_p_tp_group()
    assert [group.group_name for group in init_calls] == ["mc2", "dynamic_eplb", "fc3_quant_x"]


def test_init_parallel_state_uses_num_head_replica_and_flashcomm2_size_one(monkeypatch):
    ascend_config = make_ascend_config(
        num_head_replica=2,
        flashcomm2_oproj_tensor_parallel_size=1,
        layer_sharding=object(),
        multistream_overlap_gate=False,
    )
    _, tp_group, init_calls = patch_common_init(monkeypatch, ascend_config, flashcomm2=True)

    ps.init_ascend_model_parallel(make_parallel_config())

    assert ps.get_p_tp_group().group_name.startswith("p_tp_")
    assert ps.get_shard_weight_group().group_name == "shard_weight"
    assert ps.get_flashcomm2_otp_group() is None
    assert ps.get_flashcomm2_odp_group() is tp_group
    assert [group.group_name for group in init_calls] == ["p_tp_0", "mc2", "otp", "shard_weight"]


def test_init_parallel_state_uses_dsa_cp_shard_weight_branch(monkeypatch):
    ascend_config = make_ascend_config(
        pd_head_ratio=1,
        layer_sharding=object(),
        otp_size=0,
        lmhead_tp_size=0,
        embedding_tp_size=0,
        mlp_tp_size=0,
    )
    _, _, init_calls = patch_common_init(monkeypatch, ascend_config, flashcomm2=False, enable_dsa=True)

    ps.init_ascend_model_parallel(make_parallel_config())

    assert ps.get_shard_weight_group().group_name == "shard_weight"
    assert [group.group_name for group in init_calls] == ["mc2", "shard_weight"]


def test_init_parallel_state_uses_standard_tp_shard_weight_branch(monkeypatch):
    ascend_config = make_ascend_config(
        pd_head_ratio=1,
        layer_sharding=object(),
        otp_size=0,
        lmhead_tp_size=0,
        embedding_tp_size=0,
        mlp_tp_size=0,
    )
    _, _, init_calls = patch_common_init(monkeypatch, ascend_config, flashcomm2=False, enable_dsa=False)

    ps.init_ascend_model_parallel(make_parallel_config())

    assert ps.get_shard_weight_group().group_name == "shard_weight"
    assert [group.group_name for group in init_calls] == ["mc2", "shard_weight"]