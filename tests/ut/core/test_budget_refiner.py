from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd

from vllm_ascend.core.scheduler_dynamic_batch import BudgetRefiner


class FakeLookupTable:

    def __init__(self, groups):
        self._groups = groups

    def groupby(self, keys):
        assert keys == ["ctx_len", "d_num"]
        return self._groups


@patch("vllm_ascend.core.scheduler_dynamic_batch.logger.info")
def test_budget_refiner_disabled_without_positive_slo_limit(mock_info):
    refiner = BudgetRefiner(default_budget=64, slo_limit=0)

    assert refiner.enabled is False
    mock_info.assert_not_called()


@patch("vllm_ascend.core.scheduler_dynamic_batch.logger.error")
@patch("vllm_ascend.core.scheduler_dynamic_batch.os.path.exists", return_value=False)
def test_budget_refiner_disables_itself_when_lookup_table_missing(
    _mock_exists,
    mock_error,
):
    refiner = BudgetRefiner(default_budget=32, slo_limit=10)

    assert refiner.enabled is False
    mock_error.assert_called_once()


@patch("vllm_ascend.core.scheduler_dynamic_batch.os.path.exists", return_value=True)
@patch("vllm_ascend.core.scheduler_dynamic_batch.pd.read_csv")
def test_budget_refiner_loads_valid_lookup_entries(mock_read_csv, _mock_exists):
    groups = [
        (
            (16, 2),
            pd.DataFrame(
                {
                    "cost": [5, 12],
                    "chunk_size": [64, 128],
                }
            ),
        ),
        (
            (32, 4),
            pd.DataFrame(
                {
                    "cost": [18],
                    "chunk_size": [256],
                }
            ),
        ),
    ]
    mock_read_csv.return_value = FakeLookupTable(groups)

    refiner = BudgetRefiner(default_budget=32, slo_limit=10)

    assert refiner.enabled is True
    assert refiner.lookup == {(16, 2): 64}
    assert refiner.context_keys == {16}
    assert refiner.dnum_keys == {2}


def test_budget_refiner_align_key_returns_next_key_or_none():
    refiner = BudgetRefiner(default_budget=32, slo_limit=0)

    assert refiner._align_key(5, [3, 6, 9]) == 6
    assert refiner._align_key(9, [3, 6, 9]) == 9
    assert refiner._align_key(10, [3, 6, 9]) is None


@patch("vllm_ascend.core.scheduler_dynamic_batch.logger.warn")
def test_budget_refiner_get_max_budget_handles_alignment_and_table_miss(mock_warn):
    refiner = BudgetRefiner.__new__(BudgetRefiner)
    refiner.default_budget = 99
    refiner.lookup = {}
    refiner.context_keys = {10}
    refiner.dnum_keys = {2}

    assert refiner._get_max_budget(20, 1) == 99
    mock_warn.assert_not_called()

    assert refiner._get_max_budget(9, 2) == 99
    mock_warn.assert_called_once()


def test_budget_refiner_get_max_budget_returns_exact_lookup_value():
    refiner = BudgetRefiner.__new__(BudgetRefiner)
    refiner.default_budget = 99
    refiner.lookup = {(10, 2): 64}
    refiner.context_keys = {10}
    refiner.dnum_keys = {2}

    assert refiner._get_max_budget(10, 2) == 64


def test_budget_refiner_refine_budget_returns_original_budget_when_disabled_or_no_decode():
    disabled_refiner = BudgetRefiner(default_budget=64, slo_limit=0)
    requests = [SimpleNamespace(num_tokens_with_spec=16, num_computed_tokens=4, num_prompt_tokens=8)]

    assert disabled_refiner.refine_budget(requests, 48) == 48

    enabled_refiner = BudgetRefiner.__new__(BudgetRefiner)
    enabled_refiner.enabled = True
    enabled_refiner._get_max_budget = MagicMock(return_value=77)

    assert enabled_refiner.refine_budget(requests, 48) == 48
    enabled_refiner._get_max_budget.assert_not_called()


def test_budget_refiner_refine_budget_uses_average_decode_context():
    refiner = BudgetRefiner.__new__(BudgetRefiner)
    refiner.enabled = True
    refiner._get_max_budget = MagicMock(return_value=77)
    running_requests = [
        SimpleNamespace(num_tokens_with_spec=16, num_computed_tokens=8, num_prompt_tokens=8),
        SimpleNamespace(num_tokens_with_spec=24, num_computed_tokens=12, num_prompt_tokens=12),
        SimpleNamespace(num_tokens_with_spec=100, num_computed_tokens=7, num_prompt_tokens=8),
    ]

    refined = refiner.refine_budget(running_requests, 48)

    assert refined == 77
    refiner._get_max_budget.assert_called_once_with(20.0, 2)
