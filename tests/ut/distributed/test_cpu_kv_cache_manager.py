from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.cpu_offload import cpu_kv_cache_manager as manager_module


class FakeBlock:

    def __init__(self, block_id):
        self.block_id = block_id


def test_cpu_cache_stats_log_and_reset(monkeypatch):
    logged = []
    time_values = iter([100, 105, 112])
    monkeypatch.setattr(manager_module.time, "time", lambda: next(time_values))
    monkeypatch.setattr(manager_module.logger, "info", lambda *args: logged.append(args))

    stats = manager_module.CPUCacheStats(enable_prefix_caching=True, log_stats=True)
    stats.cpu_prefix_cache_metrics = SimpleNamespace(hit_rate=0.25)

    stats.log()
    assert logged == []

    stats.log()
    assert len(logged) == 1
    assert logged[0][0] == "CPU Prefix cache hit rate: %.1f%%"

    old_stats = stats.prefix_cache_stats
    new_stats = stats.make_prefix_cache_stats()
    assert new_stats is old_stats
    assert stats.prefix_cache_stats is not old_stats


def test_cpu_cache_stats_handles_disabled_logging():
    stats = manager_module.CPUCacheStats(enable_prefix_caching=False, log_stats=False)

    assert stats.make_prefix_cache_stats() is None
    stats.update(10, 3)
    assert stats.prefix_cache_stats is None


def test_cpu_cache_stats_update_and_set_cache_stats():
    stats = manager_module.CPUCacheStats(enable_prefix_caching=True, log_stats=True)

    stats.update(8, 3)
    assert stats.prefix_cache_stats.requests == 1
    assert stats.prefix_cache_stats.queries == 8
    assert stats.prefix_cache_stats.hits == 3

    stats.set_cache_stats(12, 6)
    assert stats.prefix_cache_stats.requests == 1
    assert stats.prefix_cache_stats.queries == 12
    assert stats.prefix_cache_stats.hits == 6


def make_manager(monkeypatch, *, caching_hash_algo="builtin", use_eagle=False):
    fake_block_pool = SimpleNamespace(
        touch=MagicMock(),
        get_num_free_blocks=MagicMock(return_value=10),
        free_blocks=MagicMock(),
    )
    fake_single_type_manager = SimpleNamespace(
        kv_cache_spec="kv-spec",
        block_pool=fake_block_pool,
        find_longest_cache_hit=MagicMock(return_value=[[FakeBlock(1), FakeBlock(2)]]),
        get_num_blocks_to_allocate=MagicMock(return_value=1),
        save_new_computed_blocks=MagicMock(),
        allocate_new_blocks=MagicMock(return_value=[FakeBlock(3)]),
        cache_blocks=MagicMock(),
        free=MagicMock(),
    )
    fake_cpu_cache_stats = SimpleNamespace(
        prefix_cache_stats=SimpleNamespace(),
        set_cache_stats=MagicMock(),
        cpu_prefix_cache_metrics=SimpleNamespace(observe=MagicMock()),
        log=MagicMock(),
    )

    monkeypatch.setattr(manager_module, "BlockPool", lambda *args: fake_block_pool)
    monkeypatch.setattr(manager_module, "get_manager_for_kv_cache_spec", lambda **kwargs: fake_single_type_manager)
    monkeypatch.setattr(manager_module, "CPUCacheStats", lambda **kwargs: fake_cpu_cache_stats)

    manager = manager_module.CPUKVCacheManager(
        kv_cache_spec=SimpleNamespace(block_size=4),
        num_cpu_blocks=12,
        caching_hash_algo=caching_hash_algo,
        use_eagle=use_eagle,
    )
    return manager, fake_block_pool, fake_single_type_manager, fake_cpu_cache_stats


def test_cpu_kv_cache_manager_init_uses_requested_hash(monkeypatch):
    manager, fake_block_pool, fake_single_type_manager, _ = make_manager(monkeypatch, caching_hash_algo="sha256")

    assert manager.block_size == 4
    assert manager.num_cpu_blocks == 12
    assert manager.caching_hash_fn is manager_module.sha256
    assert manager.block_pool is fake_block_pool
    assert manager.single_type_manager is fake_single_type_manager


def test_get_matched_num_and_touch_skips_prompt_logprobs(monkeypatch):
    manager, fake_block_pool, fake_single_type_manager, fake_stats = make_manager(monkeypatch)
    request = SimpleNamespace(sampling_params=SimpleNamespace(prompt_logprobs=1))

    assert manager.get_matched_num_and_touch(request) == (0, False)
    fake_single_type_manager.find_longest_cache_hit.assert_not_called()
    fake_block_pool.touch.assert_not_called()
    fake_stats.set_cache_stats.assert_not_called()


def test_get_matched_num_and_touch_computes_and_caches_block_hashes(monkeypatch):
    manager, fake_block_pool, fake_single_type_manager, fake_stats = make_manager(monkeypatch, use_eagle=True)
    request = SimpleNamespace(
        request_id="req-1",
        sampling_params=SimpleNamespace(prompt_logprobs=None),
        block_hashes=["hash-a", "hash-b"],
        num_tokens=9,
    )

    result = manager.get_matched_num_and_touch(request)

    assert result == (8, False)
    assert manager.req_to_block_hashes["req-1"] == ["hash-a", "hash-b"]
    assert manager.req_to_computed_blocks["req-1"] == fake_single_type_manager.find_longest_cache_hit.return_value[0]
    fake_single_type_manager.find_longest_cache_hit.assert_called_once_with(
        block_hashes=["hash-a", "hash-b"],
        max_length=8,
        kv_cache_group_ids=[0],
        block_pool=fake_block_pool,
        kv_cache_spec="kv-spec",
        use_eagle=True,
        alignment_tokens=4,
    )
    fake_block_pool.touch.assert_called_once()
    fake_stats.set_cache_stats.assert_called_once_with(9, 8)
    fake_stats.cpu_prefix_cache_metrics.observe.assert_called_once_with(fake_stats.prefix_cache_stats)
    fake_stats.log.assert_called_once()


def test_get_matched_num_and_touch_reuses_existing_block_hashes(monkeypatch):
    manager, _fake_block_pool, fake_single_type_manager, _fake_stats = make_manager(monkeypatch)
    manager.req_to_block_hashes["req-2"] = ["cached-hash"]
    request = SimpleNamespace(
        request_id="req-2",
        sampling_params=SimpleNamespace(prompt_logprobs=None),
        block_hashes=["new-hash"],
        num_tokens=5,
    )

    manager.get_matched_num_and_touch(request)

    assert fake_single_type_manager.find_longest_cache_hit.call_args.kwargs["block_hashes"] == ["cached-hash"]


def test_release_ahead_touch_handles_present_and_missing_blocks(monkeypatch):
    manager, _fake_block_pool, fake_single_type_manager, _fake_stats = make_manager(monkeypatch)
    manager.req_to_computed_blocks["req-1"] = [FakeBlock(1), FakeBlock(2)]

    manager._release_ahead_touch("req-1")

    fake_single_type_manager.block_pool.free_blocks.assert_called_once()
    assert "req-1" not in manager.req_to_computed_blocks

    fake_single_type_manager.block_pool.free_blocks.reset_mock()
    manager.req_to_computed_blocks["req-2"] = []
    manager._release_ahead_touch("req-2")
    fake_single_type_manager.block_pool.free_blocks.assert_not_called()


def test_allocate_slots_covers_skip_failure_and_success(monkeypatch):
    manager, fake_block_pool, fake_single_type_manager, _fake_stats = make_manager(monkeypatch)
    manager.req_to_computed_blocks["skip"] = [FakeBlock(11)]
    manager.req_failed_to_allocate["skip"] = True
    manager.req_to_computed_blocks["fail"] = [FakeBlock(21)]
    manager.req_to_computed_blocks["ok"] = [FakeBlock(31)]
    free_calls = []

    monkeypatch.setattr(manager, "_free_slots", lambda request_id: free_calls.append(request_id))

    fake_single_type_manager.get_num_blocks_to_allocate.side_effect = [20, 1]
    fake_block_pool.get_num_free_blocks.side_effect = [5, 5]
    fake_single_type_manager.allocate_new_blocks.return_value = [FakeBlock(32), FakeBlock(33)]

    result = manager.allocate_slots(
        {"skip": 7, "fail": 8, "ok": 9},
        {"stale-1", "stale-2"},
    )

    assert set(free_calls) == {"stale-1", "stale-2"}
    assert manager.req_failed_to_allocate["fail"] is True
    saved_request_id, saved_blocks = fake_single_type_manager.save_new_computed_blocks.call_args.args
    assert saved_request_id == "ok"
    assert [block.block_id for block in saved_blocks] == [31]
    fake_single_type_manager.allocate_new_blocks.assert_called_once_with("ok", 9)
    assert manager.req_to_num_tokens["ok"] == 9
    assert result == {"ok": [31, 32, 33]}


def test_record_request_cache_and_free_slots(monkeypatch):
    manager, *_ = make_manager(monkeypatch)
    request = SimpleNamespace(request_id="req-1")

    manager.record_request_cache_and_free_slots(request)

    assert manager.req_to_free["req-1"] is request


def test_cache_and_free_slots_logs_missing_request(monkeypatch):
    manager, _fake_block_pool, fake_single_type_manager, _fake_stats = make_manager(monkeypatch)
    errors = []

    monkeypatch.setattr(manager_module.logger, "Error", lambda message: errors.append(message), raising=False)

    assert manager.cache_and_free_slots("missing") is None
    fake_single_type_manager.cache_blocks.assert_not_called()
    assert any("missing" in message for message in errors)


def test_cache_and_free_slots_caches_then_frees(monkeypatch):
    manager, _fake_block_pool, fake_single_type_manager, _fake_stats = make_manager(monkeypatch)
    request = SimpleNamespace(request_id="req-1")
    free_calls = []

    manager.req_to_free["req-1"] = request
    manager.req_to_num_tokens["req-1"] = 12
    monkeypatch.setattr(manager, "_free_slots", lambda request_id: free_calls.append(request_id))

    manager.cache_and_free_slots("req-1")

    fake_single_type_manager.cache_blocks.assert_called_once_with(request, 12)
    assert free_calls == ["req-1"]
    assert "req-1" not in manager.req_to_free


def test_cache_and_free_slots_skips_cache_when_allocate_failed(monkeypatch):
    manager, _fake_block_pool, fake_single_type_manager, _fake_stats = make_manager(monkeypatch)
    free_calls = []

    manager.req_to_free["req-2"] = SimpleNamespace(request_id="req-2")
    manager.req_failed_to_allocate["req-2"] = True
    manager.req_to_num_tokens["req-2"] = 6
    monkeypatch.setattr(manager, "_free_slots", lambda request_id: free_calls.append(request_id))

    manager.cache_and_free_slots("req-2")

    fake_single_type_manager.cache_blocks.assert_not_called()
    assert free_calls == ["req-2"]
    assert "req-2" not in manager.req_to_free


def test_free_slots_is_reentrant(monkeypatch):
    manager, _fake_block_pool, fake_single_type_manager, _fake_stats = make_manager(monkeypatch)
    released = []

    monkeypatch.setattr(manager, "_release_ahead_touch", lambda request_id: released.append(request_id))
    manager.req_to_block_hashes["req-3"] = ["hash"]
    manager.req_to_computed_blocks["req-3"] = [FakeBlock(1)]
    manager.req_failed_to_allocate["req-3"] = True
    manager.req_to_num_tokens["req-3"] = 5

    manager._free_slots("req-3")
    manager._free_slots("req-3")

    assert released == ["req-3", "req-3"]
    assert fake_single_type_manager.free.call_args_list[0].args == ("req-3",)
    assert fake_single_type_manager.free.call_args_list[1].args == ("req-3",)
    assert "req-3" not in manager.req_to_block_hashes
    assert "req-3" not in manager.req_to_computed_blocks
    assert "req-3" not in manager.req_failed_to_allocate
    assert "req-3" not in manager.req_to_num_tokens