from types import SimpleNamespace

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import pool_scheduler as scheduler_module
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    AscendConnectorMetadata,
    LoadSpec,
    RequestTracker,
)


class FakeHash:

    def __init__(self, value):
        self._value = value

    def hex(self):
        return self._value


class FakeBlocks:

    def __init__(self, block_ids):
        self._block_ids = block_ids

    def get_block_ids(self):
        return [self._block_ids]


def make_vllm_config(*, kv_role="kv_both", extra=None, block_size=4, pcp_size=1, dcp_size=1):
    extra = extra or {}
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_role=kv_role,
            kv_connector_extra_config=extra,
            get_from_extra_config=lambda key, default: extra.get(key, default),
        ),
        cache_config=SimpleNamespace(block_size=block_size),
        parallel_config=SimpleNamespace(
            data_parallel_rank=1,
            prefill_context_parallel_size=pcp_size,
            decode_context_parallel_size=dcp_size,
        ),
    )


def make_request(
    req_id="req-1",
    *,
    prompt_token_ids=None,
    block_hashes=None,
    num_tokens=None,
    block_ids=None,
    num_computed_tokens=0,
    all_token_ids=None,
):
    prompt_token_ids = prompt_token_ids or [1, 2, 3, 4, 5, 6, 7, 8]
    if num_tokens is None:
        num_tokens = len(prompt_token_ids)
    return SimpleNamespace(
        request_id=req_id,
        req_id=req_id,
        prompt_token_ids=prompt_token_ids,
        block_hashes=block_hashes or [FakeHash("h1"), FakeHash("h2")],
        num_tokens=num_tokens,
        block_ids=block_ids or [[10, 11]],
        num_computed_tokens=num_computed_tokens,
        all_token_ids=all_token_ids or prompt_token_ids + [9, 10, 11],
    )


def make_scheduler_output(**overrides):
    base = dict(
        finished_req_ids=[],
        preempted_req_ids=set(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[], new_block_ids=[], num_computed_tokens=[]),
        num_scheduled_tokens={},
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def make_scheduler(monkeypatch, *, lookup_result=6, **cfg_kwargs):
    class FakeClient:
        def __init__(self, _cfg):
            self.calls = []

        def lookup(self, token_len, block_hashes):
            self.calls.append((token_len, block_hashes))
            return lookup_result

    monkeypatch.setattr(scheduler_module, "LookupKeyClient", FakeClient)
    scheduler = scheduler_module.KVPoolScheduler(make_vllm_config(**cfg_kwargs), use_layerwise=False)
    return scheduler, scheduler.client


def test_scheduler_init_scales_block_size_for_pcp_and_dcp(monkeypatch):
    scheduler, _client = make_scheduler(monkeypatch, pcp_size=2, dcp_size=3, block_size=4)

    assert scheduler.original_block_size == 4
    assert scheduler._block_size == 24


def test_get_num_new_matched_tokens_covers_consumer_small_request_and_async(monkeypatch):
    scheduler, client = make_scheduler(
        monkeypatch,
        lookup_result=8,
        kv_role="kv_consumer",
        extra={"consumer_is_to_load": False, "load_async": True},
    )
    request = make_request(prompt_token_ids=[1, 2, 3], num_tokens=3)

    assert scheduler.get_num_new_matched_tokens(request, 1) == (0, False)
    assert client.calls == []

    scheduler.kv_role = "kv_both"
    scheduler.consumer_is_to_load = True
    assert scheduler.get_num_new_matched_tokens(request, 1) == (0, False)

    full_request = make_request(req_id="req-2")
    need_to_allocate, async_load = scheduler.get_num_new_matched_tokens(full_request, 2)

    assert (need_to_allocate, async_load) == (5, True)
    assert scheduler.load_specs["req-2"] == LoadSpec(vllm_cached_tokens=2, kvpool_cached_tokens=7, can_load=False, token_len=0)
    assert len(client.calls) == 1


def test_get_num_new_matched_tokens_adjusts_full_hit_and_below_computed(monkeypatch):
    scheduler, _client = make_scheduler(monkeypatch, lookup_result=8)
    request = make_request(req_id="req-hit")

    assert scheduler.get_num_new_matched_tokens(request, 8) == (0, False)
    assert scheduler.load_specs == {}

    scheduler.client.calls.clear()
    scheduler2, _client2 = make_scheduler(monkeypatch, lookup_result=2)
    assert scheduler2.get_num_new_matched_tokens(make_request(req_id="req-low"), 3) == (0, False)


def test_get_num_new_matched_tokens_without_partial_discard_uses_full_prompt_len(monkeypatch):
    scheduler, client = make_scheduler(monkeypatch, lookup_result=5, extra={"discard_partial_chunks": False})
    request = make_request(req_id="req-full", prompt_token_ids=[1, 2, 3, 4, 5])

    assert scheduler.get_num_new_matched_tokens(request, 1) == (3, False)
    assert client.calls[0][0] == 5


def test_update_state_after_alloc_sets_unfinished_and_load_state(monkeypatch):
    scheduler, _client = make_scheduler(monkeypatch)
    request = make_request(req_id="req-alloc")
    scheduler.load_specs["req-alloc"] = LoadSpec(vllm_cached_tokens=2, kvpool_cached_tokens=6, can_load=False)

    scheduler.update_state_after_alloc(request, FakeBlocks([20, 21]), 4)
    assert scheduler._unfinished_requests["req-alloc"] == (request, [20, 21])
    assert scheduler._unfinished_request_ids == {"req-alloc"}
    assert scheduler.load_specs["req-alloc"].can_load is True

    scheduler.load_specs["req-zero"] = LoadSpec(vllm_cached_tokens=2, kvpool_cached_tokens=6, can_load=True)
    zero_request = make_request(req_id="req-zero")
    scheduler.update_state_after_alloc(zero_request, FakeBlocks([30]), 0)
    assert scheduler.load_specs["req-zero"].can_load is False

    scheduler.load_specs["req-bad"] = LoadSpec(vllm_cached_tokens=2, kvpool_cached_tokens=6, can_load=False)
    with pytest.raises(AssertionError, match="Mismatch in number of tokens"):
        scheduler.update_state_after_alloc(make_request(req_id="req-bad"), FakeBlocks([1]), 3)


def test_update_state_after_alloc_returns_early_when_request_has_no_load_spec(monkeypatch):
    scheduler, _client = make_scheduler(monkeypatch)
    request = make_request(req_id="req-plain")

    assert scheduler.update_state_after_alloc(request, FakeBlocks([8]), 1) is None
    assert scheduler._unfinished_requests["req-plain"] == (request, [8])


def test_build_connector_meta_covers_new_cached_preempted_and_unfinished(monkeypatch):
    scheduler, _client = make_scheduler(monkeypatch, extra={"discard_partial_chunks": False})
    scheduler.load_specs = {
        "new": LoadSpec(vllm_cached_tokens=2, kvpool_cached_tokens=6, can_load=True),
        "preempted": LoadSpec(vllm_cached_tokens=1, kvpool_cached_tokens=5, can_load=True),
        "idle": LoadSpec(vllm_cached_tokens=1, kvpool_cached_tokens=7, can_load=True),
    }
    new_request = make_request(req_id="new", block_ids=[[100, 101]], prompt_token_ids=[1, 2, 3, 4, 5, 6])
    cached_request = make_request(
        req_id="cached",
        prompt_token_ids=[1, 2, 3, 4, 5, 6, 7],
        all_token_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9],
    )
    preempted_request = make_request(req_id="preempted", prompt_token_ids=[1, 2, 3, 4, 5])
    idle_request = make_request(req_id="idle", prompt_token_ids=[1, 2, 3, 4, 5, 6, 7, 8])

    scheduler._unfinished_requests = {
        "new": (new_request, [100, 101]),
        "cached": (cached_request, [200, 201]),
        "preempted": (preempted_request, [300, 301]),
        "idle": (idle_request, [400, 401]),
        "finished": (make_request(req_id="finished"), [999]),
    }
    scheduler._unfinished_request_ids = {"new", "cached", "preempted", "idle", "finished"}
    scheduler._preempted_req_ids = {"preempted"}
    scheduler._request_trackers = {
        "cached": RequestTracker(req_id="cached", token_len=2, allocated_block_ids=[200], num_saved_tokens=0, token_ids=[1, 2]),
        "finished": RequestTracker(req_id="finished", token_len=1, allocated_block_ids=[9], num_saved_tokens=0),
    }

    output = make_scheduler_output(
        finished_req_ids=["finished"],
        preempted_req_ids=set(),
        scheduled_new_reqs=[
            SimpleNamespace(req_id="new", num_computed_tokens=2, block_ids=[[100, 101]], prompt_token_ids=new_request.prompt_token_ids),
        ],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["cached", "preempted"],
            new_block_ids=[[202], ([302, 303],)],
            num_computed_tokens=[1, 0],
        ),
        num_scheduled_tokens={"new": 3, "cached": 2, "preempted": 4},
    )

    meta = scheduler.build_connector_meta(output)

    assert isinstance(meta, AscendConnectorMetadata)
    assert meta.unfinished_request_ids == {"new", "cached", "preempted", "idle"}
    assert meta.preempted_req_ids == set()
    assert {req.req_id for req in meta.requests} == {"new", "cached", "preempted", "idle"}
    meta_by_id = {req.req_id: req for req in meta.requests}
    assert meta_by_id["new"].load_spec.can_load is True
    assert meta_by_id["cached"].load_spec is None
    assert meta_by_id["preempted"].block_ids == [302, 303]
    assert meta_by_id["idle"].load_spec.kvpool_cached_tokens == 7
    assert meta_by_id["idle"].load_spec.token_len == 0
    assert "finished" not in scheduler._unfinished_requests
    assert "preempted" not in scheduler.load_specs


def test_build_connector_meta_skips_save_for_consumer_and_missing_unfinished_raises(monkeypatch):
    scheduler, _client = make_scheduler(monkeypatch, kv_role="kv_consumer", extra={"consumer_is_to_put": False})
    scheduler._request_trackers["req"] = RequestTracker(req_id="req", token_len=2, allocated_block_ids=[1], num_saved_tokens=0, token_ids=[1, 2])
    scheduler._unfinished_requests["req"] = (make_request(req_id="req", prompt_token_ids=[1, 2, 3]), [1])

    output = make_scheduler_output(
        scheduled_cached_reqs=SimpleNamespace(req_ids=["req"], new_block_ids=[[2]], num_computed_tokens=[0]),
        num_scheduled_tokens={"req": 1},
    )
    meta = scheduler.build_connector_meta(output)
    assert meta.requests == []

    scheduler2, _client2 = make_scheduler(monkeypatch)
    scheduler2._request_trackers["ghost"] = RequestTracker(req_id="ghost", token_len=1, allocated_block_ids=[1], num_saved_tokens=0)
    with pytest.raises(ValueError, match="not in _unfinished_requests"):
        scheduler2.build_connector_meta(
            make_scheduler_output(
                scheduled_cached_reqs=SimpleNamespace(req_ids=["ghost"], new_block_ids=[[2]], num_computed_tokens=[0]),
                num_scheduled_tokens={"ghost": 1},
            )
        )


def test_build_connector_meta_covers_preempted_cleanup_new_req_plain_blocks_and_cached_skips(monkeypatch):
    scheduler, _client = make_scheduler(monkeypatch, extra={"discard_partial_chunks": False})
    scheduler._request_trackers = {"gone": RequestTracker(req_id="gone", token_len=1, allocated_block_ids=[1], num_saved_tokens=0)}
    scheduler._unfinished_requests = {"gone": (make_request(req_id="gone"), [1]), "plain": (make_request(req_id="plain"), [5, 6])}
    scheduler._unfinished_request_ids = {"gone", "plain"}

    output = make_scheduler_output(
        preempted_req_ids={"gone"},
        scheduled_new_reqs=[
            SimpleNamespace(
                req_id="plain",
                num_computed_tokens=0,
                block_ids=[5, 6],
                prompt_token_ids=[1, 2, 3],
            )
        ],
        scheduled_cached_reqs=SimpleNamespace(req_ids=["skip"], new_block_ids=[[]], num_computed_tokens=[0]),
        num_scheduled_tokens={"plain": 1},
    )

    meta = scheduler.build_connector_meta(output)

    assert scheduler._request_trackers.get("gone") is None
    assert scheduler._unfinished_requests.get("gone") is None
    assert scheduler._preempted_req_ids == {"gone"}
    assert len(meta.requests) == 1
    assert meta.requests[0].req_id == "plain"
    assert meta.requests[0].block_ids == [5, 6]


def test_build_connector_meta_new_and_cached_loops_cover_multiple_iterations(monkeypatch):
    scheduler, _client = make_scheduler(monkeypatch, extra={"discard_partial_chunks": False})
    req_a = make_request(req_id="new-a", prompt_token_ids=[1, 2, 3, 4])
    req_b = make_request(req_id="new-b", prompt_token_ids=[1, 2, 3, 4, 5])
    cached_a = make_request(req_id="cached-a", prompt_token_ids=[1, 2, 3, 4, 5], all_token_ids=[1, 2, 3, 4, 5, 6])
    cached_b = make_request(req_id="cached-b", prompt_token_ids=[1, 2, 3, 4, 5, 6], all_token_ids=[1, 2, 3, 4, 5, 6, 7])
    scheduler._unfinished_requests = {
        "new-a": (req_a, [10]),
        "new-b": (req_b, [11]),
        "cached-a": (cached_a, [20]),
        "cached-b": (cached_b, [21]),
    }
    scheduler._unfinished_request_ids = {"new-a", "new-b", "cached-a", "cached-b"}
    scheduler._request_trackers = {
        "cached-a": RequestTracker(req_id="cached-a", token_len=1, allocated_block_ids=[20], num_saved_tokens=0, token_ids=[1]),
        "cached-b": RequestTracker(req_id="cached-b", token_len=2, allocated_block_ids=[21], num_saved_tokens=0, token_ids=[1, 2]),
    }

    meta = scheduler.build_connector_meta(
        make_scheduler_output(
            scheduled_new_reqs=[
                SimpleNamespace(req_id="new-a", num_computed_tokens=0, block_ids=[10], prompt_token_ids=req_a.prompt_token_ids),
                SimpleNamespace(req_id="new-b", num_computed_tokens=1, block_ids=[[11]], prompt_token_ids=req_b.prompt_token_ids),
            ],
            scheduled_cached_reqs=SimpleNamespace(
                req_ids=["cached-a", "cached-b"],
                new_block_ids=[[22], [23]],
                num_computed_tokens=[0, 1],
            ),
            num_scheduled_tokens={"new-a": 1, "new-b": 2, "cached-a": 1, "cached-b": 2},
        )
    )

    assert {req.req_id for req in meta.requests} == {"new-a", "new-b", "cached-a", "cached-b"}


def test_build_connector_meta_covers_preempted_list_blocks_and_cached_complete_prompt(monkeypatch):
    scheduler, _client = make_scheduler(monkeypatch, extra={"discard_partial_chunks": False})
    preempted_request = make_request(req_id="preempted-list", prompt_token_ids=[1, 2, 3, 4])
    full_cached_request = make_request(req_id="cached-full", prompt_token_ids=[1, 2, 3], all_token_ids=[1, 2, 3])
    scheduler._unfinished_requests = {
        "preempted-list": (preempted_request, [30, 31]),
        "cached-full": (full_cached_request, [40]),
    }
    scheduler._unfinished_request_ids = {"preempted-list", "cached-full"}
    scheduler._preempted_req_ids = {"preempted-list"}
    scheduler._request_trackers = {
        "cached-full": RequestTracker(req_id="cached-full", token_len=1, allocated_block_ids=[40], num_saved_tokens=0, token_ids=[1]),
    }
    scheduler.load_specs["preempted-list"] = LoadSpec(vllm_cached_tokens=1, kvpool_cached_tokens=4, can_load=True)

    meta = scheduler.build_connector_meta(
        make_scheduler_output(
            scheduled_cached_reqs=SimpleNamespace(
                req_ids=["preempted-list", "cached-full"],
                new_block_ids=[[32, 33], [41]],
                num_computed_tokens=[0, 3],
            ),
            num_scheduled_tokens={"preempted-list": 2, "cached-full": 1},
        )
    )

    meta_by_id = {req.req_id: req for req in meta.requests}
    assert meta_by_id["preempted-list"].block_ids == [32, 33]
    assert "cached-full" not in meta_by_id


def test_build_connector_meta_unfinished_request_paths_cover_continue_and_round_up(monkeypatch):
    scheduler, _client = make_scheduler(monkeypatch)
    scheduler._unfinished_requests = {
        "no-load": (make_request(req_id="no-load", prompt_token_ids=[1, 2, 3, 4]), [1]),
        "round-up": (make_request(req_id="round-up", prompt_token_ids=[1, 2, 3, 4, 5]), [7, 8]),
    }
    scheduler._unfinished_request_ids = {"no-load", "round-up"}
    scheduler.load_specs = {
        "round-up": LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=3, can_load=True),
    }

    meta = scheduler.build_connector_meta(make_scheduler_output())

    assert {req.req_id for req in meta.requests} == {"round-up"}
    round_up_meta = meta.requests[0]
    assert round_up_meta.token_len_chunk == 0
    assert round_up_meta.load_spec.kvpool_cached_tokens == 3


def test_build_connector_meta_unfinished_loop_covers_multiple_meta_additions(monkeypatch):
    scheduler, _client = make_scheduler(monkeypatch)
    scheduler._unfinished_requests = {
        "idle-a": (make_request(req_id="idle-a", prompt_token_ids=[1, 2, 3, 4]), [1]),
        "idle-b": (make_request(req_id="idle-b", prompt_token_ids=[1, 2, 3, 4]), [2]),
    }
    scheduler._unfinished_request_ids = {"idle-a", "idle-b"}
    scheduler.load_specs = {
        "idle-a": LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=4, can_load=True),
        "idle-b": LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=4, can_load=True),
    }

    meta = scheduler.build_connector_meta(make_scheduler_output())

    assert {req.req_id for req in meta.requests} == {"idle-a", "idle-b"}


def test_request_finished_covers_consumer_no_save_tracker_gate_and_delay(monkeypatch):
    scheduler, _client = make_scheduler(monkeypatch, kv_role="kv_consumer", extra={"consumer_is_to_put": False})
    request = make_request(req_id="finish")

    assert scheduler.request_finished(request, [1]) == (False, None)

    scheduler.kv_role = "kv_both"
    scheduler._request_trackers["finish"] = RequestTracker(req_id="finish", token_len=2, allocated_block_ids=[1], num_saved_tokens=0)
    assert scheduler.request_finished(request, [1]) == (False, None)

    scheduler._request_trackers["finish"].num_saved_tokens = 4
    assert scheduler.request_finished(request, []) == (False, None)
    assert scheduler.request_finished(request, [1, 2]) == (True, None)


def test_lookup_key_client_lookup_and_close(monkeypatch):
    fake_socket = SimpleNamespace(
        sent=None,
        send_multipart=lambda frames, copy=False: setattr(fake_socket, "sent", (frames, copy)),
        recv=lambda: (5).to_bytes(4, "big"),
        close=lambda linger=0: setattr(fake_socket, "closed", linger),
    )
    monkeypatch.setattr(scheduler_module.zmq, "Context", lambda: "ctx")
    monkeypatch.setattr(scheduler_module, "make_zmq_socket", lambda *args, **kwargs: fake_socket)

    client = scheduler_module.LookupKeyClient(make_vllm_config(extra={"lookup_rpc_port": 9}))
    result = client.lookup(12, [FakeHash("aa"), FakeHash("bb")])

    assert result == 5
    frames, copy_flag = fake_socket.sent
    assert int.from_bytes(frames[0], "big") == 12
    assert copy_flag is False

    client.close()
    assert fake_socket.closed == 0


def test_get_zmq_rpc_path_lookup_prefers_lookup_port_and_warns_on_legacy(monkeypatch):
    debug_logs = []
    warning_logs = []
    monkeypatch.setattr(scheduler_module.envs, "VLLM_RPC_BASE_PATH", "/tmp/vllm")
    monkeypatch.setattr(scheduler_module.logger, "debug", lambda *args: debug_logs.append(args))
    monkeypatch.setattr(scheduler_module.logger, "warning", lambda *args: warning_logs.append(args))

    cfg = make_vllm_config(extra={"lookup_rpc_port": 12})
    assert scheduler_module.get_zmq_rpc_path_lookup(cfg) == "ipc:///tmp/vllm/lookup_rpc_port_12_dp_rank1"

    legacy_cfg = make_vllm_config(extra={"mooncake_rpc_port": 15})
    assert scheduler_module.get_zmq_rpc_path_lookup(legacy_cfg) == "ipc:///tmp/vllm/lookup_rpc_port_15_dp_rank1"
    assert warning_logs
    assert debug_logs

    default_cfg = make_vllm_config(extra={})
    assert scheduler_module.get_zmq_rpc_path_lookup(default_cfg) == "ipc:///tmp/vllm/lookup_rpc_port_0_dp_rank1"
