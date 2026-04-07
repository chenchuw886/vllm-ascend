from types import SimpleNamespace

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import config_data as config_module


class FakeBlockHash:

    def __init__(self, value):
        self._value = value

    def hex(self):
        return self._value


def make_metadata():
    return config_module.KeyMetadata(
        model_name="test-model",
        head_or_tp_rank=2,
        pcp_rank=1,
        dcp_rank=3,
        pp_rank=0,
    )


def test_pool_key_and_layer_pool_key_helpers():
    metadata = make_metadata()
    pool_key = config_module.PoolKey(metadata, "chunk-1")

    assert isinstance(hash(pool_key), int)
    assert pool_key.to_string() == "test-model@pcp1@dcp3@head_or_tp_rank:2@pp_rank:0@chunk-1"

    layer_keys = pool_key.split_layers(3)
    assert [key.layer_id for key in layer_keys] == [0, 1, 2]
    assert all(key.key_metadata is metadata for key in layer_keys)

    layer_key = config_module.LayerPoolKey(metadata, "chunk-2", 5)
    assert isinstance(hash(layer_key), int)
    assert layer_key.to_string() == "test-model@pcp1@dcp3@head_or_tp_rank:2@chunk-2@5"


def test_chunked_token_database_basic_helpers_and_prepare_values():
    database = config_module.ChunkedTokenDatabase(make_metadata(), block_size=4, partitions=None)
    database.set_kv_caches_base_addr([100, 200, 300, 400])
    database.set_block_len([8, 12])

    key = database._make_key_by_hash("hash-a")
    assert isinstance(key, config_module.PoolKey)
    assert key.chunk_hash == "hash-a"

    addr_list, size_list, block_id = database.prepare_value(start=4, end=8, block_ids=[5, 6])
    assert (addr_list, size_list, block_id) == ([148, 272, 348, 472], [8, 12, 8, 12], 6)

    database.set_kv_caches_base_addr([1000, 2000, 3000, 4000])
    database.set_block_len([16, 32])
    layer_addr, layer_size = database.prepare_value_layer(start=4, end=8, block_ids=[7, 8], layer_id=1)
    assert layer_addr == [3128, 3256]
    assert layer_size == [16, 32]


def test_chunked_token_database_make_key_requires_metadata():
    database = config_module.ChunkedTokenDatabase(make_metadata(), block_size=4, partitions=None)
    database.metadata = None

    with pytest.raises(AssertionError):
        database._make_key_by_hash("missing")


def test_chunked_token_database_process_tokens_handles_empty_strings_and_hash_objects():
    database = config_module.ChunkedTokenDatabase(make_metadata(), block_size=4, partitions=None)

    assert list(database.process_tokens(token_len=8, block_hashes=[])) == []

    string_results = list(database.process_tokens(token_len=6, block_hashes=["h1", "h2", "h3"], mask_num=4))
    assert [(start, end, key.chunk_hash) for start, end, key in string_results] == [(4, 6, "h2")]

    object_results = list(
        database.process_tokens(token_len=7, block_hashes=[FakeBlockHash("aa"), FakeBlockHash("bb")], mask_num=0)
    )
    assert [(start, end, key.chunk_hash) for start, end, key in object_results] == [(0, 4, "aa"), (4, 7, "bb")]


def test_decode_adaptor_prefill_pp_covers_passthrough_and_partition_split():
    database = config_module.ChunkedTokenDatabase(make_metadata(), block_size=4, partitions=None)
    keys = ["model@pp_rank:0@chunk"]
    addrs = [[1, 2, 3, 4]]
    sizes = [[10, 20, 30, 40]]

    assert database.decode_adaptor_prefill_pp(keys, addrs, sizes) == (keys, addrs, sizes)

    database.partitions = [1]
    assert database.decode_adaptor_prefill_pp(keys, addrs, sizes) == (keys, addrs, sizes)

    database.partitions = [1, 1]
    new_keys, new_addrs, new_sizes = database.decode_adaptor_prefill_pp(keys, addrs, sizes)
    assert new_keys == ["model@pp_rank:0@chunk", "model@pp_rank:1@chunk"]
    assert new_addrs == [[1, 2], [3, 4]]
    assert new_sizes == [[10, 20], [30, 40]]


def test_request_tracker_from_new_request_supports_flat_and_nested_block_ids():
    flat_request = SimpleNamespace(req_id="req-flat", prompt_token_ids=[1, 2, 3, 4], block_ids=[9, 10])
    nested_request = SimpleNamespace(req_id="req-nested", prompt_token_ids=[5, 6, 7], block_ids=[[11, 12]])

    flat_tracker = config_module.RequestTracker.from_new_request(flat_request, num_tokens_to_compute=3)
    nested_tracker = config_module.RequestTracker.from_new_request(nested_request, num_tokens_to_compute=2)

    assert flat_tracker == config_module.RequestTracker(
        req_id="req-flat",
        token_len=3,
        allocated_block_ids=[9, 10],
        num_saved_tokens=0,
        token_ids=[1, 2, 3],
    )
    assert nested_tracker.allocated_block_ids == [11, 12]
    assert nested_tracker.token_ids == [5, 6]


def test_request_tracker_update_covers_empty_tuple_list_and_invalid_type():
    tracker = config_module.RequestTracker(req_id="req", token_len=4, allocated_block_ids=[1], token_ids=None)

    tracker.update([])
    tracker.update(([2, 3],))
    tracker.update([4])

    assert tracker.allocated_block_ids == [1, 2, 3, 4]

    with pytest.raises(ValueError, match="Unsupported new_block_ids type"):
        tracker.update("bad")


def test_req_meta_from_request_tracker_returns_none_when_nothing_to_save_or_load():
    tracker = config_module.RequestTracker(
        req_id="req-none",
        token_len=3,
        allocated_block_ids=[1, 2],
        num_saved_tokens=4,
        token_ids=None,
    )

    assert config_module.ReqMeta.from_request_tracker(tracker, block_size=4) is None


def test_req_meta_from_request_tracker_updates_saved_tokens_and_loads_when_allowed(monkeypatch):
    tracker = config_module.RequestTracker(
        req_id="req-load",
        token_len=10,
        allocated_block_ids=[7, 8],
        num_saved_tokens=0,
        token_ids=[101, 102],
    )
    load_spec = config_module.LoadSpec(vllm_cached_tokens=2, kvpool_cached_tokens=6, can_load=True, token_len=6)
    debug_logs = []

    monkeypatch.setattr(config_module.logger, "debug", lambda *args: debug_logs.append(args))

    req_meta = config_module.ReqMeta.from_request_tracker(
        tracker,
        block_size=4,
        load_spec=load_spec,
        block_hashes=["hash-1"],
        is_last_chunk=True,
        original_block_size=16,
    )

    assert tracker.num_saved_tokens == 8
    assert req_meta == config_module.ReqMeta(
        req_id="req-load",
        token_len_chunk=8,
        block_ids=[7, 8],
        can_save=True,
        load_spec=load_spec,
        block_hashes=["hash-1"],
        is_last_chunk=True,
        token_ids=[101, 102],
        original_block_size=16,
    )
    assert any("Scheduled to load" in args[0] for args in debug_logs)


def test_req_meta_from_request_tracker_clears_load_spec_when_cannot_load(monkeypatch):
    tracker = config_module.RequestTracker(
        req_id="req-no-load",
        token_len=5,
        allocated_block_ids=[3],
        num_saved_tokens=0,
        token_ids=[],
    )
    debug_logs = []
    monkeypatch.setattr(config_module.logger, "debug", lambda *args: debug_logs.append(args))

    req_meta = config_module.ReqMeta.from_request_tracker(
        tracker,
        block_size=4,
        load_spec=config_module.LoadSpec(vllm_cached_tokens=1, kvpool_cached_tokens=2, can_load=False),
        skip_save=True,
        discard_partial_chunks=False,
        block_hashes=None,
        is_last_chunk=False,
    )

    assert tracker.num_saved_tokens == 0
    assert req_meta.load_spec is None
    assert req_meta.can_save is False
    assert req_meta.token_len_chunk == 5
    assert req_meta.block_hashes == []
    assert any("meta save spec:False" in args[0] for args in debug_logs)


def test_ascend_connector_metadata_and_layer_multi_block_req_meta():
    connector_metadata = config_module.AscendConnectorMetadata(unfinished_request_ids={"req-1"}, preempted_req_ids={"req-2"})
    req_meta = config_module.ReqMeta(
        req_id="req-3",
        token_len_chunk=4,
        block_ids=[1],
        block_hashes=["hash"],
    )

    connector_metadata.add_request(req_meta)

    assert connector_metadata.requests == [req_meta]
    assert connector_metadata.unfinished_request_ids == {"req-1"}
    assert connector_metadata.preempted_req_ids == {"req-2"}

    layer_meta = config_module.LasyerMultiBlockReqMeta(
        req_id="req-layer",
        keys=[config_module.LayerPoolKey(make_metadata(), "chunk", 0)],
        starts=[0],
        ends=[4],
        block_ids=[9],
        layer_id=0,
    )
    assert layer_meta.req_id == "req-layer"
    assert layer_meta.is_last_chunk is True