from unittest.mock import MagicMock, call

from vllm_ascend.distributed.kv_transfer import register_connector


def test_register_connector_replaces_multi_connector(monkeypatch):
    register = MagicMock()
    registry = {"MultiConnector": object(), "OtherConnector": object()}

    monkeypatch.setattr(
        "vllm_ascend.distributed.kv_transfer.KVConnectorFactory.register_connector",
        register,
    )
    monkeypatch.setattr("vllm_ascend.distributed.kv_transfer.KVConnectorFactory._registry", registry)

    register_connector()

    assert "MultiConnector" not in registry
    assert register.call_args_list == [
        call("MultiConnector", "vllm_ascend.distributed.kv_transfer.ascend_multi_connector", "AscendMultiConnector"),
        call("MooncakeConnectorV1", "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector", "MooncakeConnector"),
        call(
            "MooncakeConnectorStoreV1",
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector",
            "AscendStoreConnector",
        ),
        call(
            "AscendStoreConnector",
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector",
            "AscendStoreConnector",
        ),
        call(
            "MooncakeLayerwiseConnector",
            "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector",
            "MooncakeLayerwiseConnector",
        ),
        call("UCMConnector", "vllm_ascend.distributed.kv_transfer.kv_pool.ucm_connector", "UCMConnectorV1"),
        call(
            "LMCacheAscendConnector",
            "vllm_ascend.distributed.kv_transfer.kv_pool.lmcache_ascend_connector",
            "LMCacheConnectorV1",
        ),
    ]


def test_register_connector_handles_missing_multi_connector(monkeypatch):
    register = MagicMock()
    registry = {"Existing": object()}

    monkeypatch.setattr(
        "vllm_ascend.distributed.kv_transfer.KVConnectorFactory.register_connector",
        register,
    )
    monkeypatch.setattr("vllm_ascend.distributed.kv_transfer.KVConnectorFactory._registry", registry)

    register_connector()

    assert list(registry.keys()) == ["Existing"]
    assert register.call_count == 7