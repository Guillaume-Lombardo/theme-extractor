from __future__ import annotations

import pytest

from theme_extractor.search.adapters import ElasticClientAdapter, OpenSearchClientAdapter
from theme_extractor.search.factory import build_search_backend

_EXPECTED_OPERATION_COUNT = 4


class _StubClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, object]]] = []

    def search(self, *, index: str, body: dict[str, object]) -> dict[str, object]:
        self.calls.append(("search", index, body))
        return {"hits": {"hits": []}, "index": index, "body": body}


def test_elastic_adapter_delegates_all_operations() -> None:
    client = _StubClient()
    adapter = ElasticClientAdapter(client=client)

    body = {"query": {"match_all": {}}}

    adapter.search_documents(index="idx", body=body)
    adapter.terms_aggregation(index="idx", body=body)
    adapter.significant_terms_aggregation(index="idx", body=body)
    adapter.significant_text_aggregation(index="idx", body=body)

    assert len(client.calls) == _EXPECTED_OPERATION_COUNT
    assert all(call[0] == "search" for call in client.calls)
    assert all(call[1] == "idx" for call in client.calls)


def test_opensearch_adapter_delegates_all_operations() -> None:
    client = _StubClient()
    adapter = OpenSearchClientAdapter(client=client)

    body = {"query": {"match": {"content": "banking"}}}

    adapter.search_documents(index="docs", body=body)
    adapter.terms_aggregation(index="docs", body=body)
    adapter.significant_terms_aggregation(index="docs", body=body)
    adapter.significant_text_aggregation(index="docs", body=body)

    assert len(client.calls) == _EXPECTED_OPERATION_COUNT
    assert all(call[1] == "docs" for call in client.calls)


def test_backend_factory_supports_elasticsearch_and_opensearch_with_client_instance() -> None:
    elastic = build_search_backend(backend="elasticsearch", client=_StubClient())
    opensearch = build_search_backend(backend="opensearch", client=_StubClient())

    assert elastic.backend_name == "elasticsearch"
    assert opensearch.backend_name == "opensearch"


def test_backend_factory_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unsupported backend"):
        build_search_backend(backend="unknown", client=_StubClient())
