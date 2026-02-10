from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from theme_extractor.domain import BackendName
from theme_extractor.errors import (
    MissingBackendUrlError,
    MissingOptionalDependencyError,
    UnsupportedBackendError,
)
from theme_extractor.search import adapters
from theme_extractor.search.adapters import ElasticClientAdapter, OpenSearchClientAdapter
from theme_extractor.search.factory import build_search_backend, supported_backends

_EXPECTED_CALL_COUNT = 4
_ELASTIC_TIMEOUT = 12
_OPENSEARCH_TIMEOUT = 42


@dataclass
class _SearchClientStub:
    calls: list[tuple[str, dict[str, object]]]

    def search(self, *, index: str, body: dict[str, object]) -> dict[str, object]:
        self.calls.append((index, body))
        return {"ok": True, "index": index, "body": body}


def test_supported_backends_returns_expected_values() -> None:
    assert supported_backends() == (BackendName.ELASTICSEARCH, BackendName.OPENSEARCH)


def test_build_search_backend_with_client_for_elasticsearch() -> None:
    backend = build_search_backend(backend="elasticsearch", client=_SearchClientStub(calls=[]))

    assert backend.backend_name == "elasticsearch"


def test_build_search_backend_with_client_for_opensearch() -> None:
    backend = build_search_backend(backend="opensearch", client=_SearchClientStub(calls=[]))

    assert backend.backend_name == "opensearch"


def test_build_search_backend_requires_url_without_injected_client() -> None:
    with pytest.raises(MissingBackendUrlError):
        build_search_backend(backend="elasticsearch", client=None, url=None)

    with pytest.raises(MissingBackendUrlError):
        build_search_backend(backend="opensearch", client=None, url=None)


def test_build_search_backend_rejects_unsupported_backend() -> None:
    with pytest.raises(UnsupportedBackendError, match="Unsupported backend"):
        build_search_backend(backend="foo", client=_SearchClientStub(calls=[]))


def test_elastic_adapter_methods_delegate_to_search() -> None:
    client = _SearchClientStub(calls=[])
    adapter = ElasticClientAdapter(client=client)

    body = {"query": {"match_all": {}}}
    adapter.search_documents(index="idx", body=body)
    adapter.terms_aggregation(index="idx", body=body)
    adapter.significant_terms_aggregation(index="idx", body=body)
    adapter.significant_text_aggregation(index="idx", body=body)

    assert len(client.calls) == _EXPECTED_CALL_COUNT


def test_opensearch_adapter_methods_delegate_to_search() -> None:
    client = _SearchClientStub(calls=[])
    adapter = OpenSearchClientAdapter(client=client)

    body = {"query": {"match": {"content": "risk"}}}
    adapter.search_documents(index="docs", body=body)
    adapter.terms_aggregation(index="docs", body=body)
    adapter.significant_terms_aggregation(index="docs", body=body)
    adapter.significant_text_aggregation(index="docs", body=body)

    assert len(client.calls) == _EXPECTED_CALL_COUNT


def test_elastic_from_connection_raises_when_dependency_missing(monkeypatch) -> None:
    def _raise_import_error(_module_name: str):
        raise ImportError("missing")

    monkeypatch.setattr(adapters, "import_module", _raise_import_error)

    with pytest.raises(MissingOptionalDependencyError, match="optional dependency group 'elasticsearch'"):
        ElasticClientAdapter.from_connection(
            url="http://localhost:9200",
            timeout_s=30,
            verify_certs=True,
        )


def test_opensearch_from_connection_raises_when_dependency_missing(monkeypatch) -> None:
    def _raise_import_error(_module_name: str):
        raise ImportError("missing")

    monkeypatch.setattr(adapters, "import_module", _raise_import_error)

    with pytest.raises(MissingOptionalDependencyError, match="optional dependency group 'opensearch'"):
        OpenSearchClientAdapter.from_connection(
            url="http://localhost:9200",
            timeout_s=30,
            verify_certs=True,
        )


def test_elastic_from_connection_builds_client_when_dependency_exists(monkeypatch) -> None:
    created: dict[str, object] = {}

    class _ElasticClient:
        def __init__(self, **kwargs: object):
            created.update(kwargs)
            self._initialized = True

        def search(self, *, index: str, body: dict[str, object]) -> dict[str, object]:
            _ = self._initialized
            return {"index": index, "body": body}

    monkeypatch.setattr(adapters, "import_module", lambda _: SimpleNamespace(Elasticsearch=_ElasticClient))

    adapter = ElasticClientAdapter.from_connection(
        url="http://localhost:9200",
        timeout_s=_ELASTIC_TIMEOUT,
        verify_certs=False,
    )

    assert adapter.backend_name == "elasticsearch"
    assert created["hosts"] == ["http://localhost:9200"]
    assert created["request_timeout"] == _ELASTIC_TIMEOUT
    assert created["verify_certs"] is False


def test_opensearch_from_connection_builds_client_when_dependency_exists(monkeypatch) -> None:
    created: dict[str, object] = {}

    class _OpenSearchClient:
        def __init__(self, **kwargs: object):
            created.update(kwargs)
            self._initialized = True

        def search(self, *, index: str, body: dict[str, object]) -> dict[str, object]:
            _ = self._initialized
            return {"index": index, "body": body}

    monkeypatch.setattr(adapters, "import_module", lambda _: SimpleNamespace(OpenSearch=_OpenSearchClient))

    adapter = OpenSearchClientAdapter.from_connection(
        url="https://localhost:9200",
        timeout_s=_OPENSEARCH_TIMEOUT,
        verify_certs=False,
    )

    assert adapter.backend_name == "opensearch"
    assert created["hosts"] == ["https://localhost:9200"]
    assert created["timeout"] == _OPENSEARCH_TIMEOUT
    assert created["use_ssl"] is True
    assert created["verify_certs"] is False
