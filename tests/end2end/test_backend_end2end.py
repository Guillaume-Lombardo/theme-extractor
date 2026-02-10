from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from theme_extractor.cli import main
from theme_extractor.errors import MissingOptionalDependencyError
from theme_extractor.search import adapters
from theme_extractor.search.adapters import ElasticClientAdapter, OpenSearchClientAdapter
from theme_extractor.search.factory import build_search_backend

_PARSER_ERROR_EXIT_CODE = 2


@dataclass
class _ClientStub:
    calls: list[tuple[str, dict[str, object]]]

    def search(self, *, index: str, body: dict[str, object]) -> dict[str, object]:
        self.calls.append((index, body))
        return {"hits": {"hits": []}, "index": index}


def test_end2end_backend_factory_and_adapter_calls() -> None:
    elastic = build_search_backend(backend="elasticsearch", client=_ClientStub(calls=[]))
    opensearch = build_search_backend(backend="opensearch", client=_ClientStub(calls=[]))

    body = {"query": {"match_all": {}}}

    elastic.search_documents(index="idx", body=body)
    elastic.terms_aggregation(index="idx", body=body)
    opensearch.significant_terms_aggregation(index="idx", body=body)
    opensearch.significant_text_aggregation(index="idx", body=body)

    assert elastic.backend_name == "elasticsearch"
    assert opensearch.backend_name == "opensearch"


def test_end2end_backend_factory_errors_are_explicit() -> None:
    with pytest.raises(ValueError, match="Unsupported backend"):
        build_search_backend(backend="unknown", client=_ClientStub(calls=[]))

    with pytest.raises(ValueError, match="Backend URL is required"):
        build_search_backend(backend="elasticsearch")


def test_end2end_optional_dependency_errors(monkeypatch) -> None:
    def _raise_import_error(_module_name: str):
        raise ImportError("missing")

    monkeypatch.setattr(adapters, "import_module", _raise_import_error)

    with pytest.raises(MissingOptionalDependencyError, match="elasticsearch"):
        ElasticClientAdapter.from_connection(
            url="http://localhost:9200",
            timeout_s=1,
            verify_certs=True,
        )

    with pytest.raises(MissingOptionalDependencyError, match="opensearch"):
        OpenSearchClientAdapter.from_connection(
            url="http://localhost:9200",
            timeout_s=1,
            verify_certs=True,
        )


def test_end2end_optional_dependency_success_paths(monkeypatch) -> None:
    class _ElasticClient:
        def __init__(self, **_kwargs: object):
            self.ok = True

        def search(self, *, index: str, body: dict[str, object]) -> dict[str, object]:
            _ = self.ok
            return {"index": index, "body": body}

    class _OpenSearchClient:
        def __init__(self, **_kwargs: object):
            self.ok = True

        def search(self, *, index: str, body: dict[str, object]) -> dict[str, object]:
            _ = self.ok
            return {"index": index, "body": body}

    monkeypatch.setattr(adapters, "import_module", lambda _: SimpleNamespace(Elasticsearch=_ElasticClient))
    elastic = ElasticClientAdapter.from_connection(
        url="http://localhost:9200",
        timeout_s=1,
        verify_certs=True,
    )
    elastic.search_documents(index="idx", body={"query": {"match_all": {}}})

    monkeypatch.setattr(adapters, "import_module", lambda _: SimpleNamespace(OpenSearch=_OpenSearchClient))
    opensearch = OpenSearchClientAdapter.from_connection(
        url="https://localhost:9200",
        timeout_s=1,
        verify_certs=True,
    )
    opensearch.search_documents(index="idx", body={"query": {"match_all": {}}})


def test_end2end_cli_parser_error_branch() -> None:
    exit_code = main(["extract", "--method", "invalid-choice"])
    assert exit_code == _PARSER_ERROR_EXIT_CODE


def test_end2end_cli_empty_method_and_unknown_method_errors() -> None:
    with pytest.raises(ValueError, match="At least one extraction method"):
        main(["benchmark", "--methods", ",,,"])

    with pytest.raises(ValueError, match="Unsupported extraction method"):
        main(["benchmark", "--methods", "unknown"])
