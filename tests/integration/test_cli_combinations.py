from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from theme_extractor.cli import main

_EXPECTED_METHOD_COUNT = 7


@dataclass
class _BackendStub:
    backend_name: str = "stub"

    def search_documents(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:  # noqa: PLR6301
        _ = index
        _ = body
        return {
            "hits": {
                "hits": [
                    {"_id": "doc-a", "_source": {"content": "alpha beta"}},
                    {"_id": "doc-b", "_source": {"content": "alpha gamma"}},
                ],
            },
        }

    def terms_aggregation(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:  # noqa: PLR6301
        _ = index
        _ = body
        return {
            "aggregations": {
                "terms": {
                    "buckets": [{"key": "alpha", "doc_count": 2}],
                },
            },
        }

    def significant_terms_aggregation(  # noqa: PLR6301
        self,
        *,
        index: str,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        _ = index
        _ = body
        return {
            "aggregations": {
                "themes": {
                    "buckets": [{"key": "beta", "doc_count": 1, "score": 4.2}],
                },
            },
        }

    def significant_text_aggregation(  # noqa: PLR6301
        self,
        *,
        index: str,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        _ = index
        _ = body
        return {
            "aggregations": {
                "themes": {
                    "buckets": [{"key": "gamma", "doc_count": 1, "score": 3.3}],
                },
            },
        }


def test_benchmark_supports_combined_methods_focus_and_backend(monkeypatch, capsys) -> None:
    monkeypatch.setattr("theme_extractor.cli.build_search_backend", lambda **_kwargs: _BackendStub())

    exit_code = main(
        [
            "benchmark",
            "--methods",
            "baseline_tfidf,terms,significant_terms,significant_text,keybert,bertopic,llm",
            "--focus",
            "topics",
            "--backend",
            "opensearch",
            "--index",
            "legal_docs",
            "--offline-policy",
            "preload_or_first_run",
        ],
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)

    assert len(payload["methods"]) == _EXPECTED_METHOD_COUNT
    assert payload["outputs"]["bertopic"]["metadata"]["backend"] == "opensearch"
    assert payload["outputs"]["llm"]["metadata"]["index"] == "legal_docs"
    assert payload["outputs"]["llm"]["metadata"]["offline_policy"] == "preload_or_first_run"


def test_extract_supports_strict_offline_policy_and_elasticsearch_backend(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr("theme_extractor.cli.build_search_backend", lambda **_kwargs: _BackendStub())

    exit_code = main(
        [
            "extract",
            "--method",
            "terms",
            "--focus",
            "both",
            "--backend",
            "elasticsearch",
            "--offline-policy",
            "strict",
            "--index",
            "theme_idx",
        ],
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["focus"] == "both"
    assert payload["document_topics"] == []
    assert payload["metadata"]["backend"] == "elasticsearch"
    assert payload["metadata"]["offline_policy"] == "strict"
    assert payload["metadata"]["index"] == "theme_idx"


def test_extract_bertopic_supports_matrix_flags_and_notes(monkeypatch, capsys) -> None:
    monkeypatch.setattr("theme_extractor.cli.build_search_backend", lambda **_kwargs: _BackendStub())
    monkeypatch.setattr(
        "theme_extractor.extraction.bertopic._make_embeddings_if_enabled",
        lambda **_kwargs: (None, "embedding fallback"),
    )
    monkeypatch.setattr(
        "theme_extractor.extraction.bertopic._apply_reduction",
        lambda **kwargs: (kwargs["matrix"], "reduction fallback"),
    )
    monkeypatch.setattr(
        "theme_extractor.extraction.bertopic._cluster_labels",
        lambda **_kwargs: (np.array([0, 0]), "cluster fallback"),
    )

    exit_code = main(
        [
            "extract",
            "--method",
            "bertopic",
            "--focus",
            "both",
            "--bertopic-use-embeddings",
            "--bertopic-dim-reduction",
            "umap",
            "--bertopic-clustering",
            "hdbscan",
            "--bertopic-min-topic-size",
            "1",
            "--bertopic-nr-topics",
            "2",
        ],
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["topics"]
    assert payload["document_topics"]
    assert "embedding fallback" in payload["notes"]
    assert "reduction fallback" in payload["notes"]
    assert "cluster fallback" in payload["notes"]
