from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from theme_extractor.cli import main


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


def test_benchmark_runs_all_baseline_methods_with_single_backend(monkeypatch, capsys) -> None:
    monkeypatch.setattr("theme_extractor.cli.command_handlers.build_search_backend", lambda **_kwargs: _BackendStub())

    exit_code = main(
        [
            "benchmark",
            "--methods",
            "baseline_tfidf,terms,significant_terms,significant_text",
            "--focus",
            "topics",
        ],
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["methods"] == [
        "baseline_tfidf",
        "terms",
        "significant_terms",
        "significant_text",
    ]
    assert payload["outputs"]["baseline_tfidf"]["topics"]
    assert payload["outputs"]["terms"]["topics"][0]["label"] == "alpha"
    assert payload["outputs"]["significant_terms"]["topics"][0]["label"] == "beta"
    assert payload["outputs"]["significant_text"]["topics"][0]["label"] == "gamma"
