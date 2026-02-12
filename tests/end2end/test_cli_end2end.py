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
        return {"aggregations": {"terms": {"buckets": []}}}

    def significant_terms_aggregation(  # noqa: PLR6301
        self,
        *,
        index: str,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        _ = index
        _ = body
        return {"aggregations": {"themes": {"buckets": []}}}

    def significant_text_aggregation(  # noqa: PLR6301
        self,
        *,
        index: str,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        _ = index
        _ = body
        return {"aggregations": {"themes": {"buckets": []}}}


def test_ingest_end2end_generates_json_output_file(tmp_path) -> None:
    corpus_file = tmp_path / "doc.txt"
    corpus_file.write_text(
        "Résumé de politique monétaire\\nPage 1/3\\nRésumé de politique monétaire",
        encoding="utf-8",
    )
    output_path = tmp_path / "ingest-e2e.json"

    exit_code = main(
        [
            "ingest",
            "--input",
            str(corpus_file),
            "--output",
            str(output_path),
        ],
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["command"] == "ingest"
    assert payload["input_path"] == str(corpus_file.resolve())
    assert payload["processed_documents"] == 1


def test_extract_end2end_generates_json_output_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("theme_extractor.cli.build_search_backend", lambda **_kwargs: _BackendStub())

    output_path = tmp_path / "extract-e2e.json"

    exit_code = main(
        [
            "extract",
            "--method",
            "keybert",
            "--focus",
            "topics",
            "--output",
            str(output_path),
        ],
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0"
    assert payload["metadata"]["method"] == "keybert"


def test_benchmark_end2end_generates_json_output_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("theme_extractor.cli.build_search_backend", lambda **_kwargs: _BackendStub())

    output_path = tmp_path / "benchmark-e2e.json"

    exit_code = main(
        [
            "benchmark",
            "--methods",
            "baseline_tfidf,keybert,llm",
            "--focus",
            "both",
            "--output",
            str(output_path),
        ],
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["command"] == "benchmark"
    assert payload["methods"] == ["baseline_tfidf", "keybert", "llm"]
    assert payload["outputs"]["llm"]["focus"] == "both"
