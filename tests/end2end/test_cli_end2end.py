from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from theme_extractor.cli import main
from theme_extractor.extraction import llm as llm_mod


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


@dataclass
class _EmptyBackendStub:
    backend_name: str = "stub-empty"

    def search_documents(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:  # noqa: PLR6301
        _ = index
        _ = body
        return {"hits": {"hits": []}}

    def terms_aggregation(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:  # noqa: PLR6301
        _ = index
        _ = body
        return {"aggregations": {"terms": {"buckets": []}}}

    def significant_terms_aggregation(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:  # noqa: PLR6301
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


def test_extract_bertopic_end2end_generates_json_output_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("theme_extractor.cli.build_search_backend", lambda **_kwargs: _BackendStub())

    output_path = tmp_path / "extract-bertopic-e2e.json"
    exit_code = main(
        [
            "extract",
            "--method",
            "bertopic",
            "--focus",
            "both",
            "--bertopic-min-topic-size",
            "1",
            "--bertopic-nr-topics",
            "2",
            "--output",
            str(output_path),
        ],
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["method"] == "bertopic"
    assert payload["topics"]
    assert payload["document_topics"]


def test_extract_bertopic_end2end_empty_corpus_branch(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("theme_extractor.cli.build_search_backend", lambda **_kwargs: _EmptyBackendStub())

    output_path = tmp_path / "extract-bertopic-empty-e2e.json"
    exit_code = main(
        [
            "extract",
            "--method",
            "bertopic",
            "--focus",
            "both",
            "--output",
            str(output_path),
        ],
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["topics"] == []
    assert payload["document_topics"] == []


def test_extract_bertopic_end2end_fallback_notes(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("theme_extractor.cli.build_search_backend", lambda **_kwargs: _BackendStub())

    output_path = tmp_path / "extract-bertopic-fallback-e2e.json"
    exit_code = main(
        [
            "extract",
            "--method",
            "bertopic",
            "--focus",
            "topics",
            "--bertopic-use-embeddings",
            "--bertopic-dim-reduction",
            "umap",
            "--bertopic-clustering",
            "hdbscan",
            "--bertopic-min-topic-size",
            "1",
            "--output",
            str(output_path),
        ],
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    notes = "\n".join(payload["notes"])
    assert "fell back" in notes or "fallback" in notes


def test_extract_llm_end2end_strict_offline(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("theme_extractor.cli.build_search_backend", lambda **_kwargs: _BackendStub())

    output_path = tmp_path / "extract-llm-strict-e2e.json"
    exit_code = main(
        [
            "extract",
            "--method",
            "llm",
            "--focus",
            "both",
            "--offline-policy",
            "strict",
            "--output",
            str(output_path),
        ],
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["method"] == "llm"
    assert payload["topics"]
    assert payload["document_topics"]
    assert "strict offline mode" in "\n".join(payload["notes"])


def test_extract_llm_end2end_provider_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("theme_extractor.cli.build_search_backend", lambda **_kwargs: _BackendStub())
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    monkeypatch.setattr(
        "theme_extractor.extraction.llm._extract_keywords_with_openai",
        lambda **_kwargs: [("alpha", 0.9), ("beta", 0.8)],
    )

    output_path = tmp_path / "extract-llm-provider-e2e.json"
    exit_code = main(
        [
            "extract",
            "--method",
            "llm",
            "--focus",
            "topics",
            "--offline-policy",
            "preload_or_first_run",
            "--output",
            str(output_path),
        ],
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["topics"]
    assert payload["document_topics"] is None
    assert "provider response" in "\n".join(payload["notes"])


def test_doctor_end2end_generates_json_output_file(tmp_path) -> None:
    output_path = tmp_path / "doctor-e2e.json"
    exit_code = main(
        [
            "doctor",
            "--output",
            str(output_path),
        ],
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0"
    assert payload["command"] == "doctor"
    assert "checks" in payload


def test_extract_llm_end2end_empty_corpus(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("theme_extractor.cli.build_search_backend", lambda **_kwargs: _EmptyBackendStub())

    output_path = tmp_path / "extract-llm-empty-e2e.json"
    exit_code = main(
        [
            "extract",
            "--method",
            "llm",
            "--focus",
            "topics",
            "--output",
            str(output_path),
        ],
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["topics"] == []
    assert payload["document_topics"] is None
    assert "empty corpus" in "\n".join(payload["notes"])


def test_extract_llm_end2end_empty_corpus_with_document_focus(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("theme_extractor.cli.build_search_backend", lambda **_kwargs: _EmptyBackendStub())

    output_path = tmp_path / "extract-llm-empty-doc-focus-e2e.json"
    exit_code = main(
        [
            "extract",
            "--method",
            "llm",
            "--focus",
            "both",
            "--output",
            str(output_path),
        ],
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["topics"] == []
    assert payload["document_topics"] == []


def test_extract_llm_end2end_preload_without_credentials_fallback(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("theme_extractor.cli.build_search_backend", lambda **_kwargs: _BackendStub())
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    output_path = tmp_path / "extract-llm-no-cred-e2e.json"
    exit_code = main(
        [
            "extract",
            "--method",
            "llm",
            "--focus",
            "topics",
            "--offline-policy",
            "preload_or_first_run",
            "--output",
            str(output_path),
        ],
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["topics"]
    assert "credentials were not provided" in "\n".join(payload["notes"])


def test_llm_internal_openai_parser_path(monkeypatch) -> None:
    class _Message:
        content = '{"keywords":[{"term":"fiscalite","score":0.99}]}'

    class _Choice:
        message = _Message()

    class _Completions:
        @staticmethod
        def create(**_kwargs) -> object:
            class _Response:
                def __init__(self) -> None:
                    self.choices = [_Choice()]

            return _Response()

    class _Chat:
        completions = _Completions()

    class _Client:
        def __init__(self, **_kwargs) -> None:
            self.chat = _Chat()

    class _OpenAiModule:
        OpenAI = _Client

    monkeypatch.setattr("theme_extractor.extraction.llm.import_module", lambda _name: _OpenAiModule())
    output = llm_mod._extract_keywords_with_openai(
        corpus_text="fiscalite impot taxe revenu",
        config=llm_mod.LlmExtractionConfig(),
        top_n=3,
        api_key="fake-key",
    )
    assert output == [("fiscalite", 0.99)]


def test_report_end2end_from_extract_output(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("theme_extractor.cli.build_search_backend", lambda **_kwargs: _BackendStub())

    extract_path = tmp_path / "extract.json"
    report_path = tmp_path / "report.md"

    extract_exit_code = main(
        [
            "extract",
            "--method",
            "keybert",
            "--focus",
            "topics",
            "--output",
            str(extract_path),
        ],
    )
    assert extract_exit_code == 0

    report_exit_code = main(
        [
            "report",
            "--input",
            str(extract_path),
            "--output",
            str(report_path),
        ],
    )
    assert report_exit_code == 0

    report_content = report_path.read_text(encoding="utf-8")
    assert "# Theme Extractor Report" in report_content
    assert "## Topics" in report_content


def test_evaluate_end2end_from_extract_output(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("theme_extractor.cli.build_search_backend", lambda **_kwargs: _BackendStub())

    extract_path = tmp_path / "extract.json"
    evaluation_path = tmp_path / "evaluation.json"

    extract_exit_code = main(
        [
            "extract",
            "--method",
            "keybert",
            "--focus",
            "topics",
            "--output",
            str(extract_path),
        ],
    )
    assert extract_exit_code == 0

    evaluate_exit_code = main(
        [
            "evaluate",
            "--input",
            str(extract_path),
            "--output",
            str(evaluation_path),
        ],
    )
    assert evaluate_exit_code == 0

    payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
    assert payload["command"] == "evaluate"
    assert payload["summary"]["extract_count"] == 1
