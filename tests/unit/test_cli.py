from __future__ import annotations

import json
import os
from datetime import UTC, datetime

import pytest

from theme_extractor.cli import main
from theme_extractor.errors import UnsupportedMethodError

_PARSER_ERROR_EXIT_CODE = 2
_EXPECTED_BENCHMARK_METHOD_COUNT = 2
_PROXY_URL = "http://proxy.local:8080"
_PROXY_ENV_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")
_EXPECTED_OCR_DPI = 300
_EXPECTED_OCR_MIN_CHARS = 12


class _BackendUnavailableError(RuntimeError):
    """Represent one backend unavailability error in tests."""


def _backend_stub(**_kwargs) -> object:
    class _Stub:
        backend_name = "stub"

        def search_documents(  # noqa: PLR6301
            self,
            *,
            index: str,
            body: dict[str, object],
        ) -> dict[str, object]:
            _ = index
            _ = body
            return {
                "hits": {
                    "hits": [
                        {"_id": "doc-1", "_source": {"content": "invoice payment tax"}},
                        {"_id": "doc-2", "_source": {"content": "tax declaration invoice"}},
                    ],
                },
            }

        def terms_aggregation(  # noqa: PLR6301
            self,
            *,
            index: str,
            body: dict[str, object],
        ) -> dict[str, object]:
            _ = index
            _ = body
            return {"aggregations": {"terms": {"buckets": []}}}

        def significant_terms_aggregation(  # noqa: PLR6301
            self,
            *,
            index: str,
            body: dict[str, object],
        ) -> dict[str, object]:
            _ = index
            _ = body
            return {"aggregations": {"themes": {"buckets": []}}}

        def significant_text_aggregation(  # noqa: PLR6301
            self,
            *,
            index: str,
            body: dict[str, object],
        ) -> dict[str, object]:
            _ = index
            _ = body
            return {"aggregations": {"themes": {"buckets": []}}}

    return _Stub()


def test_main_without_subcommand_returns_1(capsys) -> None:
    exit_code = main([])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "usage:" in captured.out


def test_extract_to_stdout_returns_normalized_topics_payload(capsys, monkeypatch) -> None:
    monkeypatch.setattr("theme_extractor.cli.command_handlers.build_search_backend", _backend_stub)
    exit_code = main(["extract", "--method", "keybert", "--focus", "topics"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["schema_version"] == "1.0"
    assert payload["focus"] == "topics"
    assert payload["topics"]
    assert payload["document_topics"] is None
    assert payload["metadata"]["command"] == "extract"
    assert payload["metadata"]["method"] == "keybert"


def test_extract_with_document_focus_emits_document_topics(capsys, monkeypatch) -> None:
    monkeypatch.setattr("theme_extractor.cli.command_handlers.build_search_backend", _backend_stub)
    exit_code = main(
        [
            "extract",
            "--method",
            "bertopic",
            "--focus",
            "documents",
            "--bertopic-min-topic-size",
            "1",
        ],
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["focus"] == "documents"
    assert payload["document_topics"]


def test_extract_bertopic_accepts_embedding_cache_flags(capsys, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("theme_extractor.cli.command_handlers.build_search_backend", _backend_stub)
    cache_dir = tmp_path / "emb-cache"
    exit_code = main(
        [
            "extract",
            "--method",
            "bertopic",
            "--focus",
            "topics",
            "--bertopic-min-topic-size",
            "1",
            "--bertopic-embedding-cache-enabled",
            "--bertopic-embedding-cache-dir",
            str(cache_dir),
            "--bertopic-embedding-cache-version",
            "v2",
            "--no-bertopic-use-embeddings",
        ],
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["metadata"]["method"] == "bertopic"


def test_ingest_to_file_writes_json(tmp_path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("Bonjour le monde\\nPage 1/1\\nBonjour le monde\\n", encoding="utf-8")
    output_path = tmp_path / "ingest.json"

    exit_code = main(
        [
            "ingest",
            "--input",
            str(sample),
            "--offline-policy",
            "strict",
            "--output",
            str(output_path),
        ],
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["command"] == "ingest"
    assert payload["processed_documents"] == 1
    assert payload["streaming_mode"] is True
    assert payload["runtime"]["offline_policy"] == "strict"
    assert payload["documents"][0]["path"] == str(sample.resolve())
    assert payload["runtime"]["reset_index"]["requested"] is False
    assert payload["runtime"]["reset_index"]["applied"] is False


def test_ingest_reset_index_triggers_backend_reset(tmp_path, capsys, monkeypatch) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("Bonjour le monde", encoding="utf-8")
    captured: dict[str, str] = {}

    def _fake_reset_backend_index(*, backend_url: str, index: str) -> None:
        captured["backend_url"] = backend_url
        captured["index"] = index

    monkeypatch.setattr("theme_extractor.cli.command_handlers.reset_backend_index", _fake_reset_backend_index)

    exit_code = main(
        [
            "ingest",
            "--input",
            str(sample),
            "--backend-url",
            "http://localhost:9201",
            "--index",
            "theme_extractor_test",
            "--reset-index",
            "--output",
            "-",
        ],
    )

    assert exit_code == 0
    assert captured == {
        "backend_url": "http://localhost:9201",
        "index": "theme_extractor_test",
    }
    payload = json.loads(capsys.readouterr().out)
    assert payload["runtime"]["reset_index"]["requested"] is True
    assert payload["runtime"]["reset_index"]["applied"] is True


def test_ingest_index_backend_triggers_bulk_indexing(tmp_path, capsys, monkeypatch) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("Bonjour le monde", encoding="utf-8")
    captured: dict[str, object] = {}

    def _fake_build_ingest_index_documents(
        *,
        args: object,
        result_payload: dict[str, object],
        stopwords: set[str],
    ) -> list[dict[str, object]]:
        _ = args
        captured["stopwords"] = stopwords
        documents = result_payload.get("documents")
        captured["documents_count"] = len(documents) if isinstance(documents, list) else 0
        return [{"_id": "doc-1", "_source": {"content": "bonjour monde"}}]

    def _fake_bulk_index_documents(*, backend_url: str, index: str, docs):
        captured["backend_url"] = backend_url
        captured["index"] = index
        captured["bulk_docs"] = docs

    monkeypatch.setattr(
        "theme_extractor.cli.command_handlers.build_ingest_index_documents",
        _fake_build_ingest_index_documents,
    )
    monkeypatch.setattr(
        "theme_extractor.cli.command_handlers.bulk_index_documents", _fake_bulk_index_documents
    )

    exit_code = main(
        [
            "ingest",
            "--input",
            str(sample),
            "--index-backend",
            "--backend-url",
            "http://localhost:9201",
            "--index",
            "theme_extractor_test",
            "--output",
            "-",
        ],
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["runtime"]["index_backend"]["requested"] is True
    assert payload["runtime"]["index_backend"]["applied"] is True
    assert payload["runtime"]["index_backend"]["indexed_documents"] == 1
    assert captured["backend_url"] == "http://localhost:9201"
    assert captured["index"] == "theme_extractor_test"
    assert captured["documents_count"] == 1
    assert captured["bulk_docs"] == [{"_id": "doc-1", "_source": {"content": "bonjour monde"}}]


def test_ingest_accepts_none_cleaning_option(tmp_path, capsys) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("Résumé !!!", encoding="utf-8")

    exit_code = main(
        [
            "ingest",
            "--input",
            str(sample),
            "--cleaning-options",
            "none",
        ],
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["cleaning_options"] == "none"


def test_ingest_accepts_pdf_ocr_flags(tmp_path, capsys) -> None:
    sample = tmp_path / "sample.pdf"
    sample.write_text("placeholder", encoding="utf-8")
    tessdata = tmp_path / "tessdata"
    tessdata.mkdir()

    exit_code = main(
        [
            "ingest",
            "--input",
            str(sample),
            "--pdf-ocr-fallback",
            "--pdf-ocr-languages",
            "fra+eng",
            "--pdf-ocr-dpi",
            str(_EXPECTED_OCR_DPI),
            "--pdf-ocr-min-chars",
            str(_EXPECTED_OCR_MIN_CHARS),
            "--pdf-ocr-tessdata",
            str(tessdata),
        ],
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["pdf_ocr_fallback"] is True
    assert payload["pdf_ocr_languages"] == "fra+eng"
    assert payload["pdf_ocr_dpi"] == _EXPECTED_OCR_DPI
    assert payload["pdf_ocr_min_chars"] == _EXPECTED_OCR_MIN_CHARS
    assert payload["pdf_ocr_tessdata"] == str(tessdata)


def test_ingest_accepts_msg_extraction_flags(tmp_path, capsys) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("hello", encoding="utf-8")

    exit_code = main(
        [
            "ingest",
            "--input",
            str(sample),
            "--no-msg-include-metadata",
            "--msg-attachments-policy",
            "text",
        ],
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["msg_include_metadata"] is False
    assert payload["msg_attachments_policy"] == "text"


def test_benchmark_rejects_unknown_method() -> None:
    with pytest.raises(UnsupportedMethodError, match="Unsupported extraction method: unknown"):
        main(["benchmark", "--methods", "unknown"])


def test_benchmark_rejects_empty_methods_list() -> None:
    with pytest.raises(ValueError, match="At least one extraction method"):
        main(["benchmark", "--methods", ",,,"])


def test_benchmark_ignores_empty_method_tokens(capsys, monkeypatch) -> None:
    monkeypatch.setattr("theme_extractor.cli.command_handlers.build_search_backend", _backend_stub)
    exit_code = main(["benchmark", "--methods", ",,llm,,", "--focus", "topics"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["methods"] == ["llm"]


def test_benchmark_deduplicates_methods_and_outputs_json(capsys, monkeypatch) -> None:
    monkeypatch.setattr("theme_extractor.cli.command_handlers.build_search_backend", _backend_stub)
    exit_code = main(
        [
            "benchmark",
            "--methods",
            "keybert,keybert,llm",
            "--focus",
            "both",
        ],
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["command"] == "benchmark"
    assert payload["methods"] == ["keybert", "llm"]
    assert set(payload["outputs"].keys()) == {"keybert", "llm"}
    assert "comparison" in payload
    assert payload["comparison"]["method_count"] == _EXPECTED_BENCHMARK_METHOD_COUNT
    assert payload["outputs"]["keybert"]["focus"] == "both"
    assert payload["outputs"]["keybert"]["document_topics"]


def test_benchmark_with_topic_focus_keeps_document_topics_none(capsys, monkeypatch) -> None:
    monkeypatch.setattr("theme_extractor.cli.command_handlers.build_search_backend", _backend_stub)
    exit_code = main(["benchmark", "--methods", "llm", "--focus", "topics"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["outputs"]["llm"]["document_topics"] is None


def test_main_returns_parser_error_exit_code_for_invalid_choice() -> None:
    exit_code = main(["extract", "--method", "invalid-choice"])
    assert exit_code == _PARSER_ERROR_EXIT_CODE


def test_doctor_returns_diagnostics_payload(capsys) -> None:
    exit_code = main(["doctor", "--output", "-"])
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "doctor"
    assert payload["schema_version"] == "1.0"
    assert "optional_dependencies" in payload["checks"]
    assert "local_models" in payload["checks"]
    assert payload["checks"]["backend_connectivity"]["checked"] is False


def test_doctor_backend_check_reports_error(monkeypatch, capsys) -> None:
    class _FailingBackend:
        backend_name = "stub"

        @staticmethod
        def search_documents(*, index: str, body: dict[str, object]) -> dict[str, object]:
            _ = index
            _ = body
            raise _BackendUnavailableError

    monkeypatch.setattr(
        "theme_extractor.cli.command_handlers.build_search_backend", lambda **_kwargs: _FailingBackend()
    )

    exit_code = main(["doctor", "--check-backend", "--output", "-"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
    assert payload["checks"]["backend_connectivity"]["checked"] is True
    assert payload["checks"]["backend_connectivity"]["ok"] is False
    assert "_BackendUnavailableError" in payload["checks"]["backend_connectivity"]["error"]


def test_report_reads_extract_json_and_emits_markdown(tmp_path, capsys) -> None:
    input_path = tmp_path / "extract.json"
    payload = {
        "schema_version": "1.0",
        "focus": "topics",
        "topics": [
            {
                "topic_id": 0,
                "label": "invoice",
                "score": 1.0,
                "keywords": [{"term": "invoice", "score": 1.0}],
                "document_ids": ["doc-1"],
                "representative_documents": [],
                "summary": None,
            },
        ],
        "document_topics": None,
        "notes": ["note"],
        "metadata": {
            "run_id": "run-1",
            "generated_at": datetime.now(tz=UTC).isoformat(),
            "command": "extract",
            "method": "keybert",
            "offline_policy": "strict",
            "backend": "elasticsearch",
            "index": "theme_extractor",
        },
    }
    input_path.write_text(json.dumps(payload), encoding="utf-8")

    exit_code = main(["report", "--input", str(input_path), "--output", "-"])
    assert exit_code == 0
    rendered = capsys.readouterr().out
    assert "# Theme Extractor Report" in rendered
    assert "invoice" in rendered


def test_evaluate_reads_extract_json_and_emits_metrics(tmp_path, capsys) -> None:
    input_path = tmp_path / "extract.json"
    payload = {
        "schema_version": "1.0",
        "focus": "topics",
        "topics": [
            {
                "topic_id": 0,
                "label": "invoice",
                "score": 1.0,
                "keywords": [
                    {"term": "invoice", "score": 1.0},
                    {"term": "tax", "score": 0.5},
                ],
                "document_ids": ["doc-1"],
                "representative_documents": [],
                "summary": None,
            },
        ],
        "document_topics": None,
        "notes": ["note"],
        "metadata": {
            "run_id": "run-1",
            "generated_at": datetime.now(tz=UTC).isoformat(),
            "command": "extract",
            "method": "keybert",
            "offline_policy": "strict",
            "backend": "elasticsearch",
            "index": "theme_extractor",
        },
    }
    input_path.write_text(json.dumps(payload), encoding="utf-8")

    exit_code = main(["evaluate", "--input", str(input_path), "--output", "-"])
    assert exit_code == 0
    metrics = json.loads(capsys.readouterr().out)
    assert metrics["command"] == "evaluate"
    assert metrics["summary"]["extract_count"] == 1


def test_main_applies_proxy_environment_from_flag(tmp_path, monkeypatch) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("Bonjour le monde", encoding="utf-8")

    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.delenv("http_proxy", raising=False)
    monkeypatch.delenv("https_proxy", raising=False)
    original_proxy_values = {key: os.environ.get(key) for key in _PROXY_ENV_KEYS}

    try:
        exit_code = main(
            [
                "ingest",
                "--input",
                str(sample),
                "--proxy-url",
                _PROXY_URL,
                "--output",
                "-",
            ],
        )

        assert exit_code == 0
        assert os.environ["HTTP_PROXY"] == _PROXY_URL
        assert os.environ["HTTPS_PROXY"] == _PROXY_URL
        assert os.environ["http_proxy"] == _PROXY_URL
        assert os.environ["https_proxy"] == _PROXY_URL
    finally:
        for key, value in original_proxy_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_extract_uses_proxy_url_default_from_environment(monkeypatch, capsys) -> None:
    monkeypatch.setattr("theme_extractor.cli.command_handlers.build_search_backend", _backend_stub)
    monkeypatch.setenv("THEME_EXTRACTOR_PROXY_URL", _PROXY_URL)
    original_proxy_values = {key: os.environ.get(key) for key in _PROXY_ENV_KEYS}

    try:
        exit_code = main(["extract", "--method", "terms", "--focus", "topics"])
        assert exit_code == 0
        _ = json.loads(capsys.readouterr().out)
        assert os.environ["HTTP_PROXY"] == _PROXY_URL
        assert os.environ["HTTPS_PROXY"] == _PROXY_URL
        assert os.environ["http_proxy"] == _PROXY_URL
        assert os.environ["https_proxy"] == _PROXY_URL
    finally:
        for key, value in original_proxy_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
