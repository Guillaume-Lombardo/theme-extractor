from __future__ import annotations

import json

import pytest

from theme_extractor.cli import main
from theme_extractor.errors import UnsupportedMethodError

_PARSER_ERROR_EXIT_CODE = 2


def test_main_without_subcommand_returns_1(capsys) -> None:
    exit_code = main([])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "usage:" in captured.out


def test_extract_to_stdout_returns_normalized_topics_payload(capsys) -> None:
    exit_code = main(["extract", "--method", "keybert", "--focus", "topics"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["schema_version"] == "1.0"
    assert payload["focus"] == "topics"
    assert payload["topics"] == []
    assert payload["document_topics"] is None
    assert payload["metadata"]["command"] == "extract"
    assert payload["metadata"]["method"] == "keybert"


def test_extract_with_document_focus_emits_document_topics(capsys) -> None:
    exit_code = main(["extract", "--method", "bertopic", "--focus", "documents"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["focus"] == "documents"
    assert payload["document_topics"] == []


def test_ingest_to_file_writes_json(tmp_path) -> None:
    output_path = tmp_path / "ingest.json"

    exit_code = main(
        [
            "ingest",
            "--input",
            str(tmp_path),
            "--offline-policy",
            "strict",
            "--output",
            str(output_path),
        ],
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["command"] == "ingest"
    assert payload["status"] == "planned"
    assert payload["config"]["offline_policy"] == "strict"


def test_benchmark_rejects_unknown_method() -> None:
    with pytest.raises(UnsupportedMethodError, match="Unsupported extraction method: unknown"):
        main(["benchmark", "--methods", "unknown"])


def test_benchmark_rejects_empty_methods_list() -> None:
    with pytest.raises(ValueError, match="At least one extraction method"):
        main(["benchmark", "--methods", ",,,"])


def test_benchmark_ignores_empty_method_tokens(capsys) -> None:
    exit_code = main(["benchmark", "--methods", ",,llm,,", "--focus", "topics"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["methods"] == ["llm"]


def test_benchmark_deduplicates_methods_and_outputs_json(capsys) -> None:
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
    assert payload["outputs"]["keybert"]["focus"] == "both"
    assert payload["outputs"]["keybert"]["document_topics"] == []


def test_benchmark_with_topic_focus_keeps_document_topics_none(capsys) -> None:
    exit_code = main(["benchmark", "--methods", "llm", "--focus", "topics"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["outputs"]["llm"]["document_topics"] is None


def test_main_returns_parser_error_exit_code_for_invalid_choice() -> None:
    exit_code = main(["extract", "--method", "invalid-choice"])
    assert exit_code == _PARSER_ERROR_EXIT_CODE
