from __future__ import annotations

import json

from theme_extractor.cli import main


def test_ingest_end2end_generates_json_output_file(tmp_path) -> None:
    output_path = tmp_path / "ingest-e2e.json"

    exit_code = main(
        [
            "ingest",
            "--input",
            str(tmp_path),
            "--output",
            str(output_path),
        ],
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["command"] == "ingest"
    assert payload["config"]["input"] == str(tmp_path)


def test_extract_end2end_generates_json_output_file(tmp_path) -> None:
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


def test_benchmark_end2end_generates_json_output_file(tmp_path) -> None:
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
