from __future__ import annotations

import json

from theme_extractor.cli import main

_EXPECTED_METHOD_COUNT = 7


def test_benchmark_supports_combined_methods_focus_and_backend(capsys) -> None:
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


def test_extract_supports_strict_offline_policy_and_elasticsearch_backend(capsys) -> None:
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
