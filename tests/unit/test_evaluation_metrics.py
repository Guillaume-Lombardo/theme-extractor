from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from theme_extractor.domain import BenchmarkOutput, UnifiedExtractionOutput
from theme_extractor.evaluation.metrics import (
    UnsupportedEvaluationPayloadError,
    evaluate_benchmark_output,
    evaluate_extract_output,
    evaluate_payload_files,
)

_EXPECTED_TOPIC_COUNT = 2
_EXPECTED_METHOD_COUNT = 2
_EXPECTED_INPUT_COUNT = 2


def _extract_payload(method: str = "keybert") -> UnifiedExtractionOutput:
    payload = {
        "schema_version": "1.0",
        "focus": "topics",
        "topics": [
            {
                "topic_id": 0,
                "label": "invoice",
                "score": 1.0,
                "keywords": [
                    {"term": "invoice", "score": 0.9},
                    {"term": "payment", "score": 0.8},
                    {"term": "tax", "score": 0.4},
                ],
                "document_ids": ["doc-1", "doc-2"],
                "representative_documents": [],
                "summary": None,
            },
            {
                "topic_id": 1,
                "label": "legal",
                "score": 0.8,
                "keywords": [
                    {"term": "contract", "score": 0.7},
                    {"term": "clause", "score": 0.5},
                ],
                "document_ids": ["doc-3"],
                "representative_documents": [],
                "summary": None,
            },
        ],
        "document_topics": [],
        "notes": [],
        "metadata": {
            "run_id": "run-1",
            "generated_at": datetime.now(tz=UTC).isoformat(),
            "command": "extract",
            "method": method,
            "offline_policy": "strict",
            "backend": "elasticsearch",
            "index": "theme_extractor",
        },
    }
    return UnifiedExtractionOutput.model_validate(payload)


def _benchmark_payload() -> BenchmarkOutput:
    extract_keybert = _extract_payload("keybert")
    extract_tfidf = UnifiedExtractionOutput.model_validate(
        {
            **_extract_payload("baseline_tfidf").model_dump(mode="json"),
            "topics": [
                {
                    "topic_id": 0,
                    "label": "finance",
                    "score": 1.1,
                    "keywords": [
                        {"term": "invoice", "score": 0.6},
                        {"term": "budget", "score": 0.5},
                    ],
                    "document_ids": ["doc-1"],
                    "representative_documents": [],
                    "summary": None,
                },
            ],
        },
    )

    return BenchmarkOutput(
        methods=[extract_keybert.metadata.method, extract_tfidf.metadata.method],
        outputs={
            extract_keybert.metadata.method.value: extract_keybert,
            extract_tfidf.metadata.method.value: extract_tfidf,
        },
        comparison={"method_count": 2, "pairwise_overlap": []},
    )


def test_evaluate_extract_output_returns_expected_metrics() -> None:
    metrics = evaluate_extract_output(_extract_payload())

    assert metrics["method"] == "keybert"
    assert metrics["topic_count"] == _EXPECTED_TOPIC_COUNT
    assert metrics["avg_keywords_per_topic"] is not None
    assert metrics["topic_coherence_proxy"] is not None


def test_evaluate_benchmark_output_returns_per_method_metrics() -> None:
    metrics = evaluate_benchmark_output(_benchmark_payload())

    assert metrics["method_count"] == _EXPECTED_METHOD_COUNT
    assert "keybert" in metrics["per_method"]
    assert "baseline_tfidf" in metrics["per_method"]


def test_evaluate_payload_files_handles_extract_and_benchmark(tmp_path) -> None:
    extract_path = tmp_path / "extract.json"
    benchmark_path = tmp_path / "benchmark.json"
    extract_path.write_text(json.dumps(_extract_payload().model_dump(mode="json")), encoding="utf-8")
    benchmark_path.write_text(json.dumps(_benchmark_payload().model_dump(mode="json")), encoding="utf-8")

    metrics = evaluate_payload_files([extract_path, benchmark_path])

    assert metrics["summary"]["input_count"] == _EXPECTED_INPUT_COUNT
    assert metrics["summary"]["extract_count"] == 1
    assert metrics["summary"]["benchmark_count"] == 1


def test_evaluate_payload_files_raises_for_invalid_payload(tmp_path) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps({"hello": "world"}), encoding="utf-8")

    with pytest.raises(UnsupportedEvaluationPayloadError):
        evaluate_payload_files([invalid])
