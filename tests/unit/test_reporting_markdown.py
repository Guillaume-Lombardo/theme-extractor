from __future__ import annotations

import json

import pytest

from theme_extractor.domain import (
    BackendName,
    BenchmarkOutput,
    CommandName,
    ExtractionRunMetadata,
    ExtractMethod,
    OfflinePolicy,
    OutputFocus,
    TopicKeyword,
    TopicResult,
    UnifiedExtractionOutput,
)
from theme_extractor.reporting.markdown import (
    UnsupportedReportPayloadError,
    load_report_payload,
    render_benchmark_markdown,
    render_extract_markdown,
    render_report_markdown,
)


def _extract_payload() -> UnifiedExtractionOutput:
    return UnifiedExtractionOutput(
        focus=OutputFocus.BOTH,
        topics=[
            TopicResult(
                topic_id=0,
                label="invoice",
                score=1.23,
                keywords=[TopicKeyword(term="invoice", score=0.9)],
                document_ids=["doc-1", "doc-2"],
            ),
        ],
        document_topics=[],
        notes=["sample note"],
        metadata=ExtractionRunMetadata(
            command=CommandName.EXTRACT,
            method=ExtractMethod.KEYBERT,
            offline_policy=OfflinePolicy.STRICT,
            backend=BackendName.ELASTICSEARCH,
            index="theme_extractor",
        ),
    )


def test_render_extract_markdown_contains_core_sections() -> None:
    rendered = render_extract_markdown(_extract_payload(), title="Custom")

    assert "# Custom" in rendered
    assert "## Run Metadata" in rendered
    assert "## Topics" in rendered
    assert "invoice" in rendered
    assert "## Document Associations" in rendered
    assert "## Notes" in rendered


def test_render_benchmark_markdown_contains_comparison_and_details() -> None:
    extract_payload = _extract_payload()
    benchmark_payload = BenchmarkOutput(
        methods=[ExtractMethod.KEYBERT],
        outputs={ExtractMethod.KEYBERT.value: extract_payload},
        comparison={
            "method_count": 1,
            "pairwise_overlap": [],
        },
    )

    rendered = render_benchmark_markdown(benchmark_payload)

    assert "# Theme Extractor Benchmark Report" in rendered
    assert "## Method Comparison" in rendered
    assert "## Strategy Details" in rendered
    assert "### `keybert`" in rendered


def test_load_report_payload_and_render_from_file(tmp_path) -> None:
    payload = _extract_payload().model_dump(mode="json")
    input_path = tmp_path / "extract.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_report_payload(input_path)
    assert isinstance(loaded, UnifiedExtractionOutput)

    rendered = render_report_markdown(input_path)
    assert "Theme Extractor Report" in rendered
    assert "invoice" in rendered


def test_load_report_payload_benchmark_from_file(tmp_path) -> None:
    extract_payload = _extract_payload()
    benchmark_payload = BenchmarkOutput(
        methods=[ExtractMethod.KEYBERT],
        outputs={ExtractMethod.KEYBERT.value: extract_payload},
        comparison={"method_count": 1, "pairwise_overlap": []},
    )
    input_path = tmp_path / "benchmark.json"
    input_path.write_text(
        json.dumps(benchmark_payload.model_dump(mode="json")),
        encoding="utf-8",
    )

    loaded = load_report_payload(input_path)
    assert isinstance(loaded, BenchmarkOutput)

    rendered = render_report_markdown(input_path)
    assert "Theme Extractor Benchmark Report" in rendered
    assert "Method Comparison" in rendered


def test_load_report_payload_evaluation_from_file(tmp_path) -> None:
    input_path = tmp_path / "evaluation.json"
    input_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "command": "evaluate",
                "summary": {
                    "input_count": 1,
                    "extract_count": 0,
                    "benchmark_count": 1,
                },
                "extracts": [],
                "benchmarks": [
                    {
                        "path": "benchmark.json",
                        "metrics": {
                            "method_count": 2,
                            "cross_method_mean_jaccard": 0.1234,
                            "per_method": {
                                "terms": {
                                    "topic_count": 10,
                                    "document_topic_count": 0,
                                    "avg_keywords_per_topic": 1.0,
                                    "topic_coherence_proxy": 1.0,
                                    "inter_topic_diversity": 1.0,
                                    "inter_topic_mean_jaccard": 0.0,
                                },
                            },
                        },
                    },
                ],
            },
        ),
        encoding="utf-8",
    )

    loaded = load_report_payload(input_path)
    assert isinstance(loaded, dict)
    assert loaded["command"] == "evaluate"

    rendered = render_report_markdown(input_path)
    assert "Theme Extractor Evaluation Report" in rendered
    assert "## Benchmark Metrics" in rendered
    assert "## Benchmark Method Details" in rendered
    assert "terms" in rendered
    assert "0.1234" in rendered


def test_load_report_payload_raises_for_unsupported_shape(tmp_path) -> None:
    input_path = tmp_path / "invalid.json"
    input_path.write_text(json.dumps({"hello": "world"}), encoding="utf-8")

    with pytest.raises(UnsupportedReportPayloadError):
        load_report_payload(input_path)
