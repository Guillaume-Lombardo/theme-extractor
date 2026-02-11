from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from theme_extractor.domain import (
    BackendName,
    CommandName,
    ExtractionRunMetadata,
    ExtractMethod,
    OfflinePolicy,
    OutputFocus,
    UnifiedExtractionOutput,
)
from theme_extractor.extraction import BaselineExtractionConfig, run_baseline_method
from theme_extractor.extraction.baselines import BaselineRunRequest

_EXPECTED_DOC_TOPICS = 2
_EXPECTED_SIG_TERMS_SCORE = 12.2
_EXPECTED_SIG_TEXT_SCORE = 7.1


@dataclass
class _BackendStub:
    search_response: dict[str, Any]
    terms_response: dict[str, Any]
    sig_terms_response: dict[str, Any]
    sig_text_response: dict[str, Any]
    backend_name: str = "stub"

    def search_documents(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        _ = index
        _ = body
        return self.search_response

    def terms_aggregation(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        _ = index
        _ = body
        return self.terms_response

    def significant_terms_aggregation(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        _ = index
        _ = body
        return self.sig_terms_response

    def significant_text_aggregation(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        _ = index
        _ = body
        return self.sig_text_response


def _make_output(*, method: ExtractMethod, focus: OutputFocus) -> UnifiedExtractionOutput:
    metadata = ExtractionRunMetadata(
        command=CommandName.EXTRACT,
        method=method,
        offline_policy=OfflinePolicy.STRICT,
        backend=BackendName.ELASTICSEARCH,
        index="idx",
    )
    return UnifiedExtractionOutput(focus=focus, metadata=metadata)


def test_run_baseline_tfidf_builds_topic_and_document_links() -> None:
    backend = _BackendStub(
        search_response={
            "hits": {
                "hits": [
                    {"_id": "doc-1", "_source": {"content": "alpha beta"}},
                    {"_id": "doc-2", "_source": {"content": "alpha gamma"}},
                ],
            },
        },
        terms_response={},
        sig_terms_response={},
        sig_text_response={},
    )

    output = run_baseline_method(
        backend=backend,
        request=BaselineRunRequest(
            method=ExtractMethod.BASELINE_TFIDF,
            index="idx",
            focus=OutputFocus.BOTH,
            config=BaselineExtractionConfig(top_n=3),
        ),
        output=_make_output(method=ExtractMethod.BASELINE_TFIDF, focus=OutputFocus.BOTH),
    )

    assert len(output.topics) == 1
    assert output.topics[0].label == "tfidf"
    assert output.topics[0].keywords
    assert output.document_topics is not None
    assert len(output.document_topics) == _EXPECTED_DOC_TOPICS


def test_run_baseline_terms_uses_bucket_mapping() -> None:
    backend = _BackendStub(
        search_response={},
        terms_response={
            "aggregations": {
                "terms": {
                    "buckets": [
                        {"key": "prudential", "doc_count": 4},
                        {"key": "risk", "doc_count": 2},
                    ],
                },
            },
        },
        sig_terms_response={},
        sig_text_response={},
    )

    output = run_baseline_method(
        backend=backend,
        request=BaselineRunRequest(
            method=ExtractMethod.TERMS,
            index="idx",
            focus=OutputFocus.TOPICS,
            config=BaselineExtractionConfig(top_n=2),
        ),
        output=_make_output(method=ExtractMethod.TERMS, focus=OutputFocus.TOPICS),
    )

    assert [topic.label for topic in output.topics] == ["prudential", "risk"]
    assert output.document_topics is None


def test_run_baseline_significant_methods_map_scores() -> None:
    backend = _BackendStub(
        search_response={},
        terms_response={},
        sig_terms_response={
            "aggregations": {
                "themes": {
                    "buckets": [{"key": "liquidity", "doc_count": 3, "score": 12.2}],
                },
            },
        },
        sig_text_response={
            "aggregations": {
                "themes": {
                    "buckets": [{"key": "stress", "doc_count": 5, "score": 7.1}],
                },
            },
        },
    )

    sig_terms_output = run_baseline_method(
        backend=backend,
        request=BaselineRunRequest(
            method=ExtractMethod.SIGNIFICANT_TERMS,
            index="idx",
            focus=OutputFocus.TOPICS,
            config=BaselineExtractionConfig(top_n=2),
        ),
        output=_make_output(method=ExtractMethod.SIGNIFICANT_TERMS, focus=OutputFocus.TOPICS),
    )
    sig_text_output = run_baseline_method(
        backend=backend,
        request=BaselineRunRequest(
            method=ExtractMethod.SIGNIFICANT_TEXT,
            index="idx",
            focus=OutputFocus.TOPICS,
            config=BaselineExtractionConfig(top_n=2),
        ),
        output=_make_output(method=ExtractMethod.SIGNIFICANT_TEXT, focus=OutputFocus.TOPICS),
    )

    assert sig_terms_output.topics[0].label == "liquidity"
    assert sig_terms_output.topics[0].score == _EXPECTED_SIG_TERMS_SCORE
    assert sig_text_output.topics[0].label == "stress"
    assert sig_text_output.topics[0].score == _EXPECTED_SIG_TEXT_SCORE


def test_run_baseline_method_raises_for_non_baseline_method() -> None:
    backend = _BackendStub(
        search_response={},
        terms_response={},
        sig_terms_response={},
        sig_text_response={},
    )

    with pytest.raises(ValueError, match="Unsupported baseline extraction method"):
        run_baseline_method(
            backend=backend,
            request=BaselineRunRequest(
                method=ExtractMethod.LLM,
                index="idx",
                focus=OutputFocus.TOPICS,
                config=BaselineExtractionConfig(top_n=2),
            ),
            output=_make_output(method=ExtractMethod.LLM, focus=OutputFocus.TOPICS),
        )
