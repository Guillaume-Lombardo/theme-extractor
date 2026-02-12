from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from theme_extractor.domain import (
    BackendName,
    BertopicClustering,
    BertopicDimReduction,
    CommandName,
    ExtractionRunMetadata,
    ExtractMethod,
    OfflinePolicy,
    OutputFocus,
    UnifiedExtractionOutput,
)
from theme_extractor.extraction import BaselineExtractionConfig
from theme_extractor.extraction.bertopic import (
    BertopicExtractionConfig,
    BertopicRunRequest,
    run_bertopic_method,
)


@dataclass
class _BackendStub:
    search_response: dict[str, Any]
    backend_name: str = "stub"

    def search_documents(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        _ = index
        _ = body
        return self.search_response

    def terms_aggregation(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:  # noqa: PLR6301
        _ = index
        _ = body
        return {}

    def significant_terms_aggregation(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:  # noqa: PLR6301
        _ = index
        _ = body
        return {}

    def significant_text_aggregation(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:  # noqa: PLR6301
        _ = index
        _ = body
        return {}


def _make_output(*, focus: OutputFocus) -> UnifiedExtractionOutput:
    metadata = ExtractionRunMetadata(
        command=CommandName.EXTRACT,
        method=ExtractMethod.BERTOPIC,
        offline_policy=OfflinePolicy.STRICT,
        backend=BackendName.ELASTICSEARCH,
        index="idx",
    )
    return UnifiedExtractionOutput(focus=focus, metadata=metadata)


def test_run_bertopic_builds_topics_and_doc_links() -> None:
    backend = _BackendStub(
        search_response={
            "hits": {
                "hits": [
                    {"_id": "doc-1", "_source": {"content": "invoice tax payment declaration"}},
                    {"_id": "doc-2", "_source": {"content": "invoice due date payment tax"}},
                    {"_id": "doc-3", "_source": {"content": "copropriete reglement assemblée syndic"}},
                    {"_id": "doc-4", "_source": {"content": "syndic copropriete charges assemblée"}},
                ],
            },
        },
    )

    output = run_bertopic_method(
        backend=backend,
        request=BertopicRunRequest(
            index="idx",
            focus=OutputFocus.BOTH,
            baseline_config=BaselineExtractionConfig(top_n=3, search_size=20),
            bertopic_config=BertopicExtractionConfig(
                reduce_dim=BertopicDimReduction.SVD,
                clustering=BertopicClustering.KMEANS,
                min_topic_size=1,
                nr_topics=2,
            ),
        ),
        output=_make_output(focus=OutputFocus.BOTH),
    )

    assert output.topics
    assert output.document_topics
    assert "BERTopic strategy executed." in output.notes


def test_run_bertopic_empty_corpus_sets_empty_doc_topics() -> None:
    backend = _BackendStub(search_response={"hits": {"hits": []}})
    output = run_bertopic_method(
        backend=backend,
        request=BertopicRunRequest(
            index="idx",
            focus=OutputFocus.BOTH,
            baseline_config=BaselineExtractionConfig(),
            bertopic_config=BertopicExtractionConfig(),
        ),
        output=_make_output(focus=OutputFocus.BOTH),
    )
    assert output.topics == []
    assert output.document_topics == []
    assert "BERTopic executed with empty corpus from backend search." in output.notes
