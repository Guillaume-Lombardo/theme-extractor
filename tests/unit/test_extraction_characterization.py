from __future__ import annotations

from theme_extractor.domain import (
    BackendName,
    CommandName,
    DocumentTopicLink,
    ExtractionRunMetadata,
    ExtractMethod,
    OfflinePolicy,
    OutputFocus,
    TopicKeyword,
    TopicResult,
    UnifiedExtractionOutput,
)
from theme_extractor.extraction.characterization import build_benchmark_comparison, characterize_output

_EXPECTED_REPRESENTATIVE_DOCUMENTS = 3
_EXPECTED_METHOD_COUNT = 2


def _metadata_for(method: ExtractMethod) -> ExtractionRunMetadata:
    return ExtractionRunMetadata(
        command=CommandName.EXTRACT,
        method=method,
        offline_policy=OfflinePolicy.STRICT,
        backend=BackendName.ELASTICSEARCH,
        index="idx",
    )


def test_characterize_output_enriches_label_summary_and_representative_documents() -> None:
    output = UnifiedExtractionOutput(
        focus=OutputFocus.BOTH,
        topics=[
            TopicResult(
                topic_id=0,
                label="tfidf",
                keywords=[
                    TopicKeyword(term="facture", score=0.9),
                    TopicKeyword(term="paiement", score=0.8),
                ],
                document_ids=["doc-1", "doc-2", "doc-3"],
            ),
        ],
        document_topics=[
            DocumentTopicLink(document_id="doc-1", topic_id=0, rank=1),
            DocumentTopicLink(document_id="doc-2", topic_id=0, rank=2),
        ],
        metadata=_metadata_for(ExtractMethod.BASELINE_TFIDF),
    )

    enriched = characterize_output(output)

    assert enriched.topics[0].label == "facture / paiement"
    assert enriched.topics[0].summary is not None
    assert len(enriched.topics[0].representative_documents) == _EXPECTED_REPRESENTATIVE_DOCUMENTS
    assert enriched.topics[0].representative_documents[0].document_id == "doc-1"
    assert "Topic characterization layer applied" in "\n".join(enriched.notes)


def test_build_benchmark_comparison_emits_pairwise_overlap() -> None:
    first = UnifiedExtractionOutput(
        focus=OutputFocus.TOPICS,
        topics=[
            TopicResult(
                topic_id=0,
                label="a",
                keywords=[TopicKeyword(term="alpha"), TopicKeyword(term="beta")],
            ),
        ],
        metadata=_metadata_for(ExtractMethod.KEYBERT),
    )
    second = UnifiedExtractionOutput(
        focus=OutputFocus.TOPICS,
        topics=[
            TopicResult(
                topic_id=0,
                label="b",
                keywords=[TopicKeyword(term="beta"), TopicKeyword(term="gamma")],
            ),
        ],
        metadata=_metadata_for(ExtractMethod.LLM),
    )

    comparison = build_benchmark_comparison({"keybert": first, "llm": second})

    assert comparison["method_count"] == _EXPECTED_METHOD_COUNT
    assert len(comparison["pairwise_overlap"]) == 1
    overlap = comparison["pairwise_overlap"][0]
    assert overlap["overlap_count"] == 1
    assert overlap["overlap_terms_preview"] == ["beta"]
