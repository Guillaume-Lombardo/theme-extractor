from __future__ import annotations

from theme_extractor.domain import (
    BackendName,
    CommandName,
    ExtractionRunMetadata,
    ExtractMethod,
    OfflinePolicy,
    OutputFocus,
    TopicKeyword,
    TopicResult,
    UnifiedExtractionOutput,
)

_TOPIC_ID = 7


def test_unified_output_defaults_to_topic_first_focus() -> None:
    metadata = ExtractionRunMetadata(
        command=CommandName.EXTRACT,
        method=ExtractMethod.BASELINE_TFIDF,
        offline_policy=OfflinePolicy.STRICT,
    )
    output = UnifiedExtractionOutput(metadata=metadata)

    assert output.focus == "topics"
    assert output.topics == []
    assert output.document_topics is None


def test_topic_result_supports_keywords_and_document_links() -> None:
    metadata = ExtractionRunMetadata(
        command=CommandName.EXTRACT,
        method=ExtractMethod.BERTOPIC,
        offline_policy=OfflinePolicy.STRICT,
        backend=BackendName.ELASTICSEARCH,
    )
    topic = TopicResult(
        topic_id=_TOPIC_ID,
        label="prudential risk",
        keywords=[TopicKeyword(term="risk", score=0.91)],
        document_ids=["doc-1", "doc-2"],
    )
    output = UnifiedExtractionOutput(
        focus=OutputFocus.BOTH,
        topics=[topic],
        document_topics=[],
        metadata=metadata,
    )

    assert output.focus == "both"
    assert len(output.topics) == 1
    assert output.topics[0].topic_id == _TOPIC_ID
    assert output.topics[0].keywords[0].term == "risk"
    assert output.topics[0].document_ids == ["doc-1", "doc-2"]
