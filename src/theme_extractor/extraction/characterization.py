"""Topic characterization and benchmark comparison helpers."""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING, Any

from theme_extractor.domain import (
    OutputFocus,
    TopicRepresentativeDocument,
    UnifiedExtractionOutput,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


_GENERIC_TOPIC_LABELS = {"tfidf", "keybert", "llm"}
_MAX_LABEL_KEYWORDS = 3
_MAX_REPRESENTATIVE_DOCUMENTS = 3
_MAX_SUMMARY_KEYWORDS = 5


def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    """Return deduplicated values while preserving their original order.

    Args:
        values (Iterable[str]): Input string values.

    Returns:
        list[str]: Deduplicated values in insertion order.

    """
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _infer_label(existing_label: str | None, keyword_terms: list[str], topic_id: int) -> str:
    """Infer a topic label from keywords when current label is generic/missing.

    Args:
        existing_label (str | None): Existing topic label.
        keyword_terms (list[str]): Ranked keyword terms.
        topic_id (int): Topic identifier.

    Returns:
        str: Inferred or preserved label.

    """
    normalized_existing = (existing_label or "").strip().lower()
    if existing_label and normalized_existing not in _GENERIC_TOPIC_LABELS:
        return existing_label
    if keyword_terms:
        return " / ".join(keyword_terms[:_MAX_LABEL_KEYWORDS])
    return f"topic-{topic_id}"


def _representative_documents(
    topic_document_ids: list[str],
    topic_id: int,
    output: UnifiedExtractionOutput,
) -> list[TopicRepresentativeDocument]:
    """Build representative documents for one topic.

    Args:
        topic_document_ids (list[str]): Topic document IDs.
        topic_id (int): Current topic identifier.
        output (UnifiedExtractionOutput): Unified extraction output.

    Returns:
        list[TopicRepresentativeDocument]: Representative document descriptors.

    """
    if not topic_document_ids:
        return []

    rank_by_id: dict[str, int] = {}
    if output.document_topics:
        for link in output.document_topics:
            if link.topic_id != topic_id or link.rank is None:
                continue
            previous_rank = rank_by_id.get(link.document_id)
            if previous_rank is None or link.rank < previous_rank:
                rank_by_id[link.document_id] = link.rank

    ordered_ids = _dedupe_preserve_order(topic_document_ids)
    return [
        TopicRepresentativeDocument(
            document_id=document_id,
            rank=rank_by_id.get(document_id),
        )
        for document_id in ordered_ids[:_MAX_REPRESENTATIVE_DOCUMENTS]
    ]


def _build_topic_summary(keyword_terms: list[str], document_count: int) -> str:
    """Build a compact topic summary.

    Args:
        keyword_terms (list[str]): Ranked keyword terms.
        document_count (int): Number of topic-linked documents.

    Returns:
        str: Human-readable summary.

    """
    if not keyword_terms:
        return f"Representative topic covering {document_count} documents."
    keywords_preview = ", ".join(keyword_terms[:_MAX_SUMMARY_KEYWORDS])
    return f"Topic centered on: {keywords_preview}. Documents: {document_count}."


def characterize_output(output: UnifiedExtractionOutput) -> UnifiedExtractionOutput:
    """Enrich unified extraction output with characterization fields.

    Args:
        output (UnifiedExtractionOutput): Extraction output to enrich.

    Returns:
        UnifiedExtractionOutput: Enriched output.

    """
    for topic in output.topics:
        keyword_terms = [keyword.term for keyword in topic.keywords if keyword.term]
        topic.label = _infer_label(topic.label, keyword_terms, topic.topic_id)
        topic.representative_documents = _representative_documents(
            topic.document_ids,
            topic.topic_id,
            output,
        )
        topic.summary = _build_topic_summary(keyword_terms, len(topic.document_ids))

    if output.focus in {OutputFocus.DOCUMENTS, OutputFocus.BOTH} and output.document_topics is None:
        output.document_topics = []

    output.notes.append("Topic characterization layer applied (labels, representatives, summaries).")
    return output


def build_benchmark_comparison(outputs: dict[str, UnifiedExtractionOutput]) -> dict[str, Any]:
    """Build a lightweight comparison report across benchmark methods.

    Args:
        outputs (dict[str, UnifiedExtractionOutput]): Benchmark outputs keyed by method.

    Returns:
        dict[str, Any]: Comparison block.

    """
    keyword_sets: dict[str, set[str]] = {
        method: {
            keyword.term.lower() for topic in payload.topics for keyword in topic.keywords if keyword.term
        }
        for method, payload in outputs.items()
    }

    pairwise_overlap: list[dict[str, Any]] = []
    for left_method, right_method in combinations(sorted(keyword_sets.keys()), 2):
        left = keyword_sets[left_method]
        right = keyword_sets[right_method]
        intersection = left & right
        union = left | right
        jaccard = 0.0 if not union else float(len(intersection) / len(union))
        pairwise_overlap.append(
            {
                "left_method": left_method,
                "right_method": right_method,
                "overlap_count": len(intersection),
                "jaccard": jaccard,
                "overlap_terms_preview": sorted(intersection)[:10],
            },
        )

    return {
        "method_count": len(outputs),
        "pairwise_overlap": pairwise_overlap,
    }
