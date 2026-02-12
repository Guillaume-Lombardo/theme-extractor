"""Baseline extraction strategies backed by TF-IDF and search aggregations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from theme_extractor.domain import (
    DocumentTopicLink,
    ExtractMethod,
    OutputFocus,
    TopicKeyword,
    TopicResult,
    UnifiedExtractionOutput,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sklearn.feature_extraction.text import TfidfVectorizer

    from theme_extractor.search.protocols import SearchBackend

_AGG_TERMS_KEY = "terms"
_AGG_THEMES_KEY = "themes"
_UNSUPPORTED_BASELINE_ERROR = "Unsupported baseline extraction method: {method!r}."


class BaselineExtractionConfig(BaseModel):
    """Represent baseline extraction runtime options.

    Args:
        query: Query string used for search/aggregation.
        fields: Text fields used in search query.
        source_field: Source field used to build TF-IDF corpus.
        top_n: Number of top terms/topics to return.
        search_size: Number of documents fetched for TF-IDF.
        aggregation_field: Aggregation field for terms/significant_terms/significant_text.
        terms_min_doc_count: Minimum bucket doc count for terms aggregation.
        sigtext_filter_duplicate: Whether significant_text should filter duplicate text.

    """

    model_config = ConfigDict(frozen=True)

    query: str = "match_all"
    fields: tuple[str, ...] = ("content", "filename", "path")
    source_field: str = "content"
    top_n: int = 25
    search_size: int = 200
    aggregation_field: str = "tokens"
    terms_min_doc_count: int = 1
    sigtext_filter_duplicate: bool = True


class BaselineRunRequest(BaseModel):
    """Represent one baseline method execution request.

    Args:
        method: Extraction method to execute.
        index: Target backend index.
        focus: Output focus mode.
        config: Baseline extraction configuration.

    """

    model_config = ConfigDict(frozen=True)

    method: ExtractMethod
    index: str
    focus: OutputFocus
    config: BaselineExtractionConfig


def _get_tfidf_vectorizer() -> type[TfidfVectorizer]:
    """Return TfidfVectorizer class via lazy import.

    Returns:
        type[TfidfVectorizer]: TfidfVectorizer class.

    """
    from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: PLC0415

    return TfidfVectorizer


def _normalized_query(config: BaselineExtractionConfig) -> dict[str, Any]:
    """Build normalized query expression.

    Args:
        config: Baseline extraction configuration.

    Returns:
        dict[str, Any]: Query expression for backend requests.

    """
    query = config.query.strip()
    if query in {"", "*", "match_all"}:
        return {"match_all": {}}

    return {
        "simple_query_string": {
            "query": query,
            "fields": list(config.fields),
            "default_operator": "and",
        },
    }


def _search_body(config: BaselineExtractionConfig) -> dict[str, Any]:
    """Build search body used to fetch TF-IDF documents.

    Args:
        config: Baseline extraction configuration.

    Returns:
        dict[str, Any]: Backend search body.

    """
    return {
        "size": max(1, config.search_size),
        "query": _normalized_query(config),
        "_source": {"includes": [config.source_field]},
    }


def _terms_aggregation_body(config: BaselineExtractionConfig) -> dict[str, Any]:
    """Build backend body for terms aggregation.

    Args:
        config: Baseline extraction configuration.

    Returns:
        dict[str, Any]: Aggregation query body.

    """
    return {
        "size": 0,
        "query": _normalized_query(config),
        "aggs": {
            _AGG_TERMS_KEY: {
                "terms": {
                    "field": config.aggregation_field,
                    "size": max(1, config.top_n),
                    "min_doc_count": max(1, config.terms_min_doc_count),
                },
            },
        },
    }


def _significant_terms_body(config: BaselineExtractionConfig) -> dict[str, Any]:
    """Build backend body for significant_terms aggregation.

    Args:
        config: Baseline extraction configuration.

    Returns:
        dict[str, Any]: Aggregation query body.

    """
    return {
        "size": 0,
        "query": _normalized_query(config),
        "aggs": {
            _AGG_THEMES_KEY: {
                "significant_terms": {
                    "field": config.aggregation_field,
                    "size": max(1, config.top_n),
                },
            },
        },
    }


def _significant_text_body(config: BaselineExtractionConfig) -> dict[str, Any]:
    """Build backend body for significant_text aggregation.

    Args:
        config: Baseline extraction configuration.

    Returns:
        dict[str, Any]: Aggregation query body.

    """
    return {
        "size": 0,
        "query": _normalized_query(config),
        "aggs": {
            _AGG_THEMES_KEY: {
                "significant_text": {
                    "field": config.aggregation_field,
                    "size": max(1, config.top_n),
                    "filter_duplicate_text": config.sigtext_filter_duplicate,
                },
            },
        },
    }


def _to_document_topics(
    *,
    focus: OutputFocus,
    document_ids: Sequence[str],
    topic_id: int,
) -> list[DocumentTopicLink] | None:
    """Build optional document-topic links for topic-focused outputs.

    Args:
        focus: Output focus mode.
        document_ids: Document identifiers associated to the topic.
        topic_id: Topic identifier.

    Returns:
        list[DocumentTopicLink] | None: Document-topic links or `None` when disabled.

    """
    if focus not in {OutputFocus.DOCUMENTS, OutputFocus.BOTH}:
        return None

    return [
        DocumentTopicLink(document_id=document_id, topic_id=topic_id, rank=rank)
        for rank, document_id in enumerate(document_ids, start=1)
    ]


def _extract_tfidf_topic(
    *,
    backend: SearchBackend,
    index: str,
    focus: OutputFocus,
    config: BaselineExtractionConfig,
) -> tuple[list[TopicResult], list[DocumentTopicLink] | None]:
    """Run TF-IDF baseline on search results.

    Args:
        backend: Search backend adapter.
        index: Target backend index.
        focus: Output focus mode.
        config: Baseline extraction configuration.

    Returns:
        tuple[list[TopicResult], list[DocumentTopicLink] | None]: Topics and document-topic links.

    """
    response = backend.search_documents(index=index, body=_search_body(config))

    hits = (response.get("hits") or {}).get("hits") or []
    documents: list[str] = []
    document_ids: list[str] = []

    for hit in hits:
        source = hit.get("_source") or {}
        value = source.get(config.source_field)
        if not isinstance(value, str) or not value.strip():
            continue
        documents.append(value)
        document_ids.append(str(hit.get("_id", f"doc-{len(document_ids)}")))

    if not documents:
        empty_doc_topics = [] if focus in {OutputFocus.DOCUMENTS, OutputFocus.BOTH} else None
        return [], empty_doc_topics

    tfidf_vectorizer = _get_tfidf_vectorizer()
    vectorizer = tfidf_vectorizer(ngram_range=(1, 3), min_df=1, max_features=50_000)
    matrix = vectorizer.fit_transform(documents)
    scores = matrix.sum(axis=0).A1
    feature_names = vectorizer.get_feature_names_out()

    ranked = sorted(
        zip(feature_names, scores, strict=True),
        key=lambda item: float(item[1]),
        reverse=True,
    )[: max(1, config.top_n)]

    keywords = [TopicKeyword(term=str(term), score=float(score)) for term, score in ranked]
    topic = TopicResult(
        topic_id=0,
        label="tfidf",
        score=float(sum(score for _, score in ranked)),
        keywords=keywords,
        document_ids=document_ids,
    )

    return [topic], _to_document_topics(focus=focus, document_ids=document_ids, topic_id=0)


def _parse_buckets(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract aggregation buckets from backend response.

    Args:
        response: Backend raw response.

    Returns:
        list[dict[str, Any]]: Aggregation buckets.

    """
    aggregations = response.get("aggregations") or {}
    if isinstance(aggregations.get(_AGG_TERMS_KEY), dict):
        return list(aggregations[_AGG_TERMS_KEY].get("buckets") or [])
    if isinstance(aggregations.get(_AGG_THEMES_KEY), dict):
        return list(aggregations[_AGG_THEMES_KEY].get("buckets") or [])
    return []


def _buckets_to_topics(buckets: Sequence[dict[str, Any]]) -> list[TopicResult]:
    """Convert aggregation buckets into normalized topics.

    Args:
        buckets: Backend aggregation buckets.

    Returns:
        list[TopicResult]: Normalized topics.

    """
    topics: list[TopicResult] = []
    for topic_id, bucket in enumerate(buckets):
        term = str(bucket.get("key", ""))
        doc_count = int(bucket.get("doc_count", 0))
        score_value = bucket.get("score")
        score = float(score_value) if isinstance(score_value, (int, float)) else float(doc_count)
        topics.append(
            TopicResult(
                topic_id=topic_id,
                label=term,
                score=score,
                keywords=[TopicKeyword(term=term, score=score)],
                document_ids=[],
            ),
        )
    return topics


def run_baseline_method(
    *,
    backend: SearchBackend,
    request: BaselineRunRequest,
    output: UnifiedExtractionOutput,
) -> UnifiedExtractionOutput:
    """Run one baseline extraction method and fill output payload.

    Args:
        backend: Search backend adapter.
        request: Baseline run request payload.
        output: Output object to enrich.

    Returns:
        UnifiedExtractionOutput: Updated output payload.

    Raises:
        ValueError: If request.method is not a supported baseline method.

    """
    if request.method == ExtractMethod.BASELINE_TFIDF:
        topics, document_topics = _extract_tfidf_topic(
            backend=backend,
            index=request.index,
            focus=request.focus,
            config=request.config,
        )
        output.topics = topics
        output.document_topics = document_topics
        output.notes.append("TF-IDF baseline executed from backend search corpus.")
        return output

    if request.method == ExtractMethod.TERMS:
        response = backend.terms_aggregation(
            index=request.index,
            body=_terms_aggregation_body(request.config),
        )
        output.topics = _buckets_to_topics(_parse_buckets(response))
        if request.focus in {OutputFocus.DOCUMENTS, OutputFocus.BOTH}:
            output.document_topics = []
        output.notes.append("Terms aggregation baseline executed.")
        return output

    if request.method == ExtractMethod.SIGNIFICANT_TERMS:
        response = backend.significant_terms_aggregation(
            index=request.index,
            body=_significant_terms_body(request.config),
        )
        output.topics = _buckets_to_topics(_parse_buckets(response))
        if request.focus in {OutputFocus.DOCUMENTS, OutputFocus.BOTH}:
            output.document_topics = []
        output.notes.append("significant_terms aggregation baseline executed.")
        return output

    if request.method == ExtractMethod.SIGNIFICANT_TEXT:
        response = backend.significant_text_aggregation(
            index=request.index,
            body=_significant_text_body(request.config),
        )
        output.topics = _buckets_to_topics(_parse_buckets(response))
        if request.focus in {OutputFocus.DOCUMENTS, OutputFocus.BOTH}:
            output.document_topics = []
        output.notes.append("significant_text aggregation baseline executed.")
        return output

    raise ValueError(_UNSUPPORTED_BASELINE_ERROR.format(method=request.method))
