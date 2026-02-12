"""KeyBERT extraction strategy with offline-friendly fallback behavior."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from theme_extractor.domain import (
    DocumentTopicLink,
    OutputFocus,
    TopicKeyword,
    TopicResult,
    UnifiedExtractionOutput,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from theme_extractor.search.protocols import SearchBackend

from theme_extractor.extraction.baselines import BaselineExtractionConfig  # noqa: TC001


class KeyBertRunRequest(BaseModel):
    """Represent one KeyBERT execution request.

    Args:
        index (str): Target backend index.
        focus (OutputFocus): Output focus mode.
        config (BaselineExtractionConfig): Runtime extraction config.
        keybert_config (KeyBertExtractionConfig): KeyBERT runtime options.

    """

    model_config = ConfigDict(frozen=True)

    index: str
    focus: OutputFocus
    config: BaselineExtractionConfig
    keybert_config: KeyBertExtractionConfig


_DEFAULT_LOCAL_MODELS_DIR = Path("data/models")


class KeyBertExtractionConfig(BaseModel):
    """Represent KeyBERT extraction runtime options.

    Args:
        use_embeddings (bool): Whether to force a custom embedding model.
        embedding_model (str): Embedding model id or local path.
        local_models_dir (Path | None): Local directory used to resolve model aliases.

    """

    model_config = ConfigDict(frozen=True)

    use_embeddings: bool = False
    embedding_model: str = "bge-m3"
    local_models_dir: Path | None = _DEFAULT_LOCAL_MODELS_DIR


_KEYWORD_TERM_AND_SCORE_LENGTH = 2


def _normalized_query(query: str, fields: Sequence[str]) -> dict[str, Any]:
    """Build normalized query expression.

    Args:
        query (str): Raw query string.
        fields (Sequence[str]): Search fields.

    Returns:
        dict[str, Any]: Query expression.

    """
    normalized = query.strip()
    if normalized in {"", "*", "match_all"}:
        return {"match_all": {}}

    return {
        "simple_query_string": {
            "query": normalized,
            "fields": list(fields),
            "default_operator": "and",
        },
    }


def _search_body(config: BaselineExtractionConfig) -> dict[str, Any]:
    """Build search body used to fetch documents for KeyBERT.

    Args:
        config (BaselineExtractionConfig): Runtime extraction config.

    Returns:
        dict[str, Any]: Backend search payload.

    """
    return {
        "size": max(1, config.search_size),
        "query": _normalized_query(config.query, config.fields),
        "_source": {"includes": [config.source_field]},
    }


def _to_document_topics(
    *,
    focus: OutputFocus,
    document_ids: Sequence[str],
    topic_id: int,
) -> list[DocumentTopicLink] | None:
    """Build optional document-topic links.

    Args:
        focus (OutputFocus): Output focus mode.
        document_ids (Sequence[str]): Document identifiers.
        topic_id (int): Topic identifier.

    Returns:
        list[DocumentTopicLink] | None: Document-topic links or `None`.

    """
    if focus not in {OutputFocus.DOCUMENTS, OutputFocus.BOTH}:
        return None

    return [
        DocumentTopicLink(document_id=document_id, topic_id=topic_id, rank=rank)
        for rank, document_id in enumerate(document_ids, start=1)
    ]


def _extract_keywords_with_tfidf_fallback(
    *,
    corpus_text: str,
    top_n: int,
) -> list[tuple[str, float]]:
    """Extract keywords with TF-IDF fallback.

    Args:
        corpus_text (str): Input corpus text.
        top_n (int): Number of keywords to return.

    Returns:
        list[tuple[str, float]]: Ranked `(term, score)` list.

    """
    from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: PLC0415

    vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=1, max_features=50_000)
    matrix = vectorizer.fit_transform([corpus_text])
    scores = matrix.sum(axis=0).A1
    feature_names = vectorizer.get_feature_names_out()

    ranked = sorted(
        zip(feature_names, scores, strict=True),
        key=lambda item: float(item[1]),
        reverse=True,
    )[: max(1, top_n)]
    return [(str(term), float(score)) for term, score in ranked]


def _extract_keywords_with_keybert(
    *,
    corpus_text: str,
    top_n: int,
    keybert_config: KeyBertExtractionConfig,
) -> list[tuple[str, float]]:
    """Extract keywords with KeyBERT if dependency is available.

    Args:
        corpus_text (str): Input corpus text.
        top_n (int): Number of keywords to return.
        keybert_config (KeyBertExtractionConfig): KeyBERT runtime options.

    Returns:
        list[tuple[str, float]]: Ranked `(term, score)` list.

    """
    from keybert import KeyBERT  # noqa: PLC0415

    model_name = _resolve_embedding_model_name(
        embedding_model=keybert_config.embedding_model,
        local_models_dir=keybert_config.local_models_dir,
    )
    if keybert_config.use_embeddings:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        embedding_model = SentenceTransformer(model_name)
        model = KeyBERT(model=embedding_model)
    else:
        model = KeyBERT()
    raw_keywords = model.extract_keywords(
        corpus_text,
        keyphrase_ngram_range=(1, 3),
        stop_words="english",
        top_n=max(1, top_n),
    )
    normalized_keywords: list[tuple[str, float]] = []
    for item in raw_keywords:
        if isinstance(item, tuple) and len(item) >= _KEYWORD_TERM_AND_SCORE_LENGTH:
            term = item[0]
            score_raw = item[1]
            if isinstance(term, str) and isinstance(score_raw, int | float):
                normalized_keywords.append((term, float(score_raw)))
    return normalized_keywords


def _resolve_embedding_model_name(
    *,
    embedding_model: str,
    local_models_dir: Path | None,
) -> str:
    """Resolve model id to a direct local path when available.

    Args:
        embedding_model (str): Raw model value from CLI.
        local_models_dir (Path | None): Optional local aliases directory.

    Returns:
        str: Resolved model value.

    """
    normalized = embedding_model.strip()
    if not normalized:
        return normalized

    direct_path = Path(normalized).expanduser()
    if direct_path.exists():
        return str(direct_path.resolve())

    if local_models_dir is not None:
        local_candidate = (local_models_dir / normalized).expanduser()
        if local_candidate.exists():
            return str(local_candidate.resolve())

    return normalized


def run_keybert_method(
    *,
    backend: SearchBackend,
    request: KeyBertRunRequest,
    output: UnifiedExtractionOutput,
) -> UnifiedExtractionOutput:
    """Run KeyBERT extraction strategy and populate unified output.

    Args:
        backend (SearchBackend): Search backend adapter.
        request (KeyBertRunRequest): KeyBERT runtime request.
        output (UnifiedExtractionOutput): Mutable output payload.

    Returns:
        UnifiedExtractionOutput: Updated output payload.

    """
    response = backend.search_documents(index=request.index, body=_search_body(request.config))
    hits = (response.get("hits") or {}).get("hits") or []

    documents: list[str] = []
    document_ids: list[str] = []
    for hit in hits:
        source = hit.get("_source") or {}
        value = source.get(request.config.source_field)
        if not isinstance(value, str) or not value.strip():
            continue
        documents.append(value)
        document_ids.append(str(hit.get("_id", f"doc-{len(document_ids)}")))

    if not documents:
        output.notes.append("KeyBERT executed with empty corpus from backend search.")
        if request.focus in {OutputFocus.DOCUMENTS, OutputFocus.BOTH}:
            output.document_topics = []
        else:
            output.document_topics = None
        return output

    corpus_text = "\n".join(documents)
    try:
        ranked_keywords = _extract_keywords_with_keybert(
            corpus_text=corpus_text,
            top_n=request.config.top_n,
            keybert_config=request.keybert_config,
        )
        output.notes.append("KeyBERT strategy executed with keybert dependency.")
    except ImportError:
        ranked_keywords = _extract_keywords_with_tfidf_fallback(
            corpus_text=corpus_text,
            top_n=request.config.top_n,
        )
        output.notes.append("KeyBERT dependency missing; TF-IDF fallback was used.")

    keywords = [TopicKeyword(term=term, score=score) for term, score in ranked_keywords]
    topic = TopicResult(
        topic_id=0,
        label="keybert",
        score=float(sum(score for _, score in ranked_keywords)),
        keywords=keywords,
        document_ids=document_ids,
    )
    output.topics = [topic]
    output.document_topics = _to_document_topics(
        focus=request.focus,
        document_ids=document_ids,
        topic_id=topic.topic_id,
    )
    return output
