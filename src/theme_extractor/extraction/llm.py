"""LLM extraction strategy with strict-offline TF-IDF fallback."""

from __future__ import annotations

import json
import os
from importlib import import_module
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from theme_extractor.domain import (
    DocumentTopicLink,
    LlmProvider,
    OfflinePolicy,
    OutputFocus,
    TopicKeyword,
    TopicResult,
    UnifiedExtractionOutput,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from theme_extractor.search.protocols import SearchBackend

from theme_extractor.extraction.baselines import BaselineExtractionConfig  # noqa: TC001

_OFFLINE_FALLBACK_NOTE = "LLM strategy executed in strict offline mode; TF-IDF fallback was used."
_MISSING_API_KEY_FALLBACK_NOTE = (
    "LLM credentials were not provided; TF-IDF fallback was used."  # pragma: allowlist secret
)
_LLM_RUNTIME_FALLBACK_NOTE = "LLM runtime failed; TF-IDF fallback was used."
_EMPTY_CORPUS_NOTE = "LLM strategy executed with empty corpus from backend search."
_LLM_SUCCESS_NOTE = "LLM strategy executed with provider response."
_OPENAI_PROMPT_MAX_CHARS = 20_000
_KEYWORD_TERM_AND_SCORE_LENGTH = 2


class LlmExtractionConfig(BaseModel):
    """Represent LLM extraction runtime options.

    Args:
        provider (LlmProvider): LLM provider identifier.
        model (str): Model name.
        api_key_env_var (str): Environment variable containing API key.
        api_base_url (str | None): Optional provider base URL.
        temperature (float): Sampling temperature.
        timeout_s (float): Request timeout in seconds.
        max_input_chars (int): Maximum corpus characters sent to provider.

    """

    model_config = ConfigDict(frozen=True)

    provider: LlmProvider = LlmProvider.OPENAI
    model: str = "gpt-4o-mini"
    api_key_env_var: str = "OPENAI_API_KEY"
    api_base_url: str | None = None
    temperature: float = 0.0
    timeout_s: float = 30.0
    max_input_chars: int = _OPENAI_PROMPT_MAX_CHARS


class LlmRunRequest(BaseModel):
    """Represent one LLM execution request.

    Args:
        index (str): Target backend index.
        focus (OutputFocus): Output focus mode.
        offline_policy (OfflinePolicy): Offline execution policy.
        baseline_config (BaselineExtractionConfig): Backend corpus retrieval options.
        llm_config (LlmExtractionConfig): LLM runtime options.

    """

    model_config = ConfigDict(frozen=True)

    index: str
    focus: OutputFocus
    offline_policy: OfflinePolicy
    baseline_config: BaselineExtractionConfig
    llm_config: LlmExtractionConfig


def _normalized_query(query: str, fields: Sequence[str]) -> dict[str, Any]:
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


def _parse_keyword_payload(
    *,
    payload_text: str,
    top_n: int,
) -> list[tuple[str, float]]:
    try:
        parsed = json.loads(payload_text)
    except json.JSONDecodeError:
        return []

    items: list[Any]
    if isinstance(parsed, dict) and isinstance(parsed.get("keywords"), list):
        items = list(parsed["keywords"])
    elif isinstance(parsed, list):
        items = list(parsed)
    else:
        return []

    ranked_keywords: list[tuple[str, float]] = []
    for item in items:
        if isinstance(item, dict):
            term = item.get("term")
            score_raw = item.get("score", 0.0)
            if isinstance(term, str) and isinstance(score_raw, int | float):
                ranked_keywords.append((term, float(score_raw)))
        elif isinstance(item, str):
            ranked_keywords.append((item, 0.0))

    return ranked_keywords[: max(1, top_n)]


def _extract_keywords_with_openai(
    *,
    corpus_text: str,
    config: LlmExtractionConfig,
    top_n: int,
    api_key: str,
) -> list[tuple[str, float]]:
    openai_module = import_module("openai")
    openai_client_cls = openai_module.OpenAI
    client = openai_client_cls(
        api_key=api_key,
        base_url=config.api_base_url,
        timeout=config.timeout_s,
    )
    clipped_corpus = corpus_text[: max(1, config.max_input_chars)]
    prompt = (
        "Extract the most representative themes from this corpus.\n"
        "Return only valid JSON as either:\n"
        '  {"keywords":[{"term":"...","score":0.0}]}\n'
        "or:\n"
        '  [{"term":"...","score":0.0}]\n'
        f"Limit to {max(1, top_n)} keywords.\n"
        "Corpus:\n"
        f"{clipped_corpus}"
    )

    response = client.chat.completions.create(
        model=config.model,
        temperature=config.temperature,
        messages=[
            {"role": "system", "content": "You are a topic extraction assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content or ""
    return _parse_keyword_payload(payload_text=content, top_n=top_n)


def _extract_keywords_with_llm(
    *,
    corpus_text: str,
    request: LlmRunRequest,
) -> tuple[list[tuple[str, float]], str]:
    if request.offline_policy == OfflinePolicy.STRICT:
        return (
            _extract_keywords_with_tfidf_fallback(
                corpus_text=corpus_text,
                top_n=request.baseline_config.top_n,
            ),
            _OFFLINE_FALLBACK_NOTE,
        )

    api_key = os.getenv(request.llm_config.api_key_env_var, "").strip()
    if not api_key:
        return (
            _extract_keywords_with_tfidf_fallback(
                corpus_text=corpus_text,
                top_n=request.baseline_config.top_n,
            ),
            _MISSING_API_KEY_FALLBACK_NOTE,
        )

    try:
        if request.llm_config.provider == LlmProvider.OPENAI:
            keywords = _extract_keywords_with_openai(
                corpus_text=corpus_text,
                config=request.llm_config,
                top_n=request.baseline_config.top_n,
                api_key=api_key,
            )
        else:
            keywords = []
    except Exception:
        return (
            _extract_keywords_with_tfidf_fallback(
                corpus_text=corpus_text,
                top_n=request.baseline_config.top_n,
            ),
            _LLM_RUNTIME_FALLBACK_NOTE,
        )

    if not keywords:
        return (
            _extract_keywords_with_tfidf_fallback(
                corpus_text=corpus_text,
                top_n=request.baseline_config.top_n,
            ),
            _LLM_RUNTIME_FALLBACK_NOTE,
        )

    return keywords, _LLM_SUCCESS_NOTE


def run_llm_method(
    *,
    backend: SearchBackend,
    request: LlmRunRequest,
    output: UnifiedExtractionOutput,
) -> UnifiedExtractionOutput:
    """Run LLM extraction strategy and populate unified output.

    Args:
        backend (SearchBackend): Search backend adapter.
        request (LlmRunRequest): LLM runtime request.
        output (UnifiedExtractionOutput): Mutable output payload.

    Returns:
        UnifiedExtractionOutput: Updated output payload.

    """
    response = backend.search_documents(index=request.index, body=_search_body(request.baseline_config))
    hits = (response.get("hits") or {}).get("hits") or []

    documents: list[str] = []
    document_ids: list[str] = []
    for hit in hits:
        source = hit.get("_source") or {}
        value = source.get(request.baseline_config.source_field)
        if not isinstance(value, str) or not value.strip():
            continue
        documents.append(value)
        document_ids.append(str(hit.get("_id", f"doc-{len(document_ids)}")))

    if not documents:
        output.notes.append(_EMPTY_CORPUS_NOTE)
        if request.focus in {OutputFocus.DOCUMENTS, OutputFocus.BOTH}:
            output.document_topics = []
        else:
            output.document_topics = None
        return output

    corpus_text = "\n".join(documents)
    ranked_keywords, note = _extract_keywords_with_llm(
        corpus_text=corpus_text,
        request=request,
    )
    output.notes.append(note)

    keywords = [TopicKeyword(term=term, score=score) for term, score in ranked_keywords]
    topic = TopicResult(
        topic_id=0,
        label="llm",
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
