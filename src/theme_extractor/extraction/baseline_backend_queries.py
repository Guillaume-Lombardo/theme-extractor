"""Backend query builders shared by baseline extraction strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sklearn.feature_extraction.text import TfidfVectorizer

    from theme_extractor.extraction.baselines import BaselineExtractionConfig

_AGG_TERMS_KEY = "terms"
_AGG_THEMES_KEY = "themes"


def get_tfidf_vectorizer() -> type[TfidfVectorizer]:
    """Return TfidfVectorizer class via lazy import.

    Returns:
        type[TfidfVectorizer]: TfidfVectorizer class.

    """
    from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: PLC0415

    return TfidfVectorizer


def normalized_query(config: BaselineExtractionConfig) -> dict[str, Any]:
    """Build normalized query expression.

    Args:
        config (BaselineExtractionConfig): Baseline extraction configuration.

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


def search_body(config: BaselineExtractionConfig) -> dict[str, Any]:
    """Build search body used to fetch TF-IDF documents.

    Returns:
        dict[str, Any]: Backend search body.

    """
    return {
        "size": max(1, config.search_size),
        "query": normalized_query(config),
        "_source": {"includes": [config.source_field]},
    }


def terms_aggregation_body(config: BaselineExtractionConfig) -> dict[str, Any]:
    """Build backend body for terms aggregation.

    Returns:
        dict[str, Any]: Backend aggregation payload.

    """
    return {
        "size": 0,
        "query": normalized_query(config),
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


def significant_terms_body(config: BaselineExtractionConfig) -> dict[str, Any]:
    """Build backend body for significant_terms aggregation.

    Returns:
        dict[str, Any]: Backend aggregation payload.

    """
    return {
        "size": 0,
        "query": normalized_query(config),
        "aggs": {
            _AGG_THEMES_KEY: {
                "significant_terms": {
                    "field": config.aggregation_field,
                    "size": max(1, config.top_n),
                },
            },
        },
    }


def significant_text_body(config: BaselineExtractionConfig) -> dict[str, Any]:
    """Build backend body for significant_text aggregation.

    Returns:
        dict[str, Any]: Backend aggregation payload.

    """
    return {
        "size": 0,
        "query": normalized_query(config),
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
