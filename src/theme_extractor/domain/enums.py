"""Typed enumerations and conversions for CLI/domain choices."""

from __future__ import annotations

from enum import IntFlag, StrEnum, auto

from theme_extractor.errors import UnsupportedMethodError


class CommandName(StrEnum):
    """Represent supported top-level CLI commands."""

    INGEST = "ingest"
    EXTRACT = "extract"
    BENCHMARK = "benchmark"


class BackendName(StrEnum):
    """Represent supported search backends."""

    ELASTICSEARCH = "elasticsearch"
    OPENSEARCH = "opensearch"


class OfflinePolicy(StrEnum):
    """Represent supported offline execution policies."""

    STRICT = "strict"
    PRELOAD_OR_FIRST_RUN = "preload_or_first_run"


class OutputFocus(StrEnum):
    """Represent output focus modes for extraction results."""

    TOPICS = "topics"
    DOCUMENTS = "documents"
    BOTH = "both"


class ExtractMethod(StrEnum):
    """Represent one extraction method identifier."""

    BASELINE_TFIDF = "baseline_tfidf"
    TERMS = "terms"
    SIGNIFICANT_TERMS = "significant_terms"
    SIGNIFICANT_TEXT = "significant_text"
    KEYBERT = "keybert"
    BERTOPIC = "bertopic"
    LLM = "llm"


class ExtractMethodFlag(IntFlag):
    """Represent one or more extraction methods as bit flags."""

    NONE = 0
    BASELINE_TFIDF = auto()
    TERMS = auto()
    SIGNIFICANT_TERMS = auto()
    SIGNIFICANT_TEXT = auto()
    KEYBERT = auto()
    BERTOPIC = auto()
    LLM = auto()


_METHOD_TO_FLAG: dict[ExtractMethod, ExtractMethodFlag] = {
    ExtractMethod.BASELINE_TFIDF: ExtractMethodFlag.BASELINE_TFIDF,
    ExtractMethod.TERMS: ExtractMethodFlag.TERMS,
    ExtractMethod.SIGNIFICANT_TERMS: ExtractMethodFlag.SIGNIFICANT_TERMS,
    ExtractMethod.SIGNIFICANT_TEXT: ExtractMethodFlag.SIGNIFICANT_TEXT,
    ExtractMethod.KEYBERT: ExtractMethodFlag.KEYBERT,
    ExtractMethod.BERTOPIC: ExtractMethodFlag.BERTOPIC,
    ExtractMethod.LLM: ExtractMethodFlag.LLM,
}


def parse_extract_method(value: str) -> ExtractMethod:
    """Parse one extraction method string.

    Args:
        value (str): Raw method value.

    Raises:
        UnsupportedMethodError: If the method value is not supported.

    Returns:
        ExtractMethod: Parsed extraction method enum value.

    """
    normalized = value.strip().lower()
    try:
        return ExtractMethod(normalized)
    except ValueError as exc:
        raise UnsupportedMethodError(normalized) from exc


def method_flag_from_string(value: str) -> ExtractMethodFlag:
    """Parse a comma-separated method list into a method flag.

    Args:
        value (str): Comma-separated methods.

    Raises:
        ValueError: If no method is provided.

    Returns:
        ExtractMethodFlag: Combined method flag value.

    """
    flag = ExtractMethodFlag.NONE

    for raw in value.split(","):
        normalized = raw.strip().lower()
        if not normalized:
            continue
        method = parse_extract_method(normalized)
        flag |= _METHOD_TO_FLAG[method]

    if flag == ExtractMethodFlag.NONE:
        raise ValueError("At least one extraction method must be provided.")  # noqa: TRY003

    return flag


def method_flag_to_methods(flag: ExtractMethodFlag) -> list[ExtractMethod]:
    """Convert a method flag into an ordered method list.

    Args:
        flag (ExtractMethodFlag): Combined method flag value.

    Returns:
        list[ExtractMethod]: Ordered extraction method values.

    """
    return [method for method in ExtractMethod if flag & _METHOD_TO_FLAG[method]]


def method_flag_to_string(flag: ExtractMethodFlag) -> str:
    """Convert a method flag into a canonical comma-separated string.

    Args:
        flag (ExtractMethodFlag): Combined method flag value.

    Returns:
        str: Canonical comma-separated method list.

    """
    methods = method_flag_to_methods(flag)
    return ",".join(method.value for method in methods)
