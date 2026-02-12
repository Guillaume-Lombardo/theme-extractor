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


class BertopicDimReduction(StrEnum):
    """Represent BERTopic dimensionality reduction choices."""

    NONE = "none"
    SVD = "svd"
    NMF = "nmf"
    UMAP = "umap"


class BertopicClustering(StrEnum):
    """Represent BERTopic clustering choices."""

    KMEANS = "kmeans"
    HDBSCAN = "hdbscan"


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


class CleaningOptionFlag(IntFlag):
    """Represent one or more cleaning options as bit flags."""

    NONE = 0
    WHITESPACE = auto()
    ACCENT_NORMALIZATION = auto()
    HEADER_FOOTER = auto()
    BOILERPLATE = auto()
    TOKEN_CLEANUP = auto()
    HTML_STRIP = auto()


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


_CLEANING_NAME_TO_FLAG: dict[str, CleaningOptionFlag] = {
    "none": CleaningOptionFlag.NONE,
    "whitespace": CleaningOptionFlag.WHITESPACE,
    "accent_normalization": CleaningOptionFlag.ACCENT_NORMALIZATION,
    "header_footer": CleaningOptionFlag.HEADER_FOOTER,
    "boilerplate": CleaningOptionFlag.BOILERPLATE,
    "token_cleanup": CleaningOptionFlag.TOKEN_CLEANUP,
    "html_strip": CleaningOptionFlag.HTML_STRIP,
}


def default_cleaning_options() -> CleaningOptionFlag:
    """Return default cleaning options for ingestion.

    Returns:
        CleaningOptionFlag: Default combined cleaning options.

    """
    return (
        CleaningOptionFlag.WHITESPACE
        | CleaningOptionFlag.ACCENT_NORMALIZATION
        | CleaningOptionFlag.HEADER_FOOTER
        | CleaningOptionFlag.BOILERPLATE
        | CleaningOptionFlag.TOKEN_CLEANUP
        | CleaningOptionFlag.HTML_STRIP
    )


def cleaning_flag_from_string(value: str) -> CleaningOptionFlag:
    """Parse comma-separated cleaning options into a flag.

    Args:
        value (str): Raw comma-separated cleaning options.

    Raises:
        ValueError: If one option is unsupported or if no option is provided.

    Returns:
        CleaningOptionFlag: Combined cleaning options.

    """
    flag = CleaningOptionFlag.NONE
    tokens = [raw.strip().lower() for raw in value.split(",") if raw.strip()]

    if not tokens:
        raise ValueError("At least one cleaning option must be provided.")  # noqa: TRY003

    if "all" in tokens and len(tokens) > 1:
        raise ValueError("'all' cannot be combined with other cleaning options.")  # noqa: TRY003
    if "none" in tokens and len(tokens) > 1:
        raise ValueError("'none' cannot be combined with other cleaning options.")  # noqa: TRY003

    if tokens == ["all"]:
        return default_cleaning_options()
    if tokens == ["none"]:
        return CleaningOptionFlag.NONE

    for normalized in tokens:
        if normalized not in _CLEANING_NAME_TO_FLAG:
            supported = ", ".join(["all", *_CLEANING_NAME_TO_FLAG.keys()])
            raise ValueError(  # noqa: TRY003
                f"Unsupported cleaning option '{normalized}'. Supported values: {supported}.",
            )
        if normalized == "none":
            continue
        flag |= _CLEANING_NAME_TO_FLAG[normalized]

    if flag == CleaningOptionFlag.NONE:
        raise ValueError("At least one cleaning option must be provided.")  # noqa: TRY003

    return flag


def cleaning_flag_to_string(flag: CleaningOptionFlag) -> str:
    """Convert cleaning option flags to canonical comma-separated form.

    Args:
        flag (CleaningOptionFlag): Combined cleaning options.

    Returns:
        str: Canonical comma-separated cleaning options.

    """
    if flag == CleaningOptionFlag.NONE:
        return "none"

    names = [
        name for name, bit in _CLEANING_NAME_TO_FLAG.items() if bit != CleaningOptionFlag.NONE and flag & bit
    ]
    return ",".join(names)
