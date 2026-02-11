"""Text cleaning utilities for ingestion pipeline."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from html import unescape

from theme_extractor.domain import CleaningOptionFlag

_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'-]*")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTISPACE_RE = re.compile(r"[ \t]+")
_PAGE_PATTERN_RE = re.compile(r"^page\s+\d+(\s*/\s*\d+)?$", re.IGNORECASE)
_NUM_PAGE_RE = re.compile(r"^\d+\s*/\s*\d+$")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
_MIN_HEADER_FOOTER_LINES = 6


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text.

    Args:
        text (str): Input text.

    Returns:
        str: Text with normalized whitespace.

    """
    lines = [
        _MULTISPACE_RE.sub(" ", line).strip()
        for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    ]
    cleaned_lines = [line for line in lines if line]
    return "\n".join(cleaned_lines)


def normalize_french_accents(text: str) -> str:
    """Normalize accents for French-like text.

    Args:
        text (str): Input text.

    Returns:
        str: Accent-stripped text.

    """
    return "".join(char for char in unicodedata.normalize("NFKD", text) if not unicodedata.combining(char))


def strip_html(text: str) -> str:
    """Strip HTML tags and decode HTML entities.

    Args:
        text (str): Raw HTML text.

    Returns:
        str: Plain text extracted from HTML.

    """
    no_tags = _HTML_TAG_RE.sub(" ", text)
    return unescape(no_tags)


def remove_boilerplate(text: str) -> str:
    """Remove basic boilerplate patterns from text.

    Args:
        text (str): Input text.

    Returns:
        str: Text without obvious boilerplate.

    """
    return _EMAIL_RE.sub(" ", _URL_RE.sub(" ", text))


def suppress_headers_footers(text: str) -> str:
    """Suppress repeated header/footer-like lines.

    Args:
        text (str): Input text.

    Returns:
        str: Text after removing repeated header/footer lines.

    """
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) < _MIN_HEADER_FOOTER_LINES:
        return text

    candidate_lines = lines[:3] + lines[-3:]
    lowered_candidates = [line.lower() for line in candidate_lines]
    counts = Counter(lowered_candidates)

    to_remove = {
        line
        for line, count in counts.items()
        if count > 1 or _PAGE_PATTERN_RE.match(line) or _NUM_PAGE_RE.match(line)
    }

    filtered = [line for line in lines if line.lower() not in to_remove]
    return "\n".join(filtered)


def tokenize_for_ingestion(text: str) -> list[str]:
    """Tokenize cleaned text for stopword and corpus analysis.

    Args:
        text (str): Cleaned text.

    Returns:
        list[str]: Lower-cased tokens.

    """
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]


def discover_auto_stopwords(
    tokenized_documents: list[list[str]],
    *,
    min_doc_ratio: float,
    max_terms: int,
) -> set[str]:
    """Discover high-frequency stopwords directly from corpus statistics.

    Args:
        tokenized_documents (list[list[str]]): Tokenized documents.
        min_doc_ratio (float): Minimum document ratio threshold.
        max_terms (int): Maximum number of generated stopwords.

    Returns:
        set[str]: Automatically discovered stopwords.

    """
    if not tokenized_documents:
        return set()

    total_docs = len(tokenized_documents)
    document_frequency: Counter[str] = Counter()

    for tokens in tokenized_documents:
        document_frequency.update(set(tokens))

    candidates = [
        term
        for term, count in document_frequency.items()
        if count / total_docs >= min_doc_ratio and len(term) > 1
    ]
    candidates.sort(key=lambda term: (-document_frequency[term], term))
    return set(candidates[:max_terms])


def apply_cleaning_options(text: str, *, options: CleaningOptionFlag) -> str:
    """Apply configured cleaning options to text.

    Args:
        text (str): Raw text.
        options (CleaningOptionFlag): Cleaning options flag.

    Returns:
        str: Cleaned text.

    """
    output = text

    if options & CleaningOptionFlag.HTML_STRIP:
        output = strip_html(output)

    if options & CleaningOptionFlag.BOILERPLATE:
        output = remove_boilerplate(output)

    if options & CleaningOptionFlag.WHITESPACE:
        output = normalize_whitespace(output)

    if options & CleaningOptionFlag.HEADER_FOOTER:
        output = suppress_headers_footers(output)

    if options & CleaningOptionFlag.ACCENT_NORMALIZATION:
        output = normalize_french_accents(output)

    if options & CleaningOptionFlag.TOKEN_CLEANUP:
        output = " ".join(tokenize_for_ingestion(output))

    if options & CleaningOptionFlag.WHITESPACE:
        output = normalize_whitespace(output)

    return output
