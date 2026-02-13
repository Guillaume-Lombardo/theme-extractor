"""Text cleaning utilities for ingestion pipeline."""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import Counter
from functools import lru_cache
from html import unescape
from importlib.resources import files
from typing import TYPE_CHECKING

from theme_extractor.domain import CleaningOptionFlag

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'-]*")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTISPACE_RE = re.compile(r"[ \t]+")
_PAGE_PATTERN_RE = re.compile(r"^page\s+\d+(\s*/\s*\d+)?$", re.IGNORECASE)
_NUM_PAGE_RE = re.compile(r"^\d+\s*/\s*\d+$")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
_MIN_HEADER_FOOTER_LINES = 6
_MIN_PAGE_SEGMENTS = 2
_HEADER_FOOTER_WINDOW = 2
_REPEATED_LINE_MIN_RATIO = 0.6
_REPEATED_LINE_MIN_LEN = 4
_REPEATED_LINE_MAX_TOKENS = 12
_MIN_LINES_PER_SEGMENT = 2
_MIN_PAGE_NUMBER_MARKERS = 2


@lru_cache(maxsize=1)
def _default_stopword_column_names() -> set[str]:
    """Load default accepted stopword CSV column names from packaged JSON.

    Returns:
        set[str]: Accepted lower-cased column names.

    """
    resource = files("theme_extractor.resources").joinpath("stopword_column_names.json")
    names = json.loads(resource.read_text(encoding="utf-8"))
    return {
        normalized for name in names if isinstance(name, str) and (normalized := _normalize_stopword(name))
    }


@lru_cache(maxsize=1)
def _french_stopwords_fallback() -> set[str]:
    """Load fallback French stopwords from packaged JSON.

    Returns:
        set[str]: Fallback French stopwords.

    """
    resource = files("theme_extractor.resources").joinpath("stopwords_fr_fallback.json")
    words = json.loads(resource.read_text(encoding="utf-8"))
    return {
        normalized for word in words if isinstance(word, str) and (normalized := _normalize_stopword(word))
    }


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
    page_segments = _split_text_into_page_segments(text)
    if len(page_segments) >= _MIN_PAGE_SEGMENTS and _has_strong_page_boundary_evidence(text, page_segments):
        repeated = _detect_repeated_page_boundary_lines(page_segments)
        filtered_pages = [
            [line for line in page_lines if not _is_page_number_line(line) and line.lower() not in repeated]
            for page_lines in page_segments
        ]
        filtered_lines = [line for page in filtered_pages for line in page]
        return "\n".join(filtered_lines)

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


def _is_page_number_line(line: str) -> bool:
    """Return whether one line looks like a page number marker.

    Args:
        line (str): Input line.

    Returns:
        bool: True if line matches one page-number pattern.

    """
    normalized = line.strip().lower()
    return bool(_PAGE_PATTERN_RE.match(normalized) or _NUM_PAGE_RE.match(normalized))


def _split_text_into_page_segments(text: str) -> list[list[str]]:
    """Split text into probable page segments for boundary-line analysis.

    Args:
        text (str): Raw text.

    Returns:
        list[list[str]]: Segments represented as non-empty stripped lines.

    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    raw_segments = re.split(r"\f+|\n\s*\n", normalized)
    segments = [[line.strip() for line in segment.split("\n") if line.strip()] for segment in raw_segments]
    return [segment for segment in segments if len(segment) >= _MIN_LINES_PER_SEGMENT]


def _has_strong_page_boundary_evidence(text: str, page_segments: list[list[str]]) -> bool:
    """Check whether multipage suppression has enough page-boundary evidence.

    Args:
        text (str): Raw text.
        page_segments (list[list[str]]): Candidate page segments.

    Returns:
        bool: True when boundaries likely represent real pages.

    """
    if "\f" in text:
        return True

    page_markers = 0
    for page_lines in page_segments:
        boundaries = page_lines[:_HEADER_FOOTER_WINDOW] + page_lines[-_HEADER_FOOTER_WINDOW:]
        if any(_is_page_number_line(line) for line in boundaries):
            page_markers += 1
    return page_markers >= _MIN_PAGE_NUMBER_MARKERS


def _is_valid_repeated_boundary_line(line: str) -> bool:
    """Validate repeated boundary-line candidates for safe suppression.

    Args:
        line (str): Candidate line.

    Returns:
        bool: True if candidate should be considered removable.

    """
    if len(line) < _REPEATED_LINE_MIN_LEN:
        return False
    tokens = line.split()
    return len(tokens) <= _REPEATED_LINE_MAX_TOKENS


def _detect_repeated_page_boundary_lines(page_segments: list[list[str]]) -> set[str]:
    """Detect repeated header/footer lines across page segments.

    Args:
        page_segments (list[list[str]]): Segments represented as non-empty stripped lines.

    Returns:
        set[str]: Lower-cased repeated boundary lines to remove.

    """
    boundary_candidates: list[str] = []
    for page_lines in page_segments:
        boundary_candidates.extend(line.lower() for line in page_lines[:_HEADER_FOOTER_WINDOW])
        boundary_candidates.extend(line.lower() for line in page_lines[-_HEADER_FOOTER_WINDOW:])

    if not boundary_candidates:
        return set()

    threshold = max(2, int(len(page_segments) * _REPEATED_LINE_MIN_RATIO))
    counts = Counter(boundary_candidates)
    return {
        line
        for line, count in counts.items()
        if count >= threshold and _is_valid_repeated_boundary_line(line)
    }


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
    min_corpus_ratio: float,
    max_terms: int,
) -> set[str]:
    """Discover high-frequency stopwords directly from corpus statistics.

    Args:
        tokenized_documents (list[list[str]]): Tokenized documents.
        min_doc_ratio (float): Minimum document ratio threshold.
        min_corpus_ratio (float): Minimum token frequency ratio in full corpus.
        max_terms (int): Maximum number of generated stopwords.

    Returns:
        set[str]: Automatically discovered stopwords.

    """
    document_frequency: Counter[str] = Counter()
    collection_frequency: Counter[str] = Counter()
    total_tokens = 0

    for tokens in tokenized_documents:
        document_frequency.update(set(tokens))
        collection_frequency.update(tokens)
        total_tokens += len(tokens)

    return discover_auto_stopwords_from_frequencies(
        document_frequency=document_frequency,
        collection_frequency=collection_frequency,
        total_docs=len(tokenized_documents),
        total_tokens=total_tokens,
        min_doc_ratio=min_doc_ratio,
        min_corpus_ratio=min_corpus_ratio,
        max_terms=max_terms,
    )


def discover_auto_stopwords_from_frequencies(  # noqa: PLR0913
    *,
    document_frequency: Mapping[str, int],
    collection_frequency: Mapping[str, int],
    total_docs: int,
    total_tokens: int,
    min_doc_ratio: float,
    min_corpus_ratio: float,
    max_terms: int,
) -> set[str]:
    """Discover stopwords from precomputed corpus frequencies.

    Args:
        document_frequency (Mapping[str, int]): Per-token document frequency.
        collection_frequency (Mapping[str, int]): Per-token corpus frequency.
        total_docs (int): Number of tokenized documents.
        total_tokens (int): Total number of tokens in corpus.
        min_doc_ratio (float): Minimum document ratio threshold.
        min_corpus_ratio (float): Minimum token ratio threshold.
        max_terms (int): Maximum number of generated stopwords.

    Returns:
        set[str]: Automatically discovered stopwords.

    """
    if total_docs <= 0 or total_tokens <= 0:
        return set()

    candidates = [
        term
        for term, count in document_frequency.items()
        if len(term) > 1
        and not term.isdigit()
        and count / total_docs >= min_doc_ratio
        and collection_frequency.get(term, 0) / total_tokens >= min_corpus_ratio
    ]
    candidates.sort(
        key=lambda term: (
            -(document_frequency[term] / total_docs),
            -(collection_frequency.get(term, 0) / total_tokens),
            term,
        ),
    )
    return set(candidates[:max_terms])


def _normalize_stopword(value: str) -> str:
    """Normalize one stopword value.

    Args:
        value (str): Raw stopword value.

    Returns:
        str: Normalized lowercase stopword.

    """
    return normalize_french_accents(value.strip().lower())


def _extract_stopwords_from_yaml_data(data: object) -> set[str]:
    """Extract stopwords from parsed YAML object.

    Args:
        data (object): Parsed YAML object.

    Returns:
        set[str]: Extracted stopwords.

    """
    if data is None:
        return set()

    if isinstance(data, str):
        normalized = _normalize_stopword(data)
        return {normalized} if normalized else set()

    if isinstance(data, list):
        return {
            normalized for item in data if isinstance(item, str) and (normalized := _normalize_stopword(item))
        }

    if isinstance(data, dict):
        dict_data: dict[str, object] = {key: value for key, value in data.items() if isinstance(key, str)}
        stopwords_value = dict_data.get("stopwords")
        if isinstance(stopwords_value, list):
            return {
                normalized
                for item in stopwords_value
                if isinstance(item, str) and (normalized := _normalize_stopword(item))
            }
        values: set[str] = set()
        for value in dict_data.values():
            values |= _extract_stopwords_from_yaml_data(value)
        return values

    return set()


def _load_yaml_stopwords(path: Path) -> set[str]:
    """Load stopwords from YAML file.

    Args:
        path (Path): YAML file path.

    Returns:
        set[str]: Loaded stopwords.

    """
    text = path.read_text(encoding="utf-8")
    try:
        from yaml import safe_load  # noqa: PLC0415

        return _extract_stopwords_from_yaml_data(safe_load(text))
    except Exception:
        # Fallback parser for simple "- word" YAML lists.
        stopwords: set[str] = set()
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("-"):
                candidate = _normalize_stopword(stripped.removeprefix("-"))
                if candidate:
                    stopwords.add(candidate)
        return stopwords


def _load_csv_stopwords(path: Path) -> set[str]:
    """Load stopwords from CSV file.

    Args:
        path (Path): CSV file path.

    Returns:
        set[str]: Loaded stopwords.

    """
    stopwords: set[str] = set()
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return stopwords

    header = lines[0].lower()
    delimiter = ";" if header.count(";") > header.count(",") else ","

    reader = csv.reader(lines, delimiter=delimiter)
    rows = list(reader)
    if not rows:
        return stopwords

    header_row = [cell.strip().lower() for cell in rows[0]]
    selected_column = next(
        (idx for idx, name in enumerate(header_row) if name in _default_stopword_column_names()),
        0,
    )
    start_index = 1 if any(name in _default_stopword_column_names() for name in header_row) else 0

    for row in rows[start_index:]:
        if not row or selected_column >= len(row):
            continue
        normalized = _normalize_stopword(row[selected_column])
        if normalized and normalized not in _default_stopword_column_names():
            stopwords.add(normalized)
    return stopwords


def load_stopwords_from_file(path: Path) -> set[str]:
    """Load stopwords from one file path.

    Supported formats are `.yaml`/`.yml`, `.csv`, and plain text.

    Args:
        path (Path): Input stopwords file path.

    Returns:
        set[str]: Loaded stopwords.

    """
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return _load_yaml_stopwords(path)
    if suffix == ".csv":
        return _load_csv_stopwords(path)

    text = path.read_text(encoding="utf-8")
    return {normalized for line in text.splitlines() if (normalized := _normalize_stopword(line))}


def load_stopwords_from_files(paths: list[Path]) -> set[str]:
    """Load stopwords from multiple files.

    Args:
        paths (list[Path]): Input file paths.

    Returns:
        set[str]: Union of all loaded stopwords.

    """
    merged: set[str] = set()
    for path in paths:
        merged |= load_stopwords_from_file(path)
    return merged


def _load_nltk_stopwords(language: str) -> set[str]:
    """Load stopwords from NLTK corpus if available.

    Args:
        language (str): NLTK language key.

    Returns:
        set[str]: Loaded stopwords.

    """
    try:
        from nltk.corpus import stopwords  # type: ignore[import-not-found]  # noqa: PLC0415

        return {_normalize_stopword(word) for word in stopwords.words(language)}
    except Exception:
        return set()


@lru_cache(maxsize=1)
def get_default_stopwords() -> set[str]:
    """Get default French and English stopwords.

    Priority:
    1. NLTK corpus (`french`, `english`) when available locally.
    2. Fallback sets (scikit-learn english + bundled french fallback).

    Returns:
        set[str]: Default FR/EN stopwords.

    """
    french = _load_nltk_stopwords("french")
    english = _load_nltk_stopwords("english")

    if not english:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS  # noqa: PLC0415

        english = {_normalize_stopword(word) for word in ENGLISH_STOP_WORDS}
    if not french:
        french = _french_stopwords_fallback()

    return french | english


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

    if options & CleaningOptionFlag.HEADER_FOOTER:
        output = suppress_headers_footers(output)

    if options & CleaningOptionFlag.ACCENT_NORMALIZATION:
        output = normalize_french_accents(output)

    if options & CleaningOptionFlag.TOKEN_CLEANUP:
        output = " ".join(tokenize_for_ingestion(output))

    if options & CleaningOptionFlag.WHITESPACE:
        output = normalize_whitespace(output)

    return output
