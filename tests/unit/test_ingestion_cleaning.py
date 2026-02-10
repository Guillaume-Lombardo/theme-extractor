from __future__ import annotations

from theme_extractor.domain import CleaningOptionFlag
from theme_extractor.ingestion.cleaning import (
    apply_cleaning_options,
    discover_auto_stopwords,
    normalize_french_accents,
    normalize_whitespace,
    suppress_headers_footers,
    tokenize_for_ingestion,
)


def test_normalize_whitespace_compacts_lines() -> None:
    text = "  Bonjour   le   monde \n\n  Ceci  est   un test  "
    assert normalize_whitespace(text) == "Bonjour le monde\nCeci est un test"


def test_normalize_french_accents() -> None:
    assert normalize_french_accents("économie très sûre") == "economie tres sure"


def test_suppress_headers_footers_removes_page_lines() -> None:
    text = "Header\nPage 1/2\nCorps\nFooter\nHeader\nPage 2/2\nCorps\nFooter"
    out = suppress_headers_footers(text)
    assert "page 1/2" not in out.lower()
    assert "page 2/2" not in out.lower()


def test_tokenize_for_ingestion_lowercases_words() -> None:
    assert tokenize_for_ingestion("Banque de France") == ["banque", "de", "france"]


def test_discover_auto_stopwords_from_doc_ratio() -> None:
    docs = [["alpha", "beta"], ["alpha", "gamma"], ["alpha", "delta"]]
    auto = discover_auto_stopwords(docs, min_doc_ratio=1.0, max_terms=10)
    assert auto == {"alpha"}


def test_apply_cleaning_options_combines_flags() -> None:
    text = "<p>Résumé   https://a.b</p>"
    options = (
        CleaningOptionFlag.HTML_STRIP
        | CleaningOptionFlag.BOILERPLATE
        | CleaningOptionFlag.ACCENT_NORMALIZATION
        | CleaningOptionFlag.WHITESPACE
    )
    out = apply_cleaning_options(text, options=options)
    assert out == "Resume"
