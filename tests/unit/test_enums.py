from __future__ import annotations

import pytest

from theme_extractor.domain import (
    CleaningOptionFlag,
    ExtractMethod,
    ExtractMethodFlag,
    cleaning_flag_from_string,
    cleaning_flag_to_string,
    default_cleaning_options,
    method_flag_from_string,
    method_flag_to_methods,
    method_flag_to_string,
    parse_extract_method,
)
from theme_extractor.errors import UnsupportedMethodError


def test_parse_extract_method_returns_enum_value() -> None:
    assert parse_extract_method("keybert") == ExtractMethod.KEYBERT


def test_parse_extract_method_raises_for_unknown_value() -> None:
    with pytest.raises(UnsupportedMethodError, match="Unsupported extraction method"):
        parse_extract_method("unknown")


def test_method_flag_from_string_combines_unique_values() -> None:
    flag = method_flag_from_string("keybert,keybert,llm")

    assert flag & ExtractMethodFlag.KEYBERT
    assert flag & ExtractMethodFlag.LLM


def test_method_flag_from_string_raises_on_empty_input() -> None:
    with pytest.raises(ValueError, match="At least one extraction method"):
        method_flag_from_string(",,,")


def test_method_flag_to_methods_returns_stable_order() -> None:
    flag = ExtractMethodFlag.LLM | ExtractMethodFlag.BASELINE_TFIDF | ExtractMethodFlag.TERMS
    methods = method_flag_to_methods(flag)

    assert methods == [ExtractMethod.BASELINE_TFIDF, ExtractMethod.TERMS, ExtractMethod.LLM]


def test_method_flag_to_string_returns_canonical_csv() -> None:
    flag = ExtractMethodFlag.BERTOPIC | ExtractMethodFlag.SIGNIFICANT_TEXT
    serialized = method_flag_to_string(flag)

    assert serialized == "significant_text,bertopic"


def test_cleaning_flag_from_string_supports_all_keyword() -> None:
    assert cleaning_flag_from_string("all") == default_cleaning_options()


def test_cleaning_flag_from_string_supports_none_keyword() -> None:
    assert cleaning_flag_from_string("none") == CleaningOptionFlag.NONE


def test_cleaning_flag_from_string_combines_values() -> None:
    flag = cleaning_flag_from_string("whitespace,boilerplate,html_strip")
    assert flag & CleaningOptionFlag.WHITESPACE
    assert flag & CleaningOptionFlag.BOILERPLATE
    assert flag & CleaningOptionFlag.HTML_STRIP


def test_cleaning_flag_from_string_rejects_none_with_other_values() -> None:
    with pytest.raises(ValueError, match="'none' cannot be combined"):
        cleaning_flag_from_string("none,whitespace")


def test_cleaning_flag_from_string_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="Unsupported cleaning option"):
        cleaning_flag_from_string("unknown")


def test_cleaning_flag_to_string_returns_canonical_names() -> None:
    flag = CleaningOptionFlag.ACCENT_NORMALIZATION | CleaningOptionFlag.TOKEN_CLEANUP
    assert cleaning_flag_to_string(flag) == "accent_normalization,token_cleanup"


def test_cleaning_flag_none_roundtrip() -> None:
    serialized = cleaning_flag_to_string(CleaningOptionFlag.NONE)
    assert serialized == "none"
    assert cleaning_flag_from_string(serialized) == CleaningOptionFlag.NONE
