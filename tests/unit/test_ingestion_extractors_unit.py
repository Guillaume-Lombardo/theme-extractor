from __future__ import annotations

from subprocess import CompletedProcess  # noqa: S404
from types import SimpleNamespace
from typing import Any, cast

import pytest

from theme_extractor.domain import MsgAttachmentPolicy
from theme_extractor.errors import MissingOptionalDependencyError
from theme_extractor.ingestion import extractors
from theme_extractor.ingestion.extractors import MsgExtractionOptions, PdfOcrOptions, extract_text


def test_extract_text_plain_file(tmp_path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("bonjour", encoding="utf-8")
    assert extract_text(path) == "bonjour"


def test_extract_text_rejects_legacy_doc(tmp_path) -> None:
    path = tmp_path / "legacy.doc"
    path.write_text("x", encoding="utf-8")

    with pytest.raises(ValueError, match=r"Legacy \.doc format is not supported"):
        extract_text(path)


def test_extract_text_pdf_missing_optional_dependency(tmp_path, monkeypatch) -> None:
    path = tmp_path / "sample.pdf"
    path.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(
        extractors,
        "import_module",
        lambda _name: (_ for _ in ()).throw(ImportError("missing")),
    )

    with pytest.raises(MissingOptionalDependencyError, match="pymupdf"):
        extract_text(path)


def test_normalize_ocr_languages_deduplicates_and_strips() -> None:
    assert extractors._normalize_ocr_languages(" fra + eng +fra+ ") == ("fra", "eng")


def test_resolve_ocr_languages_with_unavailable_tesseract(monkeypatch) -> None:
    monkeypatch.setattr(extractors, "_available_tesseract_languages", lambda _tessdata: None)
    assert extractors._resolve_ocr_languages(languages="fra+eng", tessdata=None) == "fra+eng"


def test_resolve_ocr_languages_falls_back_to_english(monkeypatch) -> None:
    monkeypatch.setattr(extractors, "_available_tesseract_languages", lambda _tessdata: {"eng", "osd"})
    assert extractors._resolve_ocr_languages(languages="fra+deu", tessdata=None) == "eng"


def test_resolve_ocr_languages_preserves_requested_when_no_match_and_no_english(monkeypatch) -> None:
    monkeypatch.setattr(extractors, "_available_tesseract_languages", lambda _tessdata: {"osd"})
    assert extractors._resolve_ocr_languages(languages="fra+deu", tessdata=None) == "fra+deu"


def test_has_enough_pdf_text_counts_alnum() -> None:
    assert extractors._has_enough_pdf_text("abc123", min_chars=6) is True
    assert extractors._has_enough_pdf_text("a !", min_chars=2) is False
    assert extractors._has_enough_pdf_text("", min_chars=0) is False


class _OcrPageNoMethod:
    @staticmethod
    def get_text(*_args: object, **_kwargs: object) -> str:
        return "ignored"


def test_extract_pdf_page_with_ocr_returns_empty_when_ocr_api_missing() -> None:
    text = extractors._extract_pdf_page_with_ocr(
        page=cast("Any", _OcrPageNoMethod()),
        pdf_ocr=PdfOcrOptions(fallback_enabled=True),
    )
    assert not text


class _OcrPageTextpageArg:
    @staticmethod
    def get_textpage_ocr(**_kwargs: object) -> object:
        return object()

    @staticmethod
    def get_text(*args: object, **kwargs: object) -> str:
        if args and args[0] == "text" and "textpage" in kwargs:
            return "ocr text"
        return ""


def test_extract_pdf_page_with_ocr_happy_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(extractors, "_resolve_ocr_languages", lambda **_kwargs: "eng")
    text = extractors._extract_pdf_page_with_ocr(
        page=_OcrPageTextpageArg(),
        pdf_ocr=PdfOcrOptions(fallback_enabled=True, tessdata=str(tmp_path / "tessdata")),
    )
    assert text == "ocr text"


class _OcrPageNeedsFallbackSignature:
    @staticmethod
    def get_textpage_ocr(**_kwargs: object) -> object:
        return object()

    @staticmethod
    def get_text(*_args: object, **kwargs: object) -> str:
        if kwargs:
            return "ocr text via fallback"
        raise TypeError


def test_extract_pdf_page_with_ocr_uses_fallback_call_signature(monkeypatch) -> None:
    monkeypatch.setattr(extractors, "_resolve_ocr_languages", lambda **_kwargs: "eng")
    text = extractors._extract_pdf_page_with_ocr(
        page=_OcrPageNeedsFallbackSignature(),
        pdf_ocr=PdfOcrOptions(fallback_enabled=True),
    )
    assert text == "ocr text via fallback"


def test_extract_pdf_page_with_ocr_returns_empty_when_ocr_raises(monkeypatch) -> None:
    class _FailingPage:
        @staticmethod
        def get_textpage_ocr(**_kwargs: object) -> object:
            raise RuntimeError

        @staticmethod
        def get_text(*_args: object, **_kwargs: object) -> str:
            return ""

    monkeypatch.setattr(extractors, "_resolve_ocr_languages", lambda **_kwargs: "eng")
    text = extractors._extract_pdf_page_with_ocr(
        page=_FailingPage(),
        pdf_ocr=PdfOcrOptions(fallback_enabled=True),
    )
    assert not text


def test_available_tesseract_languages_parses_output(monkeypatch) -> None:
    def _fake_run(*_args: object, **_kwargs: object) -> CompletedProcess[str]:
        stdout = "List of available languages in /opt/tessdata (2):\neng\nfra\n"
        return CompletedProcess(args=["tesseract", "--list-langs"], returncode=0, stdout=stdout, stderr="")

    extractors._available_tesseract_languages.cache_clear()
    monkeypatch.setattr(extractors.subprocess, "run", _fake_run)
    assert extractors._available_tesseract_languages(None) == {"eng", "fra"}


def test_available_tesseract_languages_returns_none_on_errors(monkeypatch) -> None:
    extractors._available_tesseract_languages.cache_clear()
    monkeypatch.setattr(
        extractors.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError()),
    )
    assert extractors._available_tesseract_languages(None) is None


def test_msg_attachment_name_prioritizes_first_non_empty_attribute() -> None:
    attachment = SimpleNamespace(
        longFilename="",
        filename="report.txt",
        displayName="ignored",
        name="ignored",
    )
    assert extractors._msg_attachment_name(attachment) == "report.txt"


def test_msg_attachment_text_handles_str_bytes_and_limits() -> None:
    assert extractors._msg_attachment_text(SimpleNamespace(data=" hello ")) == "hello"
    assert extractors._msg_attachment_text(SimpleNamespace(data=b"hello")) == "hello"
    assert not extractors._msg_attachment_text(SimpleNamespace(data=b"\x00binary"))
    assert not extractors._msg_attachment_text(SimpleNamespace(data=b"a" * 600_000))
    assert not extractors._msg_attachment_text(SimpleNamespace(data="a" * 101_000))


def test_extract_msg_attachments_text_with_policy_names_and_text() -> None:
    attachments = [
        SimpleNamespace(longFilename="notes.txt", data=b"content"),
        SimpleNamespace(longFilename="", filename="", data="standalone text"),
    ]
    names_lines = extractors._extract_msg_attachments_text(
        attachments=attachments,
        policy=MsgAttachmentPolicy.NAMES,
    )
    text_lines = extractors._extract_msg_attachments_text(
        attachments=attachments,
        policy=MsgAttachmentPolicy.TEXT,
    )

    assert names_lines == ["attachment: notes.txt"]
    assert "attachment: notes.txt" in text_lines
    assert "content" in text_lines
    assert "standalone text" in text_lines


def test_extract_text_msg_without_metadata(tmp_path, monkeypatch) -> None:
    class _Message:
        def __init__(self, _path: str):
            self.body = "mail body"
            self.subject = "Should be hidden"
            self.sender = "sender@example.com"
            self.to = "team@example.com"
            self.cc = ""
            self.date = "2026-02-14"
            self.attachments = []

    monkeypatch.setattr(extractors, "import_module", lambda _name: SimpleNamespace(Message=_Message))

    path = tmp_path / "mail.msg"
    path.write_text("placeholder", encoding="utf-8")

    out = extract_text(
        path,
        msg_options=MsgExtractionOptions(
            include_metadata=False,
            attachments_policy=MsgAttachmentPolicy.NONE,
        ),
    )
    assert out == "mail body"
