from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import pytest

from theme_extractor.domain import CleaningOptionFlag
from theme_extractor.ingestion import extractors
from theme_extractor.ingestion.cleaning import apply_cleaning_options
from theme_extractor.ingestion.extractors import PdfOcrOptions, extract_text, supported_suffixes


class _PdfPage:
    @staticmethod
    def get_text() -> str:
        return "pdf text"


class _PdfScannedPage:
    @staticmethod
    def get_text(*args, **kwargs) -> str:  # noqa: ANN002, ANN003
        if "textpage" in kwargs or (args and args[0] == "text"):
            return "ocr extracted text"
        return ""

    @staticmethod
    def get_textpage_ocr(*, language: str, dpi: int, tessdata: str | None = None) -> object:
        _ = language
        _ = dpi
        _ = tessdata
        return object()


class _PdfDoc:
    def __init__(self, _path: str):
        self._pages = [_PdfPage()]

    def __iter__(self) -> object:
        return iter(self._pages)


class _DocxDocument:
    def __init__(self, _path: str):
        self.paragraphs = [SimpleNamespace(text="docx text")]


class _Sheet:
    title = "Sheet1"

    @staticmethod
    def iter_rows(values_only: bool = True) -> list[tuple[str, str]]:  # noqa: FBT001, FBT002, ARG004
        return [("x", "y")]


class _Workbook:
    worksheets: ClassVar[list[_Sheet]] = [_Sheet()]


class _Slide:
    shapes: ClassVar[list[SimpleNamespace]] = [SimpleNamespace(text="slide text")]


class _Presentation:
    def __init__(self, _path: str):
        self.slides = [_Slide()]


class _Message:
    def __init__(self, _path: str):
        self.body = "mail body"


def _import_module_success(name: str) -> object:
    if name == "pymupdf":
        return SimpleNamespace(Document=_PdfDoc)
    if name == "docx":
        return SimpleNamespace(Document=_DocxDocument)
    if name == "openpyxl":
        return SimpleNamespace(load_workbook=lambda **_kwargs: _Workbook())
    if name == "pptx":
        return SimpleNamespace(Presentation=_Presentation)
    if name == "extract_msg":
        return SimpleNamespace(Message=_Message)
    raise ImportError(name)


def _import_module_pdf_scanned(name: str) -> object:
    class _ScannedPdfDoc:
        def __init__(self, _path: str):
            self._pages = [_PdfScannedPage()]

        def __iter__(self) -> object:
            return iter(self._pages)

    if name == "pymupdf":
        return SimpleNamespace(Document=_ScannedPdfDoc)
    raise ImportError(name)


def test_supported_suffixes_contains_expected_formats() -> None:
    suffixes = supported_suffixes()
    assert ".txt" in suffixes
    assert ".pdf" in suffixes
    assert ".msg" in suffixes


def test_extract_text_for_plain_and_html(tmp_path) -> None:
    txt = tmp_path / "a.txt"
    html = tmp_path / "b.html"
    txt.write_text("Bonjour", encoding="utf-8")
    html.write_text("<p>Résumé</p>", encoding="utf-8")

    assert extract_text(txt) == "Bonjour"
    assert "Résumé" in extract_text(html)


def test_extract_text_raises_for_unknown_extension(tmp_path) -> None:
    unknown = tmp_path / "a.unknown"
    unknown.write_text("x", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        extract_text(unknown)


def test_extract_text_optional_parsers_missing_dependency(tmp_path, monkeypatch) -> None:
    pdf = tmp_path / "a.pdf"
    pdf.write_text("placeholder", encoding="utf-8")

    monkeypatch.setattr(extractors, "import_module", lambda _: (_ for _ in ()).throw(ImportError("missing")))

    with pytest.raises(ImportError):
        extract_text(pdf)


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("a.pdf", "pdf text"),
        ("a.docx", "docx text"),
        ("a.xlsx", "sheet"),
        ("a.pptx", "slide text"),
        ("a.msg", "mail body"),
    ],
)
def test_extract_text_optional_parsers_success(tmp_path, monkeypatch, filename: str, expected: str) -> None:
    monkeypatch.setattr(extractors, "import_module", _import_module_success)

    file_path = tmp_path / filename
    file_path.write_text("placeholder", encoding="utf-8")

    assert expected in extract_text(file_path).lower()


def test_extract_text_pdf_ocr_fallback_enabled(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(extractors, "import_module", _import_module_pdf_scanned)

    pdf = tmp_path / "scan.pdf"
    pdf.write_text("placeholder", encoding="utf-8")

    out = extract_text(
        pdf,
        pdf_ocr=PdfOcrOptions(
            fallback_enabled=True,
            languages="fra+eng",
            dpi=200,
            min_chars=1,
        ),
    )

    assert out == "ocr extracted text"


def test_apply_cleaning_options_header_footer_and_boilerplate() -> None:
    text = "Header\nPage 1/2\nRésumé https://example.com\nFooter\nHeader\nPage 2/2\nRésumé\nFooter"
    options = (
        CleaningOptionFlag.WHITESPACE
        | CleaningOptionFlag.HEADER_FOOTER
        | CleaningOptionFlag.BOILERPLATE
        | CleaningOptionFlag.ACCENT_NORMALIZATION
    )
    out = apply_cleaning_options(text, options=options)
    assert "page" not in out.lower()
    assert "https" not in out.lower()
    assert "header" in out.lower()
