from __future__ import annotations

import json
from types import SimpleNamespace
from typing import ClassVar

from theme_extractor.cli import main
from theme_extractor.ingestion import extractors


class _PdfPage:
    @staticmethod
    def get_text() -> str:
        return "synthetic pdf text"


class _PdfScannedPage:
    @staticmethod
    def get_text(*args, **kwargs) -> str:  # noqa: ANN002, ANN003
        if "textpage" in kwargs or (args and args[0] == "text"):
            return "ocr synthetic text"
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
        self.subject = "Subject line"
        self.sender = "sender@example.com"
        self.to = "team@example.com"
        self.cc = ""
        self.date = "2026-02-13"
        self.attachments = [
            SimpleNamespace(longFilename="notes.txt", data=b"attachment body"),
        ]


def _import_module_pdf(name: str) -> object:
    if name == "pymupdf":
        return SimpleNamespace(Document=_PdfDoc)
    raise ImportError(name)


def _import_module_pdf_scanned(name: str) -> object:
    class _PdfScannedDoc:
        def __init__(self, _path: str):
            self._pages = [_PdfScannedPage()]

        def __iter__(self) -> object:
            return iter(self._pages)

    if name == "pymupdf":
        return SimpleNamespace(Document=_PdfScannedDoc)
    raise ImportError(name)


def _import_module_office(name: str) -> object:
    if name == "docx":
        return SimpleNamespace(Document=_DocxDocument)
    if name == "openpyxl":
        return SimpleNamespace(load_workbook=lambda **_kwargs: _Workbook())
    if name == "pptx":
        return SimpleNamespace(Presentation=_Presentation)
    if name == "extract_msg":
        return SimpleNamespace(Message=_Message)
    raise ImportError(name)


def test_ingest_end2end_html_cleaning_and_stopwords(tmp_path) -> None:
    html = tmp_path / "corpus.html"
    html.write_text("<html><body>Résumé alpha alpha beta https://example.com</body></html>", encoding="utf-8")
    out = tmp_path / "out.json"

    exit_code = main(
        [
            "ingest",
            "--input",
            str(html),
            "--manual-stopwords",
            "alpha",
            "--cleaning-options",
            "html_strip,boilerplate,accent_normalization,whitespace,token_cleanup",
            "--output",
            str(out),
        ],
    )

    assert exit_code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["processed_documents"] == 1
    preview = payload["documents"][0]["clean_text_preview"].lower()
    assert "resume" in preview
    assert "https" not in preview


def test_ingest_end2end_multipage_header_footer_suppression(tmp_path) -> None:
    sample = tmp_path / "multipage.txt"
    sample.write_text(
        (
            "Company Confidential\n"
            "Page 1/2\n"
            "Alpha body text\n"
            "Footer legal mention\n\n"
            "Company Confidential\n"
            "Page 2/2\n"
            "Beta body text\n"
            "Footer legal mention\n"
        ),
        encoding="utf-8",
    )
    out = tmp_path / "multipage.json"

    exit_code = main(
        [
            "ingest",
            "--input",
            str(sample),
            "--cleaning-options",
            "header_footer,whitespace",
            "--output",
            str(out),
        ],
    )

    assert exit_code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    preview = payload["documents"][0]["clean_text_preview"].lower()
    assert "company confidential" not in preview
    assert "footer legal mention" not in preview
    assert "page 1/2" not in preview
    assert "page 2/2" not in preview
    assert "alpha body text" in preview
    assert "beta body text" in preview


def test_ingest_end2end_optional_format_with_stubbed_pdf_parser(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(extractors, "import_module", _import_module_pdf)

    pdf = tmp_path / "doc.pdf"
    pdf.write_text("placeholder", encoding="utf-8")

    exit_code = main(["ingest", "--input", str(pdf)])
    assert exit_code == 0


def test_ingest_end2end_pdf_ocr_fallback_with_stubbed_parser(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(extractors, "import_module", _import_module_pdf_scanned)

    pdf = tmp_path / "scan.pdf"
    pdf.write_text("placeholder", encoding="utf-8")
    out = tmp_path / "ocr.json"

    exit_code = main(
        [
            "ingest",
            "--input",
            str(pdf),
            "--pdf-ocr-fallback",
            "--pdf-ocr-languages",
            "fra+eng",
            "--pdf-ocr-dpi",
            "300",
            "--output",
            str(out),
        ],
    )
    assert exit_code == 0

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["pdf_ocr_fallback"] is True
    assert payload["processed_documents"] == 1
    assert payload["documents"][0]["token_count"] > 0


def test_ingest_end2end_optional_formats_with_stubs(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(extractors, "import_module", _import_module_office)

    for filename in ("a.docx", "a.xlsx", "a.pptx", "a.msg"):
        file_path = tmp_path / filename
        file_path.write_text("placeholder", encoding="utf-8")
        exit_code = main(["ingest", "--input", str(file_path)])
        assert exit_code == 0


def test_ingest_end2end_msg_attachment_text_policy(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(extractors, "import_module", _import_module_office)

    msg_path = tmp_path / "mail.msg"
    msg_path.write_text("placeholder", encoding="utf-8")
    out = tmp_path / "mail.json"

    exit_code = main(
        [
            "ingest",
            "--input",
            str(msg_path),
            "--msg-include-metadata",
            "--msg-attachments-policy",
            "text",
            "--output",
            str(out),
        ],
    )

    assert exit_code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["msg_include_metadata"] is True
    assert payload["msg_attachments_policy"] == "text"
    assert payload["processed_documents"] == 1
    assert payload["documents"][0]["token_count"] > 0
