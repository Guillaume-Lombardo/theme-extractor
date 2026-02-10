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


def _import_module_pdf(name: str) -> object:
    if name == "pymupdf":
        return SimpleNamespace(Document=_PdfDoc)
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


def test_ingest_end2end_optional_format_with_stubbed_pdf_parser(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(extractors, "import_module", _import_module_pdf)

    pdf = tmp_path / "doc.pdf"
    pdf.write_text("placeholder", encoding="utf-8")

    exit_code = main(["ingest", "--input", str(pdf)])
    assert exit_code == 0


def test_ingest_end2end_optional_formats_with_stubs(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(extractors, "import_module", _import_module_office)

    for filename in ("a.docx", "a.xlsx", "a.pptx", "a.msg"):
        file_path = tmp_path / filename
        file_path.write_text("placeholder", encoding="utf-8")
        exit_code = main(["ingest", "--input", str(file_path)])
        assert exit_code == 0
