"""File extractors for supported ingestion file types."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any, Protocol

from theme_extractor.errors import MissingOptionalDependencyError

if TYPE_CHECKING:
    from pathlib import Path

_SUPPORTED_SUFFIXES = {
    ".txt",
    ".md",
    ".html",
    ".htm",
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".msg",
}
_PDF_DEPENDENCY_MSG = "Install 'pymupdf' to ingest PDF files."
_DOCX_DEPENDENCY_MSG = "Install 'python-docx' to ingest DOCX files."
_XLSX_DEPENDENCY_MSG = "Install 'openpyxl' to ingest XLSX files."
_PPTX_DEPENDENCY_MSG = "Install 'python-pptx' to ingest PPTX files."
_MSG_DEPENDENCY_MSG = "Install 'extract-msg' to ingest .msg files."


@dataclass(frozen=True)
class PdfOcrOptions:
    """Represent optional OCR fallback settings for PDF extraction."""

    fallback_enabled: bool = False
    languages: str = "fra+eng"
    dpi: int = 200
    min_chars: int = 32
    tessdata: str | None = None


class _PdfPageProtocol(Protocol):
    """Represent the subset of PyMuPDF page APIs used by OCR fallback."""

    def get_text(self, *args: object, **kwargs: object) -> str: ...

    def get_textpage_ocr(self, **kwargs: object) -> object: ...


def supported_suffixes() -> set[str]:
    """Return supported ingestion file suffixes.

    Returns:
        set[str]: Supported file suffixes.

    """
    return set(_SUPPORTED_SUFFIXES)


def extract_text(
    path: Path,
    *,
    pdf_ocr: PdfOcrOptions | None = None,
) -> str:
    """Extract text content from one file.

    Args:
        path (Path): Input file path.
        pdf_ocr (PdfOcrOptions | None): OCR fallback settings for scanned PDFs.

    Raises:
        ValueError: If the file suffix is unsupported.

    Returns:
        str: Extracted text.

    """
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md", ".html", ".htm"}:
        return path.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".pdf":
        return _extract_pdf(
            path,
            pdf_ocr=PdfOcrOptions() if pdf_ocr is None else pdf_ocr,
        )

    if suffix in {".doc", ".docx"}:
        return _extract_docx(path)

    if suffix in {".xls", ".xlsx"}:
        return _extract_xlsx(path)

    if suffix in {".ppt", ".pptx"}:
        return _extract_pptx(path)

    if suffix == ".msg":
        return _extract_msg(path)

    raise ValueError(f"Unsupported file type: {suffix}")  # noqa: TRY003


def _extract_pdf(
    path: Path,
    *,
    pdf_ocr: PdfOcrOptions,
) -> str:
    """Extract text from PDF.

    Args:
        path (Path): Input PDF path.
        pdf_ocr (PdfOcrOptions): OCR fallback settings for scanned PDFs.

    Raises:
        MissingOptionalDependencyError: If `pymupdf` is missing.

    Returns:
        str: Extracted text.

    """
    try:
        pymupdf = import_module("pymupdf")
    except ImportError as exc:
        raise MissingOptionalDependencyError(_PDF_DEPENDENCY_MSG) from exc

    document = pymupdf.Document(str(path))
    pages: list[str] = []
    for page in document:
        plain_text = str(page.get_text() or "")
        if not pdf_ocr.fallback_enabled or _has_enough_pdf_text(plain_text, min_chars=pdf_ocr.min_chars):
            pages.append(plain_text)
            continue

        ocr_text = _extract_pdf_page_with_ocr(
            page=page,
            pdf_ocr=pdf_ocr,
        )
        if _has_enough_pdf_text(ocr_text, min_chars=pdf_ocr.min_chars) or not plain_text.strip():
            pages.append(ocr_text)
        else:
            pages.append(plain_text)
    return "\n\n".join(page.strip() for page in pages if page.strip())


def _has_enough_pdf_text(text: str, *, min_chars: int) -> bool:
    """Check whether extracted PDF text is substantial enough to skip OCR fallback.

    Args:
        text (str): Extracted page text.
        min_chars (int): Minimum alphanumeric character threshold.

    Returns:
        bool: True when text volume is sufficient.

    """
    alnum_count = sum(1 for char in text if char.isalnum())
    return alnum_count >= max(min_chars, 1)


def _extract_pdf_page_with_ocr(
    *,
    page: _PdfPageProtocol,
    pdf_ocr: PdfOcrOptions,
) -> str:
    """Extract one PDF page with OCR when PyMuPDF OCR APIs are available.

    Args:
        page (object): One PyMuPDF page object.
        pdf_ocr (PdfOcrOptions): OCR fallback settings.

    Returns:
        str: OCR-extracted text, or an empty string if OCR is unavailable/fails.

    """
    get_textpage_ocr = getattr(page, "get_textpage_ocr", None)
    if not callable(get_textpage_ocr):
        return ""

    kwargs: dict[str, Any] = {
        "language": pdf_ocr.languages,
        "dpi": pdf_ocr.dpi,
    }
    if pdf_ocr.tessdata:
        kwargs["tessdata"] = pdf_ocr.tessdata

    try:
        textpage = get_textpage_ocr(**kwargs)
    except Exception:
        return ""

    try:
        return str(page.get_text("text", textpage=textpage) or "")
    except TypeError:
        try:
            return str(page.get_text(textpage=textpage) or "")
        except Exception:
            return ""
    except Exception:
        return ""


def _extract_docx(path: Path) -> str:
    """Extract text from DOC/DOCX.

    Args:
        path (Path): Input document path.

    Raises:
        MissingOptionalDependencyError: If `python-docx` is missing.

    Returns:
        str: Extracted text.

    """
    try:
        docx = import_module("docx")
    except ImportError as exc:
        raise MissingOptionalDependencyError(_DOCX_DEPENDENCY_MSG) from exc

    document = docx.Document(str(path))
    return "\n".join(paragraph.text for paragraph in document.paragraphs if paragraph.text)


def _extract_xlsx(path: Path) -> str:
    """Extract text from XLS/XLSX.

    Args:
        path (Path): Input spreadsheet path.

    Raises:
        MissingOptionalDependencyError: If `openpyxl` is missing.

    Returns:
        str: Extracted text.

    """
    try:
        openpyxl = import_module("openpyxl")
    except ImportError as exc:
        raise MissingOptionalDependencyError(_XLSX_DEPENDENCY_MSG) from exc

    workbook = openpyxl.load_workbook(filename=str(path), read_only=True, data_only=True)
    lines: list[str] = []
    for sheet in workbook.worksheets:
        lines.append(f"== sheet: {sheet.title} ==")
        for row in sheet.iter_rows(values_only=True):
            values = ["" if cell is None else str(cell) for cell in row]
            if any(values):
                lines.append("\t".join(values))
    return "\n".join(lines)


def _extract_pptx(path: Path) -> str:
    """Extract text from PPT/PPTX.

    Args:
        path (Path): Input presentation path.

    Raises:
        MissingOptionalDependencyError: If `python-pptx` is missing.

    Returns:
        str: Extracted text.

    """
    try:
        pptx = import_module("pptx")
    except ImportError as exc:
        raise MissingOptionalDependencyError(_PPTX_DEPENDENCY_MSG) from exc

    presentation = pptx.Presentation(str(path))
    lines: list[str] = []
    for slide_index, slide in enumerate(presentation.slides, start=1):
        lines.append(f"== slide {slide_index} ==")
        for shape in slide.shapes:
            text = getattr(shape, "text", None)
            if text:
                lines.append(str(text))
    return "\n".join(lines)


def _extract_msg(path: Path) -> str:
    """Extract text from MSG email files.

    Args:
        path (Path): Input `.msg` path.

    Raises:
        MissingOptionalDependencyError: If `extract-msg` is missing.

    Returns:
        str: Extracted email text.

    """
    try:
        extract_msg = import_module("extract_msg")
    except ImportError as exc:
        raise MissingOptionalDependencyError(_MSG_DEPENDENCY_MSG) from exc

    message = extract_msg.Message(str(path))
    return str(message.body or "")
