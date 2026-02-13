from __future__ import annotations

from theme_extractor.domain import CleaningOptionFlag
from theme_extractor.ingestion import IngestionConfig, run_ingestion
from theme_extractor.ingestion import pipeline as pipeline_mod
from theme_extractor.ingestion.extractors import PdfOcrOptions

_EXPECTED_TOKEN_COUNT = 2
_EXPECTED_TOKEN_COUNT_WITH_FILE_STOPWORD = 2
_EXPECTED_REMOVED_STOPWORDS = 2
_EXPECTED_OCR_DPI = 300
_EXPECTED_OCR_MIN_CHARS = 5
_EXPECTED_TESSDATA_PATH = "/opt/tessdata"


def test_run_ingestion_emits_compact_document_metadata(tmp_path) -> None:
    text = "Résumé alpha beta"
    sample = tmp_path / "sample.txt"
    sample.write_text(text, encoding="utf-8")

    result = run_ingestion(
        IngestionConfig(
            input_path=sample,
            cleaning_options=CleaningOptionFlag.NONE,
            manual_stopwords={"alpha"},
        ),
    )

    assert result.processed_documents == 1
    assert result.skipped_documents == 0

    doc = result.documents[0]
    assert doc.raw_length == len(text)
    assert doc.clean_length == len(text)
    assert doc.clean_text_preview == text
    assert doc.removed_stopword_count == 1
    assert doc.token_count == _EXPECTED_TOKEN_COUNT
    assert result.streaming_mode is True


def test_run_ingestion_loads_manual_stopwords_from_file(tmp_path) -> None:
    sample = tmp_path / "sample.txt"
    stopwords_file = tmp_path / "stopwords.yaml"
    sample.write_text("alpha beta gamma", encoding="utf-8")
    stopwords_file.write_text("stopwords:\n  - beta\n", encoding="utf-8")

    result = run_ingestion(
        IngestionConfig(
            input_path=sample,
            cleaning_options=CleaningOptionFlag.NONE,
            manual_stopwords_files=[stopwords_file],
            default_stopwords_enabled=False,
        ),
    )

    assert result.processed_documents == 1
    doc = result.documents[0]
    assert doc.token_count == _EXPECTED_TOKEN_COUNT_WITH_FILE_STOPWORD
    assert doc.removed_stopword_count == 1
    assert result.manual_stopwords == ["beta"]
    assert result.manual_stopwords_files == [str(stopwords_file)]


def test_run_ingestion_non_streaming_mode_keeps_same_counts(tmp_path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("alpha alpha beta", encoding="utf-8")

    result = run_ingestion(
        IngestionConfig(
            input_path=sample,
            cleaning_options=CleaningOptionFlag.NONE,
            manual_stopwords={"alpha"},
            default_stopwords_enabled=False,
            streaming_mode=False,
        ),
    )

    assert result.streaming_mode is False
    doc = result.documents[0]
    assert doc.token_count == 1
    assert doc.removed_stopword_count == _EXPECTED_REMOVED_STOPWORDS


def test_run_ingestion_propagates_pdf_ocr_settings(tmp_path, monkeypatch) -> None:
    sample = tmp_path / "scan.pdf"
    sample.write_text("placeholder", encoding="utf-8")
    captured_kwargs: dict[str, object] = {}

    def _fake_extract_text(path: object, **kwargs: object) -> str:
        _ = path
        captured_kwargs.update(kwargs)
        return "texte ocr"

    monkeypatch.setattr(pipeline_mod, "extract_text", _fake_extract_text)

    result = run_ingestion(
        IngestionConfig(
            input_path=sample,
            cleaning_options=CleaningOptionFlag.NONE,
            default_stopwords_enabled=False,
            pdf_ocr_fallback=True,
            pdf_ocr_languages="fra+eng",
            pdf_ocr_dpi=_EXPECTED_OCR_DPI,
            pdf_ocr_min_chars=_EXPECTED_OCR_MIN_CHARS,
            pdf_ocr_tessdata=_EXPECTED_TESSDATA_PATH,
        ),
    )

    assert result.processed_documents == 1
    assert result.pdf_ocr_fallback is True
    assert result.pdf_ocr_languages == "fra+eng"
    assert result.pdf_ocr_dpi == _EXPECTED_OCR_DPI
    assert result.pdf_ocr_min_chars == _EXPECTED_OCR_MIN_CHARS
    assert result.pdf_ocr_tessdata == _EXPECTED_TESSDATA_PATH
    pdf_ocr = captured_kwargs["pdf_ocr"]
    assert isinstance(pdf_ocr, PdfOcrOptions)
    assert pdf_ocr.fallback_enabled is True
    assert pdf_ocr.languages == "fra+eng"
    assert pdf_ocr.dpi == _EXPECTED_OCR_DPI
    assert pdf_ocr.min_chars == _EXPECTED_OCR_MIN_CHARS
    assert pdf_ocr.tessdata == _EXPECTED_TESSDATA_PATH
