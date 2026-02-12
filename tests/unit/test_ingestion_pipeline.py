from __future__ import annotations

from theme_extractor.domain import CleaningOptionFlag
from theme_extractor.ingestion import IngestionConfig, run_ingestion

_EXPECTED_TOKEN_COUNT = 2
_EXPECTED_TOKEN_COUNT_WITH_FILE_STOPWORD = 2


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
