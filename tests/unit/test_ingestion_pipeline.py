from __future__ import annotations

from theme_extractor.domain import CleaningOptionFlag
from theme_extractor.ingestion import IngestionConfig, run_ingestion

_EXPECTED_TOKEN_COUNT = 2


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
