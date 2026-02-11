"""Ingestion pipeline orchestration for corpus files."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003

from pydantic import BaseModel, Field

from theme_extractor.domain import CleaningOptionFlag, cleaning_flag_to_string, default_cleaning_options
from theme_extractor.ingestion.cleaning import (
    apply_cleaning_options,
    discover_auto_stopwords,
    tokenize_for_ingestion,
)
from theme_extractor.ingestion.extractors import extract_text, supported_suffixes


class IngestionConfig(BaseModel):
    """Represent ingestion configuration options."""

    input_path: Path
    recursive: bool = True
    cleaning_options: CleaningOptionFlag = Field(default_factory=default_cleaning_options)
    manual_stopwords: set[str] = Field(default_factory=set)
    auto_stopwords_enabled: bool = False
    auto_stopwords_min_doc_ratio: float = 0.7
    auto_stopwords_max_terms: int = 200


class IngestedDocument(BaseModel):
    """Represent one ingested and cleaned document."""

    document_id: str
    path: str
    extension: str
    raw_length: int
    clean_length: int
    token_count: int
    removed_stopword_count: int
    clean_text_preview: str


class SkippedDocument(BaseModel):
    """Represent one skipped document with reason."""

    path: str
    reason: str


class IngestionRunResult(BaseModel):
    """Represent full ingestion pipeline output."""

    command: str = "ingest"
    input_path: str
    recursive: bool
    supported_suffixes: list[str]
    cleaning_options: str
    auto_stopwords_enabled: bool
    manual_stopwords: list[str]
    auto_stopwords: list[str]
    total_candidate_files: int
    processed_documents: int
    skipped_documents: int
    documents: list[IngestedDocument]
    skipped: list[SkippedDocument]


@dataclass(frozen=True, slots=True)
class _ProcessedDocumentMetadata:
    """Represent lightweight processed document metadata."""

    path: Path
    raw_length: int
    clean_length: int
    clean_text_preview: str
    tokens: list[str]


class IngestionPipeline:
    """Run ingestion and cleaning over a local corpus path."""

    def __init__(self, config: IngestionConfig) -> None:
        """Create one ingestion pipeline instance.

        Args:
            config (IngestionConfig): Pipeline configuration.

        """
        self.config = config

    def run(self) -> IngestionRunResult:
        """Execute ingestion pipeline.

        Returns:
            IngestionRunResult: Final ingestion output payload.

        """
        files = list(self._iter_candidate_files())

        tokenized_documents: list[list[str]] = []
        processed_items: list[_ProcessedDocumentMetadata] = []
        skipped: list[SkippedDocument] = []

        for file_path in files:
            try:
                raw_text = extract_text(file_path)
            except Exception as exc:
                skipped.append(SkippedDocument(path=str(file_path), reason=str(exc)))
                continue

            cleaned_text = apply_cleaning_options(
                raw_text,
                options=self.config.cleaning_options,
            )
            tokens = tokenize_for_ingestion(cleaned_text)
            tokenized_documents.append(tokens)

            processed_items.append(
                _ProcessedDocumentMetadata(
                    path=file_path,
                    raw_length=len(raw_text),
                    clean_length=len(cleaned_text),
                    clean_text_preview=cleaned_text[:200],
                    tokens=tokens,
                ),
            )

        auto_stopwords: set[str] = set()
        if self.config.auto_stopwords_enabled:
            auto_stopwords = discover_auto_stopwords(
                tokenized_documents,
                min_doc_ratio=self.config.auto_stopwords_min_doc_ratio,
                max_terms=self.config.auto_stopwords_max_terms,
            )

        stopwords = {word.lower() for word in self.config.manual_stopwords} | auto_stopwords

        documents: list[IngestedDocument] = []
        for item in processed_items:
            filtered_tokens = [token for token in item.tokens if token.lower() not in stopwords]
            removed_count = len(item.tokens) - len(filtered_tokens)
            doc_id = _document_id_for_path(item.path)

            documents.append(
                IngestedDocument(
                    document_id=doc_id,
                    path=str(item.path),
                    extension=item.path.suffix.lower(),
                    raw_length=item.raw_length,
                    clean_length=item.clean_length,
                    token_count=len(filtered_tokens),
                    removed_stopword_count=removed_count,
                    clean_text_preview=item.clean_text_preview,
                ),
            )

        return IngestionRunResult(
            input_path=str(self.config.input_path),
            recursive=self.config.recursive,
            supported_suffixes=sorted(supported_suffixes()),
            cleaning_options=cleaning_flag_to_string(self.config.cleaning_options),
            auto_stopwords_enabled=self.config.auto_stopwords_enabled,
            manual_stopwords=sorted({word.lower() for word in self.config.manual_stopwords}),
            auto_stopwords=sorted(auto_stopwords),
            total_candidate_files=len(files),
            processed_documents=len(documents),
            skipped_documents=len(skipped),
            documents=documents,
            skipped=skipped,
        )

    def _iter_candidate_files(self) -> list[Path]:
        """Collect candidate files for ingestion.

        Returns:
            list[Path]: Candidate files matching supported suffixes.

        """
        input_path = self.config.input_path

        if input_path.is_file():
            return [input_path] if input_path.suffix.lower() in supported_suffixes() else []

        pattern = "**/*" if self.config.recursive else "*"
        return [
            path
            for path in sorted(input_path.glob(pattern))
            if path.is_file() and path.suffix.lower() in supported_suffixes()
        ]


def run_ingestion(config: IngestionConfig) -> IngestionRunResult:
    """Run ingestion pipeline in one function call.

    Args:
        config (IngestionConfig): Pipeline configuration.

    Returns:
        IngestionRunResult: Ingestion run result.

    """
    return IngestionPipeline(config=config).run()


def _document_id_for_path(path: Path) -> str:
    """Compute deterministic document ID from file path.

    Args:
        path (Path): Input file path.

    Returns:
        str: Hexadecimal hash identifier.

    """
    digest = hashlib.sha256()
    digest.update(str(path.resolve()).encode("utf-8", errors="ignore"))
    return digest.hexdigest()
