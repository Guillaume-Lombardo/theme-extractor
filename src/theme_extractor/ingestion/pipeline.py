"""Ingestion pipeline orchestration for corpus files."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path  # noqa: TC003

from pydantic import BaseModel, ConfigDict, Field

from theme_extractor.domain import CleaningOptionFlag, cleaning_flag_to_string, default_cleaning_options
from theme_extractor.ingestion.cleaning import (
    apply_cleaning_options,
    discover_auto_stopwords,
    discover_auto_stopwords_from_frequencies,
    get_default_stopwords,
    load_stopwords_from_files,
    normalize_french_accents,
    tokenize_for_ingestion,
)
from theme_extractor.ingestion.extractors import extract_text, supported_suffixes


class IngestionConfig(BaseModel):
    """Represent ingestion configuration options.

    Attributes:
        input_path (Path): Input file or directory.
        recursive (bool): Whether to recurse in input directories.
        cleaning_options (CleaningOptionFlag): Cleaning options bit flag.
        manual_stopwords (set[str]): Inline stopwords provided from CLI.
        manual_stopwords_files (list[Path]): Stopwords files (`yaml/csv/txt`) to load.
        default_stopwords_enabled (bool): Enable default FR/EN stopwords.
        auto_stopwords_enabled (bool): Enable corpus-derived stopwords.
        auto_stopwords_min_doc_ratio (float): Minimum document coverage ratio.
        auto_stopwords_min_corpus_ratio (float): Minimum corpus frequency ratio.
        auto_stopwords_max_terms (int): Maximum count of auto stopwords.
        streaming_mode (bool): Enable compact token-frequency aggregation mode.

    """

    input_path: Path
    recursive: bool = True
    cleaning_options: CleaningOptionFlag = Field(default_factory=default_cleaning_options)
    manual_stopwords: set[str] = Field(default_factory=set)
    manual_stopwords_files: list[Path] = Field(default_factory=list)
    default_stopwords_enabled: bool = True
    auto_stopwords_enabled: bool = False
    auto_stopwords_min_doc_ratio: float = 0.7
    auto_stopwords_min_corpus_ratio: float = 0.01
    auto_stopwords_max_terms: int = 200
    streaming_mode: bool = True


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
    """Represent full ingestion pipeline output.

    Attributes:
        command (str): Command name (`ingest`).
        input_path (str): Normalized input path.
        recursive (bool): Recursive mode used.
        supported_suffixes (list[str]): Supported file extensions.
        cleaning_options (str): Serialized cleaning options.
        default_stopwords_enabled (bool): Whether default stopwords were enabled.
        default_stopwords_count (int): Number of default stopwords loaded.
        auto_stopwords_enabled (bool): Whether auto stopwords were enabled.
        streaming_mode (bool): Whether compact streaming mode was enabled.
        manual_stopwords (list[str]): Effective manual stopwords (CLI + files).
        manual_stopwords_files (list[str]): Manual stopwords file paths.
        auto_stopwords (list[str]): Auto-generated stopwords.
        total_candidate_files (int): Number of candidate files found.
        processed_documents (int): Number of successfully processed documents.
        skipped_documents (int): Number of skipped documents.
        documents (list[IngestedDocument]): Processed document metadata.
        skipped (list[SkippedDocument]): Skipped entries and reasons.

    """

    command: str = "ingest"
    input_path: str
    recursive: bool
    supported_suffixes: list[str]
    cleaning_options: str
    default_stopwords_enabled: bool
    default_stopwords_count: int
    auto_stopwords_enabled: bool
    streaming_mode: bool
    manual_stopwords: list[str]
    manual_stopwords_files: list[str]
    auto_stopwords: list[str]
    total_candidate_files: int
    processed_documents: int
    skipped_documents: int
    documents: list[IngestedDocument]
    skipped: list[SkippedDocument]


class _ProcessedDocumentMetadata(BaseModel):
    """Represent lightweight processed document metadata."""

    model_config = ConfigDict(frozen=True)

    path: Path
    raw_length: int
    clean_length: int
    clean_text_preview: str
    tokens: list[str]


class _ProcessedDocumentStats(BaseModel):
    """Represent compact processed document statistics for streaming mode."""

    model_config = ConfigDict(frozen=True)

    path: Path
    raw_length: int
    clean_length: int
    clean_text_preview: str
    token_count: int
    token_frequencies: dict[str, int]


class _CorpusTokenStats(BaseModel):
    """Represent aggregated corpus token statistics."""

    model_config = ConfigDict(frozen=True)

    document_frequency: dict[str, int]
    collection_frequency: dict[str, int]
    total_docs: int
    total_tokens: int


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
        skipped: list[SkippedDocument] = []
        documents: list[IngestedDocument] = []

        if self.config.streaming_mode:
            processed_stats, skipped, corpus_stats = self._collect_processed_stats(files)
            (
                auto_stopwords,
                manual_cli_stopwords,
                manual_file_stopwords,
                default_stopwords,
                stopwords,
            ) = self._build_stopwords_from_stats(corpus_stats)
            documents = self._build_documents_from_stats(processed_stats, stopwords)
        else:
            tokenized_documents, processed_items, skipped = self._collect_processed_items(files)
            (
                auto_stopwords,
                manual_cli_stopwords,
                manual_file_stopwords,
                default_stopwords,
                stopwords,
            ) = self._build_stopwords(tokenized_documents)
            documents = self._build_documents_from_metadata(processed_items, stopwords)

        return IngestionRunResult(
            input_path=str(self.config.input_path),
            recursive=self.config.recursive,
            supported_suffixes=sorted(supported_suffixes()),
            cleaning_options=cleaning_flag_to_string(self.config.cleaning_options),
            default_stopwords_enabled=self.config.default_stopwords_enabled,
            default_stopwords_count=len(default_stopwords),
            auto_stopwords_enabled=self.config.auto_stopwords_enabled,
            streaming_mode=self.config.streaming_mode,
            manual_stopwords=sorted(manual_cli_stopwords | manual_file_stopwords),
            manual_stopwords_files=sorted(str(path) for path in self.config.manual_stopwords_files),
            auto_stopwords=sorted(auto_stopwords),
            total_candidate_files=len(files),
            processed_documents=len(documents),
            skipped_documents=len(skipped),
            documents=documents,
            skipped=skipped,
        )

    @staticmethod
    def _build_documents_from_metadata(
        processed_items: list[_ProcessedDocumentMetadata],
        stopwords: set[str],
    ) -> list[IngestedDocument]:
        """Build document output payload from token lists.

        Args:
            processed_items (list[_ProcessedDocumentMetadata]): Processed metadata with raw tokens.
            stopwords (set[str]): Effective stopwords set.

        Returns:
            list[IngestedDocument]: Final document payload items.

        """
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
        return documents

    @staticmethod
    def _build_documents_from_stats(
        processed_items: list[_ProcessedDocumentStats],
        stopwords: set[str],
    ) -> list[IngestedDocument]:
        """Build document output payload from compact token-frequency maps.

        Args:
            processed_items (list[_ProcessedDocumentStats]): Processed metadata with token frequencies.
            stopwords (set[str]): Effective stopwords set.

        Returns:
            list[IngestedDocument]: Final document payload items.

        """
        documents: list[IngestedDocument] = []
        for item in processed_items:
            removed_count = sum(
                count for token, count in item.token_frequencies.items() if token.lower() in stopwords
            )
            doc_id = _document_id_for_path(item.path)

            documents.append(
                IngestedDocument(
                    document_id=doc_id,
                    path=str(item.path),
                    extension=item.path.suffix.lower(),
                    raw_length=item.raw_length,
                    clean_length=item.clean_length,
                    token_count=item.token_count - removed_count,
                    removed_stopword_count=removed_count,
                    clean_text_preview=item.clean_text_preview,
                ),
            )
        return documents

    def _collect_processed_items(
        self,
        files: list[Path],
    ) -> tuple[list[list[str]], list[_ProcessedDocumentMetadata], list[SkippedDocument]]:
        """Extract, clean, and tokenize candidate files.

        Args:
            files (list[Path]): Candidate files.

        Returns:
            tuple[list[list[str]], list[_ProcessedDocumentMetadata], list[SkippedDocument]]:
                Tokenized docs, processed metadata, and skipped items.

        """
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

        return tokenized_documents, processed_items, skipped

    def _collect_processed_stats(
        self,
        files: list[Path],
    ) -> tuple[list[_ProcessedDocumentStats], list[SkippedDocument], _CorpusTokenStats]:
        """Extract and tokenize candidate files with compact corpus statistics.

        Args:
            files (list[Path]): Candidate files.

        Returns:
            tuple[list[_ProcessedDocumentStats], list[SkippedDocument], _CorpusTokenStats]:
                Processed compact document stats, skipped items, and corpus aggregates.

        """
        processed_items: list[_ProcessedDocumentStats] = []
        skipped: list[SkippedDocument] = []
        document_frequency: Counter[str] = Counter()
        collection_frequency: Counter[str] = Counter()
        total_tokens = 0

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
            token_counter = Counter(tokens)
            document_frequency.update(set(tokens))
            collection_frequency.update(tokens)
            total_tokens += len(tokens)

            processed_items.append(
                _ProcessedDocumentStats(
                    path=file_path,
                    raw_length=len(raw_text),
                    clean_length=len(cleaned_text),
                    clean_text_preview=cleaned_text[:200],
                    token_count=len(tokens),
                    token_frequencies=dict(token_counter),
                ),
            )

        corpus_stats = _CorpusTokenStats(
            document_frequency=dict(document_frequency),
            collection_frequency=dict(collection_frequency),
            total_docs=len(processed_items),
            total_tokens=total_tokens,
        )
        return processed_items, skipped, corpus_stats

    def _build_stopwords(
        self,
        tokenized_documents: list[list[str]],
    ) -> tuple[set[str], set[str], set[str], set[str], set[str]]:
        """Build all stopword buckets and merged stopword set.

        Args:
            tokenized_documents (list[list[str]]): Tokenized corpus documents.

        Returns:
            tuple[set[str], set[str], set[str], set[str], set[str]]:
                `(auto_stopwords, manual_cli_stopwords, manual_file_stopwords, default_stopwords, merged_stopwords)`.

        """
        auto_stopwords: set[str] = set()
        if self.config.auto_stopwords_enabled:
            auto_stopwords = discover_auto_stopwords(
                tokenized_documents,
                min_doc_ratio=self.config.auto_stopwords_min_doc_ratio,
                min_corpus_ratio=self.config.auto_stopwords_min_corpus_ratio,
                max_terms=self.config.auto_stopwords_max_terms,
            )

        manual_file_stopwords = load_stopwords_from_files(self.config.manual_stopwords_files)
        manual_cli_stopwords = {
            normalize_french_accents(word.strip().lower())
            for word in self.config.manual_stopwords
            if word.strip()
        }
        default_stopwords = get_default_stopwords() if self.config.default_stopwords_enabled else set()
        merged_stopwords = default_stopwords | manual_cli_stopwords | manual_file_stopwords | auto_stopwords
        return (
            auto_stopwords,
            manual_cli_stopwords,
            manual_file_stopwords,
            default_stopwords,
            merged_stopwords,
        )

    def _build_stopwords_from_stats(
        self,
        corpus_stats: _CorpusTokenStats,
    ) -> tuple[set[str], set[str], set[str], set[str], set[str]]:
        """Build stopwords using precomputed corpus frequencies.

        Args:
            corpus_stats (_CorpusTokenStats): Aggregated corpus token stats.

        Returns:
            tuple[set[str], set[str], set[str], set[str], set[str]]:
                `(auto_stopwords, manual_cli_stopwords, manual_file_stopwords, default_stopwords, merged_stopwords)`.

        """
        auto_stopwords: set[str] = set()
        if self.config.auto_stopwords_enabled:
            auto_stopwords = discover_auto_stopwords_from_frequencies(
                document_frequency=corpus_stats.document_frequency,
                collection_frequency=corpus_stats.collection_frequency,
                total_docs=corpus_stats.total_docs,
                total_tokens=corpus_stats.total_tokens,
                min_doc_ratio=self.config.auto_stopwords_min_doc_ratio,
                min_corpus_ratio=self.config.auto_stopwords_min_corpus_ratio,
                max_terms=self.config.auto_stopwords_max_terms,
            )

        manual_file_stopwords = load_stopwords_from_files(self.config.manual_stopwords_files)
        manual_cli_stopwords = {
            normalize_french_accents(word.strip().lower())
            for word in self.config.manual_stopwords
            if word.strip()
        }
        default_stopwords = get_default_stopwords() if self.config.default_stopwords_enabled else set()
        merged_stopwords = default_stopwords | manual_cli_stopwords | manual_file_stopwords | auto_stopwords
        return (
            auto_stopwords,
            manual_cli_stopwords,
            manual_file_stopwords,
            default_stopwords,
            merged_stopwords,
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
