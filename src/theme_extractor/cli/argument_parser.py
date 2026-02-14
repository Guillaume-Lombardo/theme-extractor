"""Argument parser construction for all CLI subcommands."""

from __future__ import annotations

import argparse
import os

from theme_extractor.cli.command_handlers import (
    _DEFAULT_BASELINE_FIELDS,
    _DEFAULT_METHODS,
    handle_benchmark,
    handle_doctor,
    handle_evaluate,
    handle_extract,
    handle_ingest,
    handle_report,
)
from theme_extractor.cli.common_runtime import _DEFAULT_BACKEND_URL, _DEFAULT_INDEX, env_bool
from theme_extractor.domain import (
    BackendName,
    BertopicClustering,
    BertopicDimReduction,
    CommandName,
    ExtractMethod,
    LlmProvider,
    MsgAttachmentPolicy,
    OfflinePolicy,
    OutputFocus,
    cleaning_flag_to_string,
    default_cleaning_options,
)


def add_shared_runtime_flags(subparser: argparse.ArgumentParser) -> None:
    """Add common runtime flags shared by subcommands."""
    subparser.add_argument(
        "--offline-policy",
        default=OfflinePolicy.STRICT.value,
        choices=[policy.value for policy in OfflinePolicy],
        help=(
            "Offline strategy: 'strict' forbids runtime downloads; "
            "'preload_or_first_run' allows model preloading at install time or first run."
        ),
    )
    subparser.add_argument(
        "--proxy-url",
        default=os.getenv("THEME_EXTRACTOR_PROXY_URL"),
        help="Optional HTTP/HTTPS proxy URL for network-enabled workflows.",
    )
    subparser.add_argument(
        "--backend",
        default=BackendName.ELASTICSEARCH.value,
        choices=[backend.value for backend in BackendName],
        help="Search backend used by baseline methods.",
    )
    subparser.add_argument(
        "--backend-url",
        default=_DEFAULT_BACKEND_URL,
        help="Backend base URL (Elasticsearch or OpenSearch).",
    )
    subparser.add_argument(
        "--index",
        default=_DEFAULT_INDEX,
        help="Target index name.",
    )


def add_output_flag(subparser: argparse.ArgumentParser) -> None:
    """Add output emission flag used by all subcommands."""
    subparser.add_argument(
        "--output",
        default="-",
        help="Output destination: '-' for stdout or path to file.",
    )


def add_baseline_strategy_flags(subparser: argparse.ArgumentParser) -> None:
    """Add extraction strategy flags shared by extract/benchmark commands."""
    subparser.add_argument(
        "--query",
        default="match_all",
        help="Search query used by backend-driven baseline extraction methods.",
    )
    subparser.add_argument(
        "--fields",
        default=",".join(_DEFAULT_BASELINE_FIELDS),
        help="Comma-separated fields used for search query matching.",
    )
    subparser.add_argument(
        "--source-field",
        default="content",
        help="Field used to collect document text for extraction methods.",
    )
    subparser.add_argument(
        "--topn",
        default=25,
        type=int,
        help="Maximum number of returned topics/terms.",
    )
    subparser.add_argument(
        "--search-size",
        default=200,
        type=int,
        help="Number of documents fetched from backend for search-driven methods.",
    )
    subparser.add_argument(
        "--agg-field",
        default="tokens",
        help="Field used by terms/significant_terms/significant_text aggregations.",
    )
    subparser.add_argument(
        "--terms-min-doc-count",
        default=1,
        type=int,
        help="Minimum bucket document count for terms aggregation.",
    )
    subparser.add_argument(
        "--sigtext-filter-duplicate",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable duplicate filtering in significant_text aggregation.",
    )
    subparser.add_argument(
        "--bertopic-use-embeddings",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Enable embedding vectors for BERTopic clustering.",
    )
    subparser.add_argument(
        "--bertopic-embedding-model",
        default="bge-m3",
        help="Embedding model name (can be a local alias).",
    )
    subparser.add_argument(
        "--bertopic-local-models-dir",
        default=os.getenv("THEME_EXTRACTOR_LOCAL_MODELS_DIR", "data/models"),
        help="Local models directory used to resolve embedding aliases.",
    )
    subparser.add_argument(
        "--bertopic-dim-reduction",
        default=BertopicDimReduction.SVD.value,
        choices=[item.value for item in BertopicDimReduction],
        help="Dimensionality reduction strategy before clustering.",
    )
    subparser.add_argument(
        "--bertopic-clustering",
        default=BertopicClustering.KMEANS.value,
        choices=[item.value for item in BertopicClustering],
        help="Clustering strategy used by BERTopic.",
    )
    subparser.add_argument(
        "--bertopic-nr-topics",
        default=None,
        type=int,
        help="Optional fixed number of topics for BERTopic KMeans clustering.",
    )
    subparser.add_argument(
        "--bertopic-min-topic-size",
        default=10,
        type=int,
        help="Minimum number of documents required to keep a BERTopic cluster.",
    )
    subparser.add_argument(
        "--bertopic-seed",
        default=42,
        type=int,
        help="Random seed used by BERTopic internals.",
    )
    subparser.add_argument(
        "--bertopic-embedding-cache-enabled",
        default=env_bool("THEME_EXTRACTOR_BERTOPIC_EMBEDDING_CACHE_ENABLED", default_value=True),
        action=argparse.BooleanOptionalAction,
        help="Enable local embedding cache for BERTopic embeddings.",
    )
    subparser.add_argument(
        "--bertopic-embedding-cache-dir",
        default=os.getenv("THEME_EXTRACTOR_BERTOPIC_EMBEDDING_CACHE_DIR", "data/cache/embeddings"),
        help="Directory used to store BERTopic embedding cache entries.",
    )
    subparser.add_argument(
        "--bertopic-embedding-cache-version",
        default=os.getenv("THEME_EXTRACTOR_BERTOPIC_EMBEDDING_CACHE_VERSION", "v1"),
        help="Cache namespace version for BERTopic embedding entries.",
    )
    subparser.add_argument(
        "--keybert-use-embeddings",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Enable embeddings for KeyBERT ranking.",
    )
    subparser.add_argument(
        "--keybert-embedding-model",
        default="bge-m3",
        help="Embedding model name used when --keybert-use-embeddings is enabled.",
    )
    subparser.add_argument(
        "--keybert-local-models-dir",
        default=os.getenv("THEME_EXTRACTOR_LOCAL_MODELS_DIR", "data/models"),
        help="Local models directory used to resolve embedding aliases.",
    )
    subparser.add_argument(
        "--llm-provider",
        default=LlmProvider.OPENAI.value,
        choices=[provider.value for provider in LlmProvider],
        help="LLM provider used by the llm extraction method.",
    )
    subparser.add_argument(
        "--llm-model",
        default="gpt-4o-mini",
        help="LLM model identifier.",
    )
    subparser.add_argument(
        "--llm-api-key-env-var",
        default="OPENAI_API_KEY",
        help="Environment variable holding the provider API key.",
    )
    subparser.add_argument(
        "--llm-api-base-url",
        default=None,
        help="Optional custom API base URL for compatible providers.",
    )
    subparser.add_argument(
        "--llm-temperature",
        default=0.0,
        type=float,
        help="LLM generation temperature.",
    )
    subparser.add_argument(
        "--llm-timeout-s",
        default=30.0,
        type=float,
        help="LLM request timeout in seconds.",
    )
    subparser.add_argument(
        "--llm-max-input-chars",
        default=20_000,
        type=int,
        help="Maximum corpus characters sent to the LLM prompt.",
    )


def build_ingest_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the ingest subcommand parser."""
    ingest_parser = subparsers.add_parser(CommandName.INGEST.value, help="Plan ingestion configuration.")
    ingest_parser.add_argument("--input", required=True, help="Input folder or file path to ingest.")
    ingest_parser.add_argument(
        "--recursive",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Recursively scan subdirectories (enabled by default).",
    )
    ingest_parser.add_argument(
        "--manual-stopwords",
        default="",
        help="Comma-separated manual stopwords to remove.",
    )
    ingest_parser.add_argument(
        "--manual-stopwords-file",
        action="append",
        default=[],
        help="Path to a YAML/JSON/CSV/text file containing extra manual stopwords. Can be repeated.",
    )
    ingest_parser.add_argument(
        "--default-stopwords",
        default=env_bool("THEME_EXTRACTOR_DEFAULT_STOPWORDS_ENABLED", default_value=True),
        action=argparse.BooleanOptionalAction,
        help="Enable default FR/EN stopwords loaded from nltk (or fallback lists).",
    )
    ingest_parser.add_argument(
        "--auto-stopwords",
        default=env_bool("THEME_EXTRACTOR_AUTO_STOPWORDS_ENABLED", default_value=False),
        action=argparse.BooleanOptionalAction,
        help="Enable automatic stopwords generation from corpus statistics.",
    )
    ingest_parser.add_argument(
        "--auto-stopwords-min-doc-ratio",
        default=float(os.getenv("THEME_EXTRACTOR_AUTO_STOPWORDS_MIN_DOC_RATIO", "0.7")),
        type=float,
        help="Minimum document ratio for auto stopwords generation.",
    )
    ingest_parser.add_argument(
        "--auto-stopwords-max-terms",
        default=int(os.getenv("THEME_EXTRACTOR_AUTO_STOPWORDS_MAX_TERMS", "200")),
        type=int,
        help="Maximum number of automatically generated stopwords.",
    )
    ingest_parser.add_argument(
        "--auto-stopwords-min-corpus-ratio",
        default=float(os.getenv("THEME_EXTRACTOR_AUTO_STOPWORDS_MIN_CORPUS_RATIO", "0.01")),
        type=float,
        help="Minimum corpus frequency ratio for auto stopwords generation.",
    )
    ingest_parser.add_argument(
        "--cleaning-options",
        default=cleaning_flag_to_string(default_cleaning_options()),
        help=(
            "Comma-separated cleaning options. "
            "Available: none, all, whitespace, accent_normalization, header_footer, "
            "boilerplate, token_cleanup, html_strip."
        ),
    )
    ingest_parser.add_argument(
        "--streaming-mode",
        default=True,
        action=argparse.BooleanOptionalAction,
        help=(
            "Enable compact ingestion mode that keeps token-frequency summaries instead of "
            "full token lists to reduce memory usage on large corpora."
        ),
    )
    ingest_parser.add_argument(
        "--pdf-ocr-fallback",
        default=env_bool("THEME_EXTRACTOR_PDF_OCR_FALLBACK_ENABLED", default_value=False),
        action=argparse.BooleanOptionalAction,
        help=("Enable OCR fallback for PDF pages with little or no embedded text (useful for scanned PDFs)."),
    )
    ingest_parser.add_argument(
        "--pdf-ocr-languages",
        default=os.getenv("THEME_EXTRACTOR_PDF_OCR_LANGUAGES", "fra+eng"),
        help="OCR language codes used when PDF OCR fallback is enabled.",
    )
    ingest_parser.add_argument(
        "--pdf-ocr-dpi",
        default=int(os.getenv("THEME_EXTRACTOR_PDF_OCR_DPI", "200")),
        type=int,
        help="OCR rendering DPI for PDF OCR fallback.",
    )
    ingest_parser.add_argument(
        "--pdf-ocr-min-chars",
        default=int(os.getenv("THEME_EXTRACTOR_PDF_OCR_MIN_CHARS", "32")),
        type=int,
        help=(
            "Minimum alphanumeric characters in embedded text to skip OCR fallback "
            "(pages with fewer trigger OCR)."
        ),
    )
    ingest_parser.add_argument(
        "--pdf-ocr-tessdata",
        default=os.getenv("THEME_EXTRACTOR_PDF_OCR_TESSDATA", None),
        help="Optional tessdata directory path for OCR runtime.",
    )
    ingest_parser.add_argument(
        "--msg-include-metadata",
        default=env_bool("THEME_EXTRACTOR_MSG_INCLUDE_METADATA", default_value=True),
        action=argparse.BooleanOptionalAction,
        help="Include `.msg` metadata fields (subject/from/to/cc/date) in extracted text.",
    )
    ingest_parser.add_argument(
        "--msg-attachments-policy",
        default=os.getenv("THEME_EXTRACTOR_MSG_ATTACHMENTS_POLICY", MsgAttachmentPolicy.NAMES.value),
        choices=[policy.value for policy in MsgAttachmentPolicy],
        help="Attachment extraction policy for `.msg` files.",
    )
    ingest_parser.add_argument(
        "--reset-index",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Reset backend index (delete/recreate) before ingestion.",
    )
    ingest_parser.add_argument(
        "--index-backend",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Index processed documents into backend using stopword-filtered extraction fields.",
    )
    add_shared_runtime_flags(ingest_parser)
    add_output_flag(ingest_parser)
    ingest_parser.set_defaults(handler=handle_ingest)


def build_extract_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the extract subcommand parser."""
    extract_parser = subparsers.add_parser(CommandName.EXTRACT.value, help="Run one extraction strategy.")
    extract_parser.add_argument(
        "--method",
        required=True,
        choices=[method.value for method in ExtractMethod],
        help="Extraction strategy to run.",
    )
    extract_parser.add_argument(
        "--focus",
        default=OutputFocus.TOPICS.value,
        choices=[focus.value for focus in OutputFocus],
        help="Unified output focus.",
    )
    add_baseline_strategy_flags(extract_parser)
    add_shared_runtime_flags(extract_parser)
    add_output_flag(extract_parser)
    extract_parser.set_defaults(handler=handle_extract)


def build_benchmark_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the benchmark subcommand parser."""
    benchmark_parser = subparsers.add_parser(
        CommandName.BENCHMARK.value,
        help="Run multiple extraction strategies under one benchmark run.",
    )
    benchmark_parser.add_argument(
        "--methods",
        default=_DEFAULT_METHODS,
        help="Comma-separated extraction methods.",
    )
    benchmark_parser.add_argument(
        "--focus",
        default=OutputFocus.TOPICS.value,
        choices=[focus.value for focus in OutputFocus],
        help="Unified output focus shared by all benchmarked methods.",
    )
    add_baseline_strategy_flags(benchmark_parser)
    add_shared_runtime_flags(benchmark_parser)
    add_output_flag(benchmark_parser)
    benchmark_parser.set_defaults(handler=handle_benchmark)


def build_doctor_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register doctor subcommand parser."""
    doctor_parser = subparsers.add_parser(
        CommandName.DOCTOR.value,
        help="Inspect local runtime/dependency/backend readiness.",
    )
    doctor_parser.add_argument(
        "--expected-local-models",
        default="bge-m3",
        help="Comma-separated expected model aliases under local models directory.",
    )
    doctor_parser.add_argument(
        "--local-models-dir",
        default="data/models",
        help="Local models directory checked by doctor command.",
    )
    doctor_parser.add_argument(
        "--check-backend",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Try one backend query to validate connectivity and credentials.",
    )
    add_shared_runtime_flags(doctor_parser)
    add_output_flag(doctor_parser)
    doctor_parser.set_defaults(handler=handle_doctor)


def build_report_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register report subcommand parser."""
    report_parser = subparsers.add_parser(
        CommandName.REPORT.value,
        help="Render markdown report from extraction/benchmark output.",
    )
    report_parser.add_argument(
        "--input",
        required=True,
        help="Input extraction/benchmark JSON file path.",
    )
    report_parser.add_argument(
        "--title",
        default=None,
        help="Optional markdown report title.",
    )
    add_output_flag(report_parser)
    report_parser.set_defaults(handler=handle_report)


def build_evaluate_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register evaluate subcommand parser."""
    evaluate_parser = subparsers.add_parser(
        CommandName.EVALUATE.value,
        help="Compute evaluation metrics from extraction/benchmark output files.",
    )
    evaluate_parser.add_argument(
        "--input",
        required=True,
        nargs="+",
        help="One or many extraction/benchmark JSON paths.",
    )
    add_output_flag(evaluate_parser)
    evaluate_parser.set_defaults(handler=handle_evaluate)


def build_parser() -> argparse.ArgumentParser:
    """Build top-level CLI parser with all subcommands.

    Returns:
        argparse.ArgumentParser: Configured top-level parser.

    """
    parser = argparse.ArgumentParser(
        prog="theme-extractor",
        description="Unified CLI for ingestion, extraction, benchmarking, evaluation, and reporting.",
    )
    subparsers = parser.add_subparsers(dest="command")

    build_ingest_parser(subparsers)
    build_extract_parser(subparsers)
    build_benchmark_parser(subparsers)
    build_doctor_parser(subparsers)
    build_report_parser(subparsers)
    build_evaluate_parser(subparsers)

    return parser
