"""Unified CLI for ingestion, extraction, and benchmarking workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from theme_extractor.domain import (
    BackendName,
    BenchmarkOutput,
    CommandName,
    DocumentTopicLink,
    ExtractionRunMetadata,
    ExtractMethod,
    OfflinePolicy,
    OutputFocus,
    UnifiedExtractionOutput,
    cleaning_flag_from_string,
    cleaning_flag_to_string,
    default_cleaning_options,
    method_flag_from_string,
    method_flag_to_methods,
    parse_extract_method,
)
from theme_extractor.ingestion import IngestionConfig, run_ingestion

_DEFAULT_BACKEND_URL = "http://localhost:9200"
_DEFAULT_INDEX = "theme_extractor"
_DEFAULT_METHODS = "baseline_tfidf,keybert,bertopic,llm"


def _emit_payload(payload: dict[str, Any] | BaseModel, output: str) -> None:
    """Emit payload to stdout or to a JSON file.

    Args:
        payload (dict[str, Any] | BaseModel): Data payload to serialize.
        output (str): Output path or "-" for stdout.

    """
    payload_dict = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
    serialized = json.dumps(payload_dict, ensure_ascii=False, indent=2)

    if output == "-":
        print(serialized)
        return

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(serialized + "\n", encoding="utf-8")


def _add_shared_runtime_flags(subparser: argparse.ArgumentParser) -> None:
    """Add common runtime flags shared by subcommands.

    Args:
        subparser (argparse.ArgumentParser): Subcommand parser to enrich.

    """
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
        default=None,
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


def _add_output_flag(subparser: argparse.ArgumentParser) -> None:
    """Add output target flag for JSON payloads.

    Args:
        subparser (argparse.ArgumentParser): Subcommand parser to enrich.

    """
    subparser.add_argument(
        "--output",
        default="-",
        help="Output file path for JSON result. Use '-' to print to stdout.",
    )


def _build_ingest_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the ingest subcommand parser.

    Args:
        subparsers (argparse._SubParsersAction[argparse.ArgumentParser]): Subparser registry.

    """
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
        "--auto-stopwords",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Enable automatic stopwords generation from corpus statistics.",
    )
    ingest_parser.add_argument(
        "--auto-stopwords-min-doc-ratio",
        default=0.7,
        type=float,
        help="Minimum document ratio for auto stopwords generation.",
    )
    ingest_parser.add_argument(
        "--auto-stopwords-max-terms",
        default=200,
        type=int,
        help="Maximum number of automatically generated stopwords.",
    )
    ingest_parser.add_argument(
        "--cleaning-options",
        default=cleaning_flag_to_string(default_cleaning_options()),
        help=(
            "Comma-separated cleaning options. "
            "Available: all, whitespace, accent_normalization, header_footer, "
            "boilerplate, token_cleanup, html_strip."
        ),
    )
    _add_shared_runtime_flags(ingest_parser)
    _add_output_flag(ingest_parser)
    ingest_parser.set_defaults(handler=_handle_ingest)


def _build_extract_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the extract subcommand parser.

    Args:
        subparsers (argparse._SubParsersAction[argparse.ArgumentParser]): Subparser registry.

    """
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
    _add_shared_runtime_flags(extract_parser)
    _add_output_flag(extract_parser)
    extract_parser.set_defaults(handler=_handle_extract)


def _build_benchmark_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the benchmark subcommand parser.

    Args:
        subparsers (argparse._SubParsersAction[argparse.ArgumentParser]): Subparser registry.

    """
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
        help="Unified output focus.",
    )
    _add_shared_runtime_flags(benchmark_parser)
    _add_output_flag(benchmark_parser)
    benchmark_parser.set_defaults(handler=_handle_benchmark)


def build_parser() -> argparse.ArgumentParser:
    """Build the root parser and all subcommands.

    Returns:
        argparse.ArgumentParser: Configured root parser.

    """
    parser = argparse.ArgumentParser(
        prog="theme-extractor",
        description="CLI to ingest corpora and benchmark topic extraction strategies.",
    )
    subparsers = parser.add_subparsers(dest="command")

    _build_ingest_parser(subparsers)
    _build_extract_parser(subparsers)
    _build_benchmark_parser(subparsers)

    return parser


def _handle_ingest(args: argparse.Namespace) -> dict[str, Any]:
    """Build a normalized JSON payload for ingestion runs.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        dict[str, Any]: Normalized ingestion payload.

    """
    manual_stopwords = {
        word.strip().lower() for word in str(args.manual_stopwords).split(",") if word.strip()
    }
    config = IngestionConfig(
        input_path=Path(args.input).expanduser().resolve(),
        recursive=bool(args.recursive),
        cleaning_options=cleaning_flag_from_string(str(args.cleaning_options)),
        manual_stopwords=manual_stopwords,
        auto_stopwords_enabled=bool(args.auto_stopwords),
        auto_stopwords_min_doc_ratio=float(args.auto_stopwords_min_doc_ratio),
        auto_stopwords_max_terms=int(args.auto_stopwords_max_terms),
    )
    result = run_ingestion(config)
    payload = result.model_dump(mode="json")
    payload["schema_version"] = "1.0"
    payload["runtime"] = {
        "offline_policy": OfflinePolicy(args.offline_policy).value,
        "proxy_url": args.proxy_url,
        "backend": BackendName(args.backend).value,
        "backend_url": args.backend_url,
        "index": args.index,
    }
    return payload


def _handle_extract(args: argparse.Namespace) -> UnifiedExtractionOutput:
    """Build a normalized JSON payload for a single extraction strategy.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        UnifiedExtractionOutput: Unified extraction output payload.

    """
    method = parse_extract_method(args.method)
    focus = OutputFocus(args.focus)
    offline_policy = OfflinePolicy(args.offline_policy)
    backend = BackendName(args.backend)

    document_topics: list[DocumentTopicLink] | None = None
    if focus in {OutputFocus.DOCUMENTS, OutputFocus.BOTH}:
        document_topics = []

    metadata = ExtractionRunMetadata(
        command=CommandName.EXTRACT,
        method=method,
        offline_policy=offline_policy,
        backend=backend,
        index=args.index,
    )
    return UnifiedExtractionOutput(
        focus=focus,
        topics=[],
        document_topics=document_topics,
        notes=[
            "Topic-first unified schema is active.",
            "Document-topic links are optional and currently empty in this phase.",
        ],
        metadata=metadata,
    )


def _handle_benchmark(args: argparse.Namespace) -> BenchmarkOutput:
    """Build a normalized JSON payload for multi-method benchmarking.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        BenchmarkOutput: Benchmark payload with one normalized output per method.

    """
    method_flag = method_flag_from_string(args.methods)
    methods = method_flag_to_methods(method_flag)
    focus = OutputFocus(args.focus)
    offline_policy = OfflinePolicy(args.offline_policy)
    backend = BackendName(args.backend)

    outputs: dict[str, UnifiedExtractionOutput] = {}

    for method in methods:
        document_topics: list[DocumentTopicLink] | None = None
        if focus in {OutputFocus.DOCUMENTS, OutputFocus.BOTH}:
            document_topics = []

        metadata = ExtractionRunMetadata(
            command=CommandName.BENCHMARK,
            method=method,
            offline_policy=offline_policy,
            backend=backend,
            index=args.index,
        )
        outputs[method.value] = UnifiedExtractionOutput(
            focus=focus,
            topics=[],
            document_topics=document_topics,
            notes=[
                "Topic-first unified schema is active.",
                "Benchmark execution engine is not implemented yet in this phase.",
            ],
            metadata=metadata,
        )

    return BenchmarkOutput(methods=methods, outputs=outputs)


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process exit code.

    Args:
        argv (list[str] | None): Optional command-line arguments.

    Returns:
        int: Process exit code.

    """
    parser = build_parser()

    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1

    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 1

    payload = handler(args)
    _emit_payload(payload=payload, output=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
