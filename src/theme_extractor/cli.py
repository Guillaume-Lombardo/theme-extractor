"""Unified CLI for ingestion, extraction, and benchmarking workflows."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from theme_extractor.domain import (
    BackendName,
    BenchmarkOutput,
    BertopicClustering,
    BertopicDimReduction,
    CommandName,
    DocumentTopicLink,
    ExtractionRunMetadata,
    ExtractMethod,
    LlmProvider,
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
from theme_extractor.evaluation import evaluate_payload_files
from theme_extractor.extraction import (
    BaselineExtractionConfig,
    BaselineRunRequest,
    BertopicExtractionConfig,
    BertopicRunRequest,
    KeyBertExtractionConfig,
    KeyBertRunRequest,
    LlmExtractionConfig,
    LlmRunRequest,
    build_benchmark_comparison,
    characterize_output,
    run_baseline_method,
    run_bertopic_method,
    run_keybert_method,
    run_llm_method,
)
from theme_extractor.ingestion import IngestionConfig, run_ingestion
from theme_extractor.reporting import render_report_markdown
from theme_extractor.search.factory import build_search_backend

if TYPE_CHECKING:
    from theme_extractor.search.protocols import SearchBackend

_DEFAULT_BACKEND_URL = "http://localhost:9200"
_DEFAULT_INDEX = "theme_extractor"
_DEFAULT_METHODS = "baseline_tfidf,keybert,bertopic,llm"
_DEFAULT_BASELINE_FIELDS = ("content", "filename", "path")
_PROXY_ENV_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")
_DOCTOR_OPTIONAL_MODULES: dict[str, tuple[str, ...]] = {
    "elasticsearch": ("elasticsearch",),
    "opensearch": ("opensearchpy",),
    "bert": ("keybert", "bertopic", "sentence_transformers"),
    "llm": ("openai",),
}
_BASELINE_METHODS = {
    ExtractMethod.BASELINE_TFIDF,
    ExtractMethod.TERMS,
    ExtractMethod.SIGNIFICANT_TERMS,
    ExtractMethod.SIGNIFICANT_TEXT,
}
_SEARCH_DRIVEN_METHODS = _BASELINE_METHODS | {ExtractMethod.KEYBERT, ExtractMethod.LLM}
_UNSUPPORTED_EXTRACT_METHOD_ERROR = "Unsupported extract method: {method!r}."


def _env_bool(name: str, *, default_value: bool) -> bool:
    """Read a boolean value from environment variables.

    Args:
        name (str): Environment variable name.
        default_value (bool): Fallback value when missing or invalid.

    Returns:
        bool: Parsed boolean value.

    """
    raw = os.getenv(name)
    if raw is None:
        return default_value
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default_value


def _emit_payload(payload: dict[str, Any] | BaseModel | str, output: str) -> None:
    """Emit payload to stdout or to a file.

    Args:
        payload (dict[str, Any] | BaseModel | str): Data payload to serialize.
        output (str): Output path or "-" for stdout.

    """
    if isinstance(payload, str):
        serialized = payload
    else:
        payload_dict = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
        serialized = json.dumps(payload_dict, ensure_ascii=False, indent=2)

    if output == "-":
        print(serialized)
        return

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if serialized.endswith("\n"):
        output_path.write_text(serialized, encoding="utf-8")
    else:
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


def _parse_baseline_fields(value: str) -> tuple[str, ...]:
    """Parse baseline fields from CLI and validate non-empty output.

    Args:
        value (str): Raw comma-separated fields value.

    Returns:
        tuple[str, ...]: Parsed non-empty field names, or default fields if empty.

    """
    fields = tuple(part.strip() for part in value.split(",") if part.strip())
    return fields or _DEFAULT_BASELINE_FIELDS


def _build_baseline_config(args: argparse.Namespace) -> BaselineExtractionConfig:
    """Build baseline extraction config from CLI args.

    Args:
        args (argparse.Namespace): Parsed CLI args.

    Returns:
        BaselineExtractionConfig: Baseline runtime config.

    """
    return BaselineExtractionConfig(
        query=str(args.query),
        fields=_parse_baseline_fields(str(args.fields)),
        source_field=str(args.source_field),
        top_n=max(1, int(args.topn)),
        search_size=max(1, int(args.search_size)),
        aggregation_field=str(args.agg_field),
        terms_min_doc_count=max(1, int(args.terms_min_doc_count)),
        sigtext_filter_duplicate=bool(args.sigtext_filter_duplicate),
    )


def _build_bertopic_config(args: argparse.Namespace) -> BertopicExtractionConfig:
    """Build BERTopic extraction config from CLI args.

    Args:
        args (argparse.Namespace): Parsed CLI args.

    Returns:
        BertopicExtractionConfig: BERTopic runtime config.

    """
    nr_topics_raw = int(args.bertopic_nr_topics)
    nr_topics = nr_topics_raw if nr_topics_raw > 0 else None
    local_models_dir_raw = str(args.bertopic_local_models_dir).strip()
    local_models_dir = None
    if local_models_dir_raw and local_models_dir_raw.lower() not in {"none", "null"}:
        local_models_dir = Path(local_models_dir_raw).expanduser()
    return BertopicExtractionConfig(
        use_embeddings=bool(args.bertopic_use_embeddings),
        embedding_model=str(args.bertopic_embedding_model),
        reduce_dim=BertopicDimReduction(str(args.bertopic_dim_reduction)),
        clustering=BertopicClustering(str(args.bertopic_clustering)),
        nr_topics=nr_topics,
        min_topic_size=max(1, int(args.bertopic_min_topic_size)),
        seed=int(args.bertopic_seed),
        local_models_dir=local_models_dir,
    )


def _build_keybert_config(args: argparse.Namespace) -> KeyBertExtractionConfig:
    """Build KeyBERT extraction config from CLI args.

    Args:
        args (argparse.Namespace): Parsed CLI args.

    Returns:
        KeyBertExtractionConfig: KeyBERT runtime config.

    """
    local_models_dir_raw = str(args.keybert_local_models_dir).strip()
    local_models_dir = None
    if local_models_dir_raw and local_models_dir_raw.lower() not in {"none", "null"}:
        local_models_dir = Path(local_models_dir_raw).expanduser()
    return KeyBertExtractionConfig(
        use_embeddings=bool(args.keybert_use_embeddings),
        embedding_model=str(args.keybert_embedding_model),
        local_models_dir=local_models_dir,
    )


def _build_llm_config(args: argparse.Namespace) -> LlmExtractionConfig:
    """Build LLM extraction config from CLI args.

    Args:
        args (argparse.Namespace): Parsed CLI args.

    Returns:
        LlmExtractionConfig: LLM runtime config.

    """
    return LlmExtractionConfig(
        provider=LlmProvider(str(args.llm_provider)),
        model=str(args.llm_model),
        api_key_env_var=str(args.llm_api_key_env_var),
        api_base_url=str(args.llm_api_base_url) if args.llm_api_base_url else None,
        temperature=float(args.llm_temperature),
        timeout_s=float(args.llm_timeout_s),
        max_input_chars=max(1, int(args.llm_max_input_chars)),
    )


def _build_baseline_backend(
    *,
    args: argparse.Namespace,
    backend: BackendName,
) -> SearchBackend:
    """Build backend adapter for baseline methods.

    Args:
        args (argparse.Namespace): Parsed CLI args.
        backend (BackendName): Backend enum value.

    Returns:
        Any: Search backend adapter.

    """
    return build_search_backend(
        backend=backend,
        url=str(args.backend_url),
        timeout_s=30.0,
        verify_certs=True,
    )


def _apply_proxy_environment(proxy_url: str | None) -> None:
    """Apply runtime proxy URL to common HTTP proxy environment variables.

    Args:
        proxy_url (str | None): Proxy URL provided by CLI/user.

    """
    if not proxy_url:
        return

    for env_key in _PROXY_ENV_KEYS:
        os.environ[env_key] = proxy_url


def _is_baseline_method(method: ExtractMethod) -> bool:
    """Check whether method is a baseline method.

    Args:
        method (ExtractMethod): Extraction method.

    Returns:
        bool: True if baseline method.

    """
    return method in _BASELINE_METHODS


def _is_search_driven_method(method: ExtractMethod) -> bool:
    """Check whether method requires a search backend corpus.

    Args:
        method (ExtractMethod): Extraction method.

    Returns:
        bool: True if method requires search backend access.

    """
    return method in (_SEARCH_DRIVEN_METHODS | {ExtractMethod.BERTOPIC})


def _add_baseline_strategy_flags(subparser: argparse.ArgumentParser) -> None:
    """Add extraction baseline tuning flags.

    Args:
        subparser (argparse.ArgumentParser): Subcommand parser to enrich.

    """
    subparser.add_argument(
        "--query",
        default="match_all",
        help="Search query used by backend-driven baseline extraction methods.",
    )
    subparser.add_argument(
        "--fields",
        default="content,filename,path",
        help="Comma-separated fields used for search query matching.",
    )
    subparser.add_argument(
        "--source-field",
        default="content",
        help="Source field used to build TF-IDF corpus from backend hits.",
    )
    subparser.add_argument(
        "--topn",
        default=25,
        type=int,
        help="Maximum number of terms/topics to return.",
    )
    subparser.add_argument(
        "--search-size",
        default=200,
        type=int,
        help="Number of backend documents fetched for TF-IDF baseline.",
    )
    subparser.add_argument(
        "--agg-field",
        default="tokens",
        help="Aggregation field used by terms/significant baselines.",
    )
    subparser.add_argument(
        "--terms-min-doc-count",
        default=1,
        type=int,
        help="Minimum document count for terms aggregation buckets.",
    )
    subparser.add_argument(
        "--sigtext-filter-duplicate",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable duplicate-text filtering for significant_text aggregation.",
    )
    subparser.add_argument(
        "--bertopic-use-embeddings",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Enable embeddings for BERTopic strategy.",
    )
    subparser.add_argument(
        "--bertopic-embedding-model",
        default="bge-m3",
        help="Embedding model name used when --bertopic-use-embeddings is enabled.",
    )
    subparser.add_argument(
        "--bertopic-local-models-dir",
        default=os.getenv("THEME_EXTRACTOR_LOCAL_MODELS_DIR", "data/models"),
        help=(
            "Directory used to resolve local embedding aliases (for example 'bge-m3' -> "
            "'<dir>/bge-m3'). Use 'none' to disable alias resolution."
        ),
    )
    subparser.add_argument(
        "--bertopic-dim-reduction",
        default=BertopicDimReduction.SVD.value,
        choices=[item.value for item in BertopicDimReduction],
        help="Dimensionality reduction strategy for BERTopic.",
    )
    subparser.add_argument(
        "--bertopic-clustering",
        default=BertopicClustering.KMEANS.value,
        choices=[item.value for item in BertopicClustering],
        help="Clustering strategy for BERTopic.",
    )
    subparser.add_argument(
        "--bertopic-nr-topics",
        default=0,
        type=int,
        help="Fixed number of topics for KMeans (>0), 0 means auto.",
    )
    subparser.add_argument(
        "--bertopic-min-topic-size",
        default=10,
        type=int,
        help="Minimum size accepted for one topic.",
    )
    subparser.add_argument(
        "--bertopic-seed",
        default=42,
        type=int,
        help="Random seed for BERTopic strategy.",
    )
    subparser.add_argument(
        "--keybert-use-embeddings",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Enable custom embeddings for KeyBERT strategy.",
    )
    subparser.add_argument(
        "--keybert-embedding-model",
        default="bge-m3",
        help="Embedding model name used when --keybert-use-embeddings is enabled.",
    )
    subparser.add_argument(
        "--keybert-local-models-dir",
        default=os.getenv("THEME_EXTRACTOR_LOCAL_MODELS_DIR", "data/models"),
        help=(
            "Directory used to resolve local embedding aliases (for example 'bge-m3' -> "
            "'<dir>/bge-m3'). Use 'none' to disable alias resolution."
        ),
    )
    subparser.add_argument(
        "--llm-provider",
        default=LlmProvider.OPENAI.value,
        choices=[provider.value for provider in LlmProvider],
        help="LLM provider to use when network mode is allowed.",
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
        "--manual-stopwords-file",
        action="append",
        default=[],
        help="Path to a YAML/CSV/text file containing extra manual stopwords. Can be repeated.",
    )
    ingest_parser.add_argument(
        "--default-stopwords",
        default=_env_bool("THEME_EXTRACTOR_DEFAULT_STOPWORDS_ENABLED", default_value=True),
        action=argparse.BooleanOptionalAction,
        help="Enable default FR/EN stopwords loaded from nltk (or fallback lists).",
    )
    ingest_parser.add_argument(
        "--auto-stopwords",
        default=_env_bool("THEME_EXTRACTOR_AUTO_STOPWORDS_ENABLED", default_value=False),
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
    _add_baseline_strategy_flags(extract_parser)
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
    _add_baseline_strategy_flags(benchmark_parser)
    _add_shared_runtime_flags(benchmark_parser)
    _add_output_flag(benchmark_parser)
    benchmark_parser.set_defaults(handler=_handle_benchmark)


def _build_doctor_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the doctor subcommand parser.

    Args:
        subparsers (argparse._SubParsersAction[argparse.ArgumentParser]): Subparser registry.

    """
    doctor_parser = subparsers.add_parser(
        CommandName.DOCTOR.value,
        help="Validate local runtime, optional dependencies, and connectivity readiness.",
    )
    doctor_parser.add_argument(
        "--check-backend",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Try one backend query to validate connectivity and credentials.",
    )
    doctor_parser.add_argument(
        "--local-models-dir",
        default=os.getenv("THEME_EXTRACTOR_LOCAL_MODELS_DIR", "data/models"),
        help="Local models directory used for offline embedding alias checks.",
    )
    doctor_parser.add_argument(
        "--expected-local-models",
        default="bge-m3,all-MiniLM-L6-v2",
        help="Comma-separated model aliases expected to exist under local models dir.",
    )
    _add_shared_runtime_flags(doctor_parser)
    _add_output_flag(doctor_parser)
    doctor_parser.set_defaults(handler=_handle_doctor)


def _build_report_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the report subcommand parser.

    Args:
        subparsers (argparse._SubParsersAction[argparse.ArgumentParser]): Subparser registry.

    """
    report_parser = subparsers.add_parser(
        CommandName.REPORT.value,
        help="Generate a markdown report from one extract/benchmark JSON output.",
    )
    report_parser.add_argument(
        "--input",
        required=True,
        help="Input JSON file generated by extract or benchmark commands.",
    )
    report_parser.add_argument(
        "--title",
        default=None,
        help="Optional markdown report title override.",
    )
    report_parser.add_argument(
        "--output",
        default="-",
        help="Output markdown path or '-' for stdout.",
    )
    report_parser.set_defaults(handler=_handle_report)


def _build_evaluate_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the evaluate subcommand parser.

    Args:
        subparsers (argparse._SubParsersAction[argparse.ArgumentParser]): Subparser registry.

    """
    evaluate_parser = subparsers.add_parser(
        CommandName.EVALUATE.value,
        help="Compute quantitative proxy metrics from extract/benchmark JSON payloads.",
    )
    evaluate_parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Input JSON path. Repeat flag to evaluate multiple files.",
    )
    _add_output_flag(evaluate_parser)
    evaluate_parser.set_defaults(handler=_handle_evaluate)


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
    _build_doctor_parser(subparsers)
    _build_report_parser(subparsers)
    _build_evaluate_parser(subparsers)

    return parser


def _module_available(module_name: str) -> bool:
    """Check whether one importable module is available in runtime.

    Args:
        module_name (str): Importable module name.

    Returns:
        bool: True if import succeeds, else False.

    """
    try:
        import_module(module_name)
    except Exception:
        return False
    return True


def _handle_doctor(args: argparse.Namespace) -> dict[str, Any]:
    """Build diagnostics payload for runtime readiness checks.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        dict[str, Any]: Doctor diagnostics payload.

    """
    optional_dependencies: dict[str, dict[str, Any]] = {}
    for group_name, modules in _DOCTOR_OPTIONAL_MODULES.items():
        missing_modules = [module_name for module_name in modules if not _module_available(module_name)]
        optional_dependencies[group_name] = {
            "ok": not missing_modules,
            "modules": list(modules),
            "missing_modules": missing_modules,
        }

    expected_local_models = [
        token.strip() for token in str(args.expected_local_models).split(",") if token.strip()
    ]
    local_models_dir = Path(str(args.local_models_dir)).expanduser().resolve()
    local_models: dict[str, Any] = {
        "path": str(local_models_dir),
        "exists": local_models_dir.exists(),
        "expected_aliases": expected_local_models,
        "present_aliases": [],
        "missing_aliases": [],
    }
    if local_models_dir.exists():
        local_models["present_aliases"] = sorted(
            child.name for child in local_models_dir.iterdir() if child.is_dir()
        )
    local_models["missing_aliases"] = sorted(
        alias for alias in expected_local_models if alias not in set(local_models["present_aliases"])
    )

    backend_connectivity: dict[str, Any] = {
        "checked": bool(args.check_backend),
        "ok": None,
        "error": None,
    }
    if args.check_backend:
        try:
            backend = _build_baseline_backend(
                args=args,
                backend=BackendName(str(args.backend)),
            )
            backend.search_documents(
                index=str(args.index),
                body={"size": 1, "query": {"match_all": {}}},
            )
            backend_connectivity["ok"] = True
        except Exception as exc:
            backend_connectivity["ok"] = False
            backend_connectivity["error"] = f"{exc.__class__.__name__}: {exc}"

    checks: dict[str, dict[str, Any]] = {
        "runtime": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "offline_policy": OfflinePolicy(str(args.offline_policy)).value,
            "backend": BackendName(str(args.backend)).value,
            "backend_url": str(args.backend_url),
            "index": str(args.index),
            "proxy_url": str(args.proxy_url) if args.proxy_url else None,
        },
        "optional_dependencies": optional_dependencies,
        "local_models": local_models,
        "backend_connectivity": backend_connectivity,
    }
    has_missing_optional = any(not check["ok"] for check in optional_dependencies.values())
    has_backend_error = backend_connectivity["checked"] and backend_connectivity["ok"] is False
    missing_local_models = bool(local_models["missing_aliases"])
    status = "ok"
    if has_missing_optional or missing_local_models:
        status = "warning"
    if has_backend_error:
        status = "error"

    warnings_count = int(has_missing_optional) + int(missing_local_models)
    errors_count = int(has_backend_error)

    return {
        "schema_version": "1.0",
        "command": CommandName.DOCTOR.value,
        "status": status,
        "checks": checks,
        "summary": {
            "warnings": warnings_count,
            "errors": errors_count,
        },
    }


def _handle_report(args: argparse.Namespace) -> str:
    """Build markdown report from one extraction/benchmark JSON payload.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        str: Markdown report content.

    """
    input_path = Path(str(args.input)).expanduser().resolve()
    return render_report_markdown(input_path=input_path, title=args.title)


def _handle_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Build quantitative proxy metrics from one or many JSON payloads.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        dict[str, Any]: Quantitative metrics payload.

    """
    input_paths = [Path(str(path)).expanduser().resolve() for path in args.input]
    return evaluate_payload_files(input_paths)


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
    manual_stopwords_files = [Path(path).expanduser().resolve() for path in args.manual_stopwords_file]
    config = IngestionConfig(
        input_path=Path(args.input).expanduser().resolve(),
        recursive=bool(args.recursive),
        cleaning_options=cleaning_flag_from_string(str(args.cleaning_options)),
        manual_stopwords=manual_stopwords,
        manual_stopwords_files=manual_stopwords_files,
        default_stopwords_enabled=bool(args.default_stopwords),
        auto_stopwords_enabled=bool(args.auto_stopwords),
        auto_stopwords_min_doc_ratio=float(args.auto_stopwords_min_doc_ratio),
        auto_stopwords_min_corpus_ratio=float(args.auto_stopwords_min_corpus_ratio),
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

    Raises:
        ValueError: If method is not wired in extract execution flow.

    """
    method = parse_extract_method(args.method)
    focus = OutputFocus(args.focus)
    offline_policy = OfflinePolicy(args.offline_policy)
    backend = BackendName(args.backend)
    baseline_config = _build_baseline_config(args)
    bertopic_config = _build_bertopic_config(args)
    keybert_config = _build_keybert_config(args)
    llm_config = _build_llm_config(args)

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
    output = UnifiedExtractionOutput(
        focus=focus,
        topics=[],
        document_topics=document_topics,
        notes=[
            "Topic-first unified schema is active.",
            "Document-topic links are optional and method-dependent.",
        ],
        metadata=metadata,
    )
    if _is_baseline_method(method):
        search_backend = _build_baseline_backend(args=args, backend=backend)
        output = run_baseline_method(
            backend=search_backend,
            request=BaselineRunRequest(
                method=method,
                index=str(args.index),
                focus=focus,
                config=baseline_config,
            ),
            output=output,
        )
        return characterize_output(output)
    if method == ExtractMethod.KEYBERT:
        search_backend = _build_baseline_backend(args=args, backend=backend)
        output = run_keybert_method(
            backend=search_backend,
            request=KeyBertRunRequest(
                index=str(args.index),
                focus=focus,
                config=baseline_config,
                keybert_config=keybert_config,
            ),
            output=output,
        )
        return characterize_output(output)
    if method == ExtractMethod.BERTOPIC:
        search_backend = _build_baseline_backend(args=args, backend=backend)
        output = run_bertopic_method(
            backend=search_backend,
            request=BertopicRunRequest(
                index=str(args.index),
                focus=focus,
                baseline_config=baseline_config,
                bertopic_config=bertopic_config,
            ),
            output=output,
        )
        return characterize_output(output)
    if method == ExtractMethod.LLM:
        search_backend = _build_baseline_backend(args=args, backend=backend)
        output = run_llm_method(
            backend=search_backend,
            request=LlmRunRequest(
                index=str(args.index),
                focus=focus,
                offline_policy=offline_policy,
                baseline_config=baseline_config,
                llm_config=llm_config,
            ),
            output=output,
        )
        return characterize_output(output)

    raise ValueError(_UNSUPPORTED_EXTRACT_METHOD_ERROR.format(method=method))


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
    baseline_config = _build_baseline_config(args)
    bertopic_config = _build_bertopic_config(args)
    keybert_config = _build_keybert_config(args)
    llm_config = _build_llm_config(args)
    search_backend = None
    if any(_is_search_driven_method(method) for method in methods):
        search_backend = _build_baseline_backend(args=args, backend=backend)

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
        output = UnifiedExtractionOutput(
            focus=focus,
            topics=[],
            document_topics=document_topics,
            notes=[
                "Topic-first unified schema is active.",
                "Benchmark execution engine runs one method at a time.",
            ],
            metadata=metadata,
        )
        if _is_baseline_method(method) and search_backend is not None:
            output = run_baseline_method(
                backend=search_backend,
                request=BaselineRunRequest(
                    method=method,
                    index=str(args.index),
                    focus=focus,
                    config=baseline_config,
                ),
                output=output,
            )
        elif method == ExtractMethod.KEYBERT and search_backend is not None:
            output = run_keybert_method(
                backend=search_backend,
                request=KeyBertRunRequest(
                    index=str(args.index),
                    focus=focus,
                    config=baseline_config,
                    keybert_config=keybert_config,
                ),
                output=output,
            )
        elif method == ExtractMethod.BERTOPIC and search_backend is not None:
            output = run_bertopic_method(
                backend=search_backend,
                request=BertopicRunRequest(
                    index=str(args.index),
                    focus=focus,
                    baseline_config=baseline_config,
                    bertopic_config=bertopic_config,
                ),
                output=output,
            )
        elif method == ExtractMethod.LLM and search_backend is not None:
            output = run_llm_method(
                backend=search_backend,
                request=LlmRunRequest(
                    index=str(args.index),
                    focus=focus,
                    offline_policy=offline_policy,
                    baseline_config=baseline_config,
                    llm_config=llm_config,
                ),
                output=output,
            )

        outputs[method.value] = characterize_output(output)

    return BenchmarkOutput(
        methods=methods,
        outputs=outputs,
        comparison=build_benchmark_comparison(outputs),
    )


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

    _apply_proxy_environment(getattr(args, "proxy_url", None))
    payload = handler(args)
    _emit_payload(payload=payload, output=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
