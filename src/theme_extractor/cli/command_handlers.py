"""CLI command handlers and command-scoped configuration builders."""

from __future__ import annotations

import platform
import sys
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any

from theme_extractor.cli.ingest_backend_indexing import (
    build_ingest_index_documents,
    bulk_index_documents,
    effective_ingest_stopwords,
    reset_backend_index,
)
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
    MsgAttachmentPolicy,
    OfflinePolicy,
    OutputFocus,
    UnifiedExtractionOutput,
    cleaning_flag_from_string,
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
from theme_extractor.ingestion.cleaning import normalize_french_accents
from theme_extractor.reporting import render_report_markdown
from theme_extractor.search.factory import build_search_backend

if TYPE_CHECKING:
    import argparse

    from theme_extractor.search.protocols import SearchBackend

_DEFAULT_METHODS = "baseline_tfidf,keybert,bertopic,llm"
_DEFAULT_BASELINE_FIELDS = ("content", "filename", "path")
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
_MISSING_BASELINE_FIELDS_ERROR = "At least one baseline field must be provided."
_MISSING_BENCHMARK_METHODS_ERROR = (
    "At least one extraction method must be provided for benchmark execution."
)


def parse_baseline_fields(value: str) -> tuple[str, ...]:
    """Parse and normalize comma-separated baseline fields.

    Args:
        value (str): Comma-separated fields.

    Returns:
        tuple[str, ...]: Normalized fields.

    Raises:
        ValueError: If no valid field is found.

    """
    fields = [field.strip() for field in value.split(",") if field.strip()]
    if not fields:
        raise ValueError(_MISSING_BASELINE_FIELDS_ERROR)
    return tuple(fields)


def build_baseline_config(args: argparse.Namespace) -> BaselineExtractionConfig:
    """Build baseline strategy runtime config from CLI args.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        BaselineExtractionConfig: Baseline extraction configuration.

    """
    return BaselineExtractionConfig(
        query=str(args.query),
        fields=parse_baseline_fields(str(args.fields)),
        source_field=str(args.source_field),
        top_n=int(args.topn),
        search_size=int(args.search_size),
        aggregation_field=str(args.agg_field),
        terms_min_doc_count=int(args.terms_min_doc_count),
        sigtext_filter_duplicate=bool(args.sigtext_filter_duplicate),
    )


def build_bertopic_config(args: argparse.Namespace) -> BertopicExtractionConfig:
    """Build BERTopic strategy runtime config from CLI args.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        BertopicExtractionConfig: BERTopic extraction configuration.

    """
    return BertopicExtractionConfig(
        use_embeddings=bool(args.bertopic_use_embeddings),
        embedding_model=str(args.bertopic_embedding_model),
        reduce_dim=BertopicDimReduction(str(args.bertopic_dim_reduction)),
        clustering=BertopicClustering(str(args.bertopic_clustering)),
        nr_topics=None if args.bertopic_nr_topics in {None, 0} else int(args.bertopic_nr_topics),
        min_topic_size=max(1, int(args.bertopic_min_topic_size)),
        seed=int(args.bertopic_seed),
        local_models_dir=(
            None
            if args.bertopic_local_models_dir in {None, ""}
            else Path(str(args.bertopic_local_models_dir)).expanduser().resolve()
        ),
        embedding_cache_enabled=bool(args.bertopic_embedding_cache_enabled),
        embedding_cache_dir=Path(str(args.bertopic_embedding_cache_dir)).expanduser().resolve(),
        embedding_cache_version=str(args.bertopic_embedding_cache_version),
    )


def build_keybert_config(args: argparse.Namespace) -> KeyBertExtractionConfig:
    """Build KeyBERT strategy runtime config from CLI args.

    Returns:
        KeyBertExtractionConfig: KeyBERT extraction configuration.

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


def build_llm_config(args: argparse.Namespace) -> LlmExtractionConfig:
    """Build LLM strategy runtime config from CLI args.

    Returns:
        LlmExtractionConfig: LLM extraction configuration.

    """
    return LlmExtractionConfig(
        provider=LlmProvider(str(args.llm_provider)),
        model=str(args.llm_model),
        api_key_env_var=str(args.llm_api_key_env_var),
        api_base_url=None if args.llm_api_base_url in {None, ""} else str(args.llm_api_base_url),
        temperature=float(args.llm_temperature),
        timeout_s=float(args.llm_timeout_s),
        max_input_chars=max(1, int(args.llm_max_input_chars)),
    )


def build_baseline_backend(
    *,
    args: argparse.Namespace,
    backend: BackendName,
) -> SearchBackend:
    """Build thin backend adapter used by search-driven methods.

    Returns:
        SearchBackend: Search backend adapter.

    """
    return build_search_backend(
        backend=backend,
        url=str(args.backend_url),
        timeout_s=30.0,
        verify_certs=True,
    )


def is_baseline_method(method: ExtractMethod) -> bool:
    """Return whether extraction method is baseline/search aggregation driven."""
    return method in _BASELINE_METHODS


def is_search_driven_method(method: ExtractMethod) -> bool:
    """Return whether extraction method queries backend documents/aggregations."""
    return method in _SEARCH_DRIVEN_METHODS


def module_available(module_name: str) -> bool:
    """Return whether importable module is available.

    Args:
        module_name (str): Module name.

    Returns:
        bool: True when module can be imported.

    """
    try:
        import_module(module_name)
    except ImportError:
        return False
    return True


def handle_doctor(args: argparse.Namespace) -> dict[str, Any]:
    """Run environment, dependency, and backend health checks.

    Returns:
        dict[str, Any]: Doctor output payload.

    """
    optional_dependencies: dict[str, dict[str, Any]] = {}
    for group_name, modules in _DOCTOR_OPTIONAL_MODULES.items():
        missing_modules = [module_name for module_name in modules if not module_available(module_name)]
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
            backend = build_baseline_backend(
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


def handle_report(args: argparse.Namespace) -> str:
    """Build markdown report from one extraction/benchmark JSON payload.

    Returns:
        str: Markdown report content.

    """
    input_path = Path(str(args.input)).expanduser().resolve()
    return render_report_markdown(input_path=input_path, title=args.title)


def handle_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Build quantitative proxy metrics from one or many JSON payloads.

    Returns:
        dict[str, Any]: Evaluation metrics payload.

    """
    input_paths = [Path(str(path)).expanduser().resolve() for path in args.input]
    return evaluate_payload_files(input_paths)


def handle_ingest(args: argparse.Namespace) -> dict[str, Any]:
    """Build a normalized JSON payload for ingestion runs.

    Returns:
        dict[str, Any]: Ingestion payload.

    """
    manual_stopwords = {
        normalize_french_accents(word.strip().lower())
        for word in str(args.manual_stopwords).split(",")
        if word.strip()
    }
    manual_stopwords_files = [Path(path).expanduser().resolve() for path in args.manual_stopwords_file]
    reset_index_requested = bool(args.reset_index)
    index_backend_requested = bool(args.index_backend)
    if reset_index_requested:
        reset_backend_index(backend_url=str(args.backend_url), index=str(args.index))

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
        streaming_mode=bool(args.streaming_mode),
        pdf_ocr_fallback=bool(args.pdf_ocr_fallback),
        pdf_ocr_languages=str(args.pdf_ocr_languages),
        pdf_ocr_dpi=int(args.pdf_ocr_dpi),
        pdf_ocr_min_chars=int(args.pdf_ocr_min_chars),
        pdf_ocr_tessdata=None if args.pdf_ocr_tessdata in {None, ""} else str(args.pdf_ocr_tessdata),
        msg_include_metadata=bool(args.msg_include_metadata),
        msg_attachments_policy=MsgAttachmentPolicy(str(args.msg_attachments_policy)),
    )
    result = run_ingestion(config)
    payload = result.model_dump(mode="json")
    indexed_documents = 0
    if index_backend_requested:
        effective_stopwords = effective_ingest_stopwords(
            default_stopwords_enabled=bool(args.default_stopwords),
            manual_stopwords=manual_stopwords,
            auto_stopwords=list(payload.get("auto_stopwords", [])),
        )
        index_docs = build_ingest_index_documents(
            args=args,
            result_payload=payload,
            stopwords=effective_stopwords,
        )
        bulk_index_documents(
            backend_url=str(args.backend_url),
            index=str(args.index),
            docs=index_docs,
        )
        indexed_documents = len(index_docs)

    payload["schema_version"] = "1.0"
    payload["runtime"] = {
        "offline_policy": OfflinePolicy(args.offline_policy).value,
        "proxy_url": args.proxy_url,
        "backend": BackendName(args.backend).value,
        "backend_url": args.backend_url,
        "index": args.index,
        "reset_index": {
            "requested": reset_index_requested,
            "applied": reset_index_requested,
        },
        "index_backend": {
            "requested": index_backend_requested,
            "applied": index_backend_requested,
            "indexed_documents": indexed_documents,
        },
    }
    return payload


def handle_extract(args: argparse.Namespace) -> UnifiedExtractionOutput:
    """Build a normalized JSON payload for a single extraction strategy.

    Raises:
        ValueError: If method is not wired in extract execution flow.

    Returns:
        UnifiedExtractionOutput: Extraction payload.

    """
    method = parse_extract_method(args.method)
    focus = OutputFocus(args.focus)
    offline_policy = OfflinePolicy(args.offline_policy)
    backend = BackendName(args.backend)
    baseline_config = build_baseline_config(args)
    bertopic_config = build_bertopic_config(args)
    keybert_config = build_keybert_config(args)
    llm_config = build_llm_config(args)

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
    if is_baseline_method(method):
        search_backend = build_baseline_backend(args=args, backend=backend)
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
        search_backend = build_baseline_backend(args=args, backend=backend)
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
        search_backend = build_baseline_backend(args=args, backend=backend)
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
        search_backend = build_baseline_backend(args=args, backend=backend)
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


def handle_benchmark(args: argparse.Namespace) -> BenchmarkOutput:
    """Run multiple extraction strategies and build benchmark payload.

    Raises:
        ValueError: If benchmark method list is empty or unsupported.

    Returns:
        BenchmarkOutput: Benchmark payload.

    """
    methods_flag = method_flag_from_string(str(args.methods))
    methods = method_flag_to_methods(methods_flag)
    if not methods:
        raise ValueError(_MISSING_BENCHMARK_METHODS_ERROR)

    focus = OutputFocus(args.focus)
    offline_policy = OfflinePolicy(args.offline_policy)
    backend = BackendName(args.backend)
    baseline_config = build_baseline_config(args)
    bertopic_config = build_bertopic_config(args)
    keybert_config = build_keybert_config(args)
    llm_config = build_llm_config(args)

    outputs: dict[str, UnifiedExtractionOutput] = {}

    for method in methods:
        metadata = ExtractionRunMetadata(
            command=CommandName.BENCHMARK,
            method=method,
            offline_policy=offline_policy,
            backend=backend,
            index=args.index,
        )
        extraction_output = UnifiedExtractionOutput(
            focus=focus,
            topics=[],
            document_topics=[] if focus in {OutputFocus.DOCUMENTS, OutputFocus.BOTH} else None,
            notes=[
                "Topic-first unified schema is active.",
                "Benchmark execution engine runs one method at a time.",
            ],
            metadata=metadata,
        )

        if is_search_driven_method(method):
            search_backend = build_baseline_backend(args=args, backend=backend)
            if method in _BASELINE_METHODS:
                extraction_output = run_baseline_method(
                    backend=search_backend,
                    request=BaselineRunRequest(
                        method=method,
                        index=str(args.index),
                        focus=focus,
                        config=baseline_config,
                    ),
                    output=extraction_output,
                )
            elif method == ExtractMethod.KEYBERT:
                extraction_output = run_keybert_method(
                    backend=search_backend,
                    request=KeyBertRunRequest(
                        index=str(args.index),
                        focus=focus,
                        config=baseline_config,
                        keybert_config=keybert_config,
                    ),
                    output=extraction_output,
                )
            elif method == ExtractMethod.LLM:
                extraction_output = run_llm_method(
                    backend=search_backend,
                    request=LlmRunRequest(
                        index=str(args.index),
                        focus=focus,
                        offline_policy=offline_policy,
                        baseline_config=baseline_config,
                        llm_config=llm_config,
                    ),
                    output=extraction_output,
                )
        elif method == ExtractMethod.BERTOPIC:
            search_backend = build_baseline_backend(args=args, backend=backend)
            extraction_output = run_bertopic_method(
                backend=search_backend,
                request=BertopicRunRequest(
                    index=str(args.index),
                    focus=focus,
                    baseline_config=baseline_config,
                    bertopic_config=bertopic_config,
                ),
                output=extraction_output,
            )
        else:
            raise ValueError(_UNSUPPORTED_EXTRACT_METHOD_ERROR.format(method=method))

        outputs[method.value] = characterize_output(extraction_output)

    comparison = build_benchmark_comparison(outputs)
    return BenchmarkOutput(
        methods=[method.value for method in methods],
        outputs=outputs,
        comparison=comparison,
    )
