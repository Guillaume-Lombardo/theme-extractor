"""Index local corpus files into Elasticsearch or OpenSearch for baseline testing."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import httpx

# Ensure local package imports work when running this script from repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from theme_extractor.domain import (  # noqa: E402
    cleaning_flag_from_string,
)
from theme_extractor.ingestion.cleaning import (  # noqa: E402
    apply_cleaning_options,
    discover_auto_stopwords,
    get_default_stopwords,
    load_stopwords_from_files,
    normalize_french_accents,
    tokenize_for_ingestion,
)
from theme_extractor.ingestion.extractors import extract_text, supported_suffixes  # noqa: E402

_BULK_ERRORS_MESSAGE = "Bulk indexing reported errors."
_BACKEND_CONNECTION_ERROR = (
    "Cannot connect to backend at '{backend_url}'. "
    "Start Docker services first, for example: "
    "'docker compose -f docker/compose.elasticsearch.yaml up -d'."
)
_DEFAULT_LOG_FORMAT = "%(levelname)s %(name)s - %(message)s"
_logger = logging.getLogger(__name__)


def _env_bool(name: str, *, default_value: bool) -> bool:
    """Read a boolean environment variable.

    Args:
        name (str): Environment variable name.
        default_value (bool): Fallback value.

    Returns:
        bool: Parsed value.

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


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for corpus indexing helper.

    Returns:
        argparse.ArgumentParser: Configured parser.

    """
    parser = argparse.ArgumentParser(prog="index-corpus")
    parser.add_argument(
        "--input",
        default=os.getenv("THEME_EXTRACTOR_INPUT_DIR", "data/raw"),
        help="Input directory containing sample documents.",
    )
    parser.add_argument(
        "--backend-url",
        default=os.getenv("THEME_EXTRACTOR_BACKEND_URL", "http://localhost:9200"),
        help="Backend URL.",
    )
    parser.add_argument(
        "--index",
        default=os.getenv("THEME_EXTRACTOR_INDEX", "theme_extractor"),
        help="Index name to create/populate.",
    )
    parser.add_argument(
        "--cleaning-options",
        default="all",
        help=(
            "Cleaning options to apply before indexing. "
            "Available: all, none, whitespace, accent_normalization, header_footer, "
            "boilerplate, token_cleanup, html_strip."
        ),
    )
    parser.add_argument(
        "--manual-stopwords",
        default="",
        help="Comma-separated manual stopwords to remove from indexed content/tokens.",
    )
    parser.add_argument(
        "--manual-stopwords-file",
        action="append",
        default=[],
        help="Path to a YAML/JSON/CSV/text file containing extra manual stopwords. Can be repeated.",
    )
    parser.add_argument(
        "--default-stopwords",
        default=_env_bool("THEME_EXTRACTOR_DEFAULT_STOPWORDS_ENABLED", default_value=True),
        action=argparse.BooleanOptionalAction,
        help="Enable default FR/EN stopwords loaded from nltk (or fallback lists).",
    )
    parser.add_argument(
        "--auto-stopwords",
        default=_env_bool("THEME_EXTRACTOR_AUTO_STOPWORDS_ENABLED", default_value=False),
        action=argparse.BooleanOptionalAction,
        help="Enable automatic stopwords generation from corpus statistics.",
    )
    parser.add_argument(
        "--auto-stopwords-min-doc-ratio",
        default=os.getenv("THEME_EXTRACTOR_AUTO_STOPWORDS_MIN_DOC_RATIO", "0.7"),
        type=float,
        help="Minimum document ratio for auto stopwords generation.",
    )
    parser.add_argument(
        "--auto-stopwords-max-terms",
        default=os.getenv("THEME_EXTRACTOR_AUTO_STOPWORDS_MAX_TERMS", "200"),
        type=int,
        help="Maximum number of automatically generated stopwords.",
    )
    parser.add_argument(
        "--auto-stopwords-min-corpus-ratio",
        default=os.getenv("THEME_EXTRACTOR_AUTO_STOPWORDS_MIN_CORPUS_RATIO", "0.01"),
        type=float,
        help="Minimum corpus frequency ratio for auto stopwords generation.",
    )
    parser.add_argument(
        "--reset-index",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Drop and recreate index before indexing documents.",
    )
    return parser


def _iter_supported_files(root: Path) -> list[Path]:
    """List supported files under root.

    Args:
        root (Path): Input root path.

    Returns:
        list[Path]: Files with supported suffixes.

    """
    suffixes = supported_suffixes()
    if root.is_file():
        return [root] if root.suffix.lower() in suffixes else []
    return [path for path in sorted(root.glob("**/*")) if path.is_file() and path.suffix.lower() in suffixes]


def _mapping_body() -> dict[str, Any]:
    """Build index mapping body.

    Returns:
        dict[str, Any]: Mapping payload.

    """
    return {
        "mappings": {
            "properties": {
                "path": {"type": "keyword"},
                "filename": {"type": "keyword"},
                "extension": {"type": "keyword"},
                "content": {"type": "text"},
                "content_raw": {"type": "text"},
                "content_clean": {"type": "text"},
                "tokens": {"type": "keyword"},
                "tokens_all": {"type": "keyword"},
                "removed_stopword_count": {"type": "integer"},
            },
        },
    }


def _parse_manual_stopwords(value: str) -> set[str]:
    """Parse inline comma-separated stopwords.

    Args:
        value (str): Raw comma-separated stopwords.

    Returns:
        set[str]: Normalized stopwords.

    """
    return {normalize_french_accents(token.strip().lower()) for token in value.split(",") if token.strip()}


def _build_stopwords(  # noqa: PLR0913
    *,
    tokenized_documents: list[list[str]],
    manual_stopwords: set[str],
    manual_stopwords_files: list[Path],
    default_stopwords_enabled: bool,
    auto_stopwords_enabled: bool,
    auto_stopwords_min_doc_ratio: float,
    auto_stopwords_min_corpus_ratio: float,
    auto_stopwords_max_terms: int,
) -> set[str]:
    """Build merged stopword set for indexing.

    Args:
        tokenized_documents (list[list[str]]): Tokenized corpus documents.
        manual_stopwords (set[str]): Inline stopwords.
        manual_stopwords_files (list[Path]): Stopword files.
        default_stopwords_enabled (bool): Whether default stopwords are enabled.
        auto_stopwords_enabled (bool): Whether auto stopwords are enabled.
        auto_stopwords_min_doc_ratio (float): Auto stopwords minimum document ratio.
        auto_stopwords_min_corpus_ratio (float): Auto stopwords minimum corpus ratio.
        auto_stopwords_max_terms (int): Auto stopwords max terms.

    Returns:
        set[str]: Merged stopwords.

    """
    auto_stopwords: set[str] = set()
    if auto_stopwords_enabled:
        auto_stopwords = discover_auto_stopwords(
            tokenized_documents,
            min_doc_ratio=auto_stopwords_min_doc_ratio,
            min_corpus_ratio=auto_stopwords_min_corpus_ratio,
            max_terms=auto_stopwords_max_terms,
        )

    file_stopwords = load_stopwords_from_files(manual_stopwords_files)
    default_stopwords = get_default_stopwords() if default_stopwords_enabled else set()
    return default_stopwords | manual_stopwords | file_stopwords | auto_stopwords


def _document_id_for_indexing(*, path: Path, input_root: Path) -> str:
    """Build stable document id used in index bulk operations.

    Args:
        path (Path): Indexed file path.
        input_root (Path): User input path.

    Returns:
        str: Document identifier.

    """
    if input_root.is_dir():
        return str(path.relative_to(input_root))
    return path.name


def _build_index_documents(
    *,
    prepared_documents: list[dict[str, Any]],
    stopwords: set[str],
) -> list[dict[str, Any]]:
    """Build final index documents with stopwords filtered for extraction fields.

    Args:
        prepared_documents (list[dict[str, Any]]): Prepared docs with cleaned text and unfiltered tokens.
        stopwords (set[str]): Effective stopwords.

    Returns:
        list[dict[str, Any]]: Bulk-ready documents.

    """
    indexed_docs: list[dict[str, Any]] = []
    for doc in prepared_documents:
        source = doc["_source"]
        tokens_all = list(source["tokens_all"])
        filtered_tokens = [token for token in tokens_all if token not in stopwords]
        indexed_docs.append(
            {
                "_id": doc["_id"],
                "_source": {
                    "path": source["path"],
                    "filename": source["filename"],
                    "extension": source["extension"],
                    # `content` and `tokens` are consumed by extraction defaults.
                    "content": " ".join(filtered_tokens),
                    "content_raw": source["content_raw"],
                    "tokens": filtered_tokens,
                    # Keep pre-stopwords data for debugging.
                    "content_clean": source["content_clean"],
                    "tokens_all": tokens_all,
                    "removed_stopword_count": len(tokens_all) - len(filtered_tokens),
                },
            },
        )
    return indexed_docs


def _reset_index(*, client: httpx.Client, backend_url: str, index: str) -> None:
    """Reset index.

    Args:
        client (httpx.Client): HTTP client.
        backend_url (str): Backend URL.
        index (str): Index name.

    """
    index_url = f"{backend_url.rstrip('/')}/{index}"
    client.delete(index_url)
    create_response = client.put(index_url, json=_mapping_body())
    create_response.raise_for_status()


def _bulk_index(
    *,
    client: httpx.Client,
    backend_url: str,
    index: str,
    docs: list[dict[str, Any]],
) -> None:
    """Bulk index prepared docs.

    Args:
        client (httpx.Client): HTTP client.
        backend_url (str): Backend URL.
        index (str): Index name.
        docs (list[dict[str, Any]]): Documents to index.

    Raises:
        RuntimeError: If backend bulk response reports errors.

    """
    if not docs:
        return

    lines: list[str] = []
    for doc in docs:
        action = {"index": {"_index": index, "_id": doc["_id"]}}
        lines.extend(
            [
                json.dumps(action, ensure_ascii=False),
                json.dumps(doc["_source"], ensure_ascii=False),
            ],
        )

    payload = "\n".join(lines) + "\n"
    bulk_url = f"{backend_url.rstrip('/')}/_bulk?refresh=true"
    response = client.post(
        bulk_url,
        content=payload.encode("utf-8"),
        headers={"Content-Type": "application/x-ndjson"},
    )
    response.raise_for_status()
    result = response.json()
    if result.get("errors"):
        raise RuntimeError(_BULK_ERRORS_MESSAGE)


def main(argv: list[str] | None = None) -> int:
    """Run corpus indexing helper.

    Args:
        argv (list[str] | None): Optional argv.

    Returns:
        int: Process exit code.

    """
    logging.basicConfig(
        level=os.getenv("THEME_EXTRACTOR_LOG_LEVEL", "INFO").upper(),
        format=_DEFAULT_LOG_FORMAT,
    )
    args = build_parser().parse_args(argv)

    input_root = Path(args.input).expanduser().resolve()
    files = _iter_supported_files(input_root)
    if not files:
        _logger.warning("No supported files found under '%s'.", input_root)
        return 1

    try:
        cleaning_options = cleaning_flag_from_string(str(args.cleaning_options))
    except ValueError:
        _logger.exception("Invalid --cleaning-options value.")
        return 1

    prepared_docs: list[dict[str, Any]] = []
    for path in files:
        try:
            text = extract_text(path)
        except Exception as exc:
            _logger.warning("Skip '%s': %s", path, exc)
            continue

        cleaned_text = apply_cleaning_options(text, options=cleaning_options)
        tokens_all = tokenize_for_ingestion(cleaned_text)
        prepared_docs.append(
            {
                "_id": _document_id_for_indexing(path=path, input_root=input_root),
                "_source": {
                    "path": str(path),
                    "filename": path.name,
                    "extension": path.suffix.lower(),
                    "content_raw": text,
                    "content_clean": cleaned_text,
                    "tokens_all": tokens_all,
                },
            },
        )
    tokenized_documents = [list(doc["_source"]["tokens_all"]) for doc in prepared_docs]
    manual_stopwords_files = [Path(path).expanduser().resolve() for path in args.manual_stopwords_file]
    stopwords = _build_stopwords(
        tokenized_documents=tokenized_documents,
        manual_stopwords=_parse_manual_stopwords(str(args.manual_stopwords)),
        manual_stopwords_files=manual_stopwords_files,
        default_stopwords_enabled=bool(args.default_stopwords),
        auto_stopwords_enabled=bool(args.auto_stopwords),
        auto_stopwords_min_doc_ratio=float(args.auto_stopwords_min_doc_ratio),
        auto_stopwords_min_corpus_ratio=float(args.auto_stopwords_min_corpus_ratio),
        auto_stopwords_max_terms=int(args.auto_stopwords_max_terms),
    )
    _logger.debug("Built %d stopwords for indexing.", len(stopwords))
    docs = _build_index_documents(prepared_documents=prepared_docs, stopwords=stopwords)

    with httpx.Client(timeout=60.0) as client:
        try:
            if args.reset_index:
                _reset_index(client=client, backend_url=str(args.backend_url), index=str(args.index))
            _bulk_index(
                client=client,
                backend_url=str(args.backend_url),
                index=str(args.index),
                docs=docs,
            )
        except httpx.ConnectError:
            _logger.exception(
                "%s",
                _BACKEND_CONNECTION_ERROR.format(backend_url=str(args.backend_url)),
            )
            return 1

    _logger.info(
        "Indexed %d documents into index '%s' with %d stopwords.",
        len(docs),
        args.index,
        len(stopwords),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
