"""Index local corpus files into Elasticsearch or OpenSearch for baseline testing."""

from __future__ import annotations

import argparse
import json
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

from theme_extractor.domain import CleaningOptionFlag, default_cleaning_options  # noqa: E402
from theme_extractor.ingestion.cleaning import apply_cleaning_options, tokenize_for_ingestion  # noqa: E402
from theme_extractor.ingestion.extractors import extract_text, supported_suffixes  # noqa: E402

_BULK_ERRORS_MESSAGE = "Bulk indexing reported errors."
_BACKEND_CONNECTION_ERROR = (
    "Cannot connect to backend at '{backend_url}'. "
    "Start Docker services first, for example: "
    "'docker compose -f docker/compose.elasticsearch.yaml up -d'."
)


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
        help="Cleaning options to apply before indexing (all or none for quick tests).",
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
                "tokens": {"type": "keyword"},
            },
        },
    }


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
        lines.extend(
            [
                f'{{"index":{{"_index":"{index}","_id":"{doc["_id"]}"}}}}',
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
    args = build_parser().parse_args(argv)

    input_root = Path(args.input).expanduser().resolve()
    files = _iter_supported_files(input_root)
    if not files:
        print(f"No supported files found under: {input_root}")
        return 1

    cleaning_options = default_cleaning_options()
    if str(args.cleaning_options).strip().lower() == "none":
        cleaning_options = CleaningOptionFlag.NONE

    docs: list[dict[str, Any]] = []
    for path in files:
        try:
            text = extract_text(path)
        except Exception as exc:
            print(f"Skip {path}: {exc}")
            continue

        cleaned_text = apply_cleaning_options(text, options=cleaning_options)
        tokens = tokenize_for_ingestion(cleaned_text)
        docs.append(
            {
                "_id": str(path.relative_to(input_root)),
                "_source": {
                    "path": str(path),
                    "filename": path.name,
                    "extension": path.suffix.lower(),
                    "content": cleaned_text,
                    "tokens": tokens,
                },
            },
        )

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
        except httpx.ConnectError as exc:
            print(_BACKEND_CONNECTION_ERROR.format(backend_url=str(args.backend_url)))
            print(f"Details: {exc}")
            return 1

    print(f"Indexed {len(docs)} documents into index '{args.index}'.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
