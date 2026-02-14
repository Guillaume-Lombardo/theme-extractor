"""Unified CLI entrypoint for ingestion, extraction, and benchmarking workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from theme_extractor.cli_common import apply_proxy_environment, emit_payload
from theme_extractor.cli_ingest_backend import (
    build_ingest_index_documents,
    bulk_index_documents,
    reset_backend_index,
)
from theme_extractor.cli_parser import build_parser

if TYPE_CHECKING:
    import argparse


def _reset_backend_index(*, backend_url: str, index: str) -> None:
    """Backward-compatible wrapper around backend index reset."""
    reset_backend_index(backend_url=backend_url, index=index)


def _build_ingest_index_documents(
    *,
    args: argparse.Namespace,
    result_payload: dict[str, Any],
    stopwords: set[str],
) -> list[dict[str, Any]]:
    """Backward-compatible wrapper around ingest indexing document builder.

    Returns:
        list[dict[str, Any]]: Backend-ready documents.

    """
    return build_ingest_index_documents(args=args, result_payload=result_payload, stopwords=stopwords)


def _bulk_index_documents(
    *,
    backend_url: str,
    index: str,
    docs: list[dict[str, Any]],
) -> None:
    """Backward-compatible wrapper around backend bulk indexing."""
    bulk_index_documents(backend_url=backend_url, index=index, docs=docs)


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

    apply_proxy_environment(getattr(args, "proxy_url", None))
    payload = handler(args)
    emit_payload(payload=payload, output=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
