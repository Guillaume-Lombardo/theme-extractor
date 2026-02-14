"""Backend indexing workflow used by the ingest command."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from theme_extractor.domain import MsgAttachmentPolicy, cleaning_flag_from_string
from theme_extractor.ingestion.cleaning import (
    apply_cleaning_options,
    get_default_stopwords,
    normalize_french_accents,
    tokenize_for_ingestion,
)
from theme_extractor.ingestion.extractors import MsgExtractionOptions, PdfOcrOptions, extract_text

if TYPE_CHECKING:
    import argparse

_BACKEND_RESET_ERROR = "Failed to reset index '{index}' on backend '{backend_url}': {error}"
_BULK_INDEX_ERRORS_MESSAGE = "Bulk indexing reported errors."
_logger = logging.getLogger(__name__)


def ingest_index_mapping_body() -> dict[str, Any]:
    """Build index mapping used by `ingest --reset-index`.

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


def reset_backend_index(*, backend_url: str, index: str) -> None:
    """Reset backend index by deleting and recreating it.

    Args:
        backend_url (str): Backend base URL.
        index (str): Target index name.

    Raises:
        RuntimeError: If backend reset operation fails.

    """
    index_url = f"{backend_url.rstrip('/')}/{index}"
    try:
        with httpx.Client(timeout=60.0) as client:
            delete_response = client.delete(index_url)
            if delete_response.status_code not in {200, 202, 404}:
                delete_response.raise_for_status()
            create_response = client.put(index_url, json=ingest_index_mapping_body())
            create_response.raise_for_status()
    except Exception as exc:  # pragma: no cover - network/backend dependent
        raise RuntimeError(
            _BACKEND_RESET_ERROR.format(
                index=index,
                backend_url=backend_url,
                error=f"{exc.__class__.__name__}: {exc}",
            ),
        ) from exc


def effective_ingest_stopwords(
    *,
    default_stopwords_enabled: bool,
    manual_stopwords: set[str],
    auto_stopwords: list[str],
) -> set[str]:
    """Build normalized stopwords used for backend indexing.

    Args:
        default_stopwords_enabled (bool): Whether default stopwords are enabled.
        manual_stopwords (set[str]): Manual stopwords from CLI/files.
        auto_stopwords (list[str]): Auto-generated stopwords from ingestion.

    Returns:
        set[str]: Normalized stopwords.

    """
    normalized_manual = {normalize_french_accents(term.strip().lower()) for term in manual_stopwords}
    normalized_auto = {normalize_french_accents(term.strip().lower()) for term in auto_stopwords}
    default_stopwords = get_default_stopwords() if default_stopwords_enabled else set()
    return default_stopwords | normalized_manual | normalized_auto


def pdf_ocr_options_from_args(args: argparse.Namespace) -> PdfOcrOptions:
    """Build PDF OCR options from parsed CLI args.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        PdfOcrOptions: OCR options.

    """
    return PdfOcrOptions(
        fallback_enabled=bool(args.pdf_ocr_fallback),
        languages=str(args.pdf_ocr_languages),
        dpi=int(args.pdf_ocr_dpi),
        min_chars=int(args.pdf_ocr_min_chars),
        tessdata=None if args.pdf_ocr_tessdata in {None, ""} else str(args.pdf_ocr_tessdata),
    )


def msg_options_from_args(args: argparse.Namespace) -> MsgExtractionOptions:
    """Build `.msg` extraction options from parsed CLI args.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        MsgExtractionOptions: `.msg` extraction options.

    """
    return MsgExtractionOptions(
        include_metadata=bool(args.msg_include_metadata),
        attachments_policy=MsgAttachmentPolicy(str(args.msg_attachments_policy)),
    )


def build_ingest_index_documents(
    *,
    args: argparse.Namespace,
    result_payload: dict[str, Any],
    stopwords: set[str],
) -> list[dict[str, Any]]:
    """Build bulk-ready backend documents from an ingestion result payload.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        result_payload (dict[str, Any]): Ingestion payload dictionary.
        stopwords (set[str]): Effective stopwords used for filtered extraction fields.

    Returns:
        list[dict[str, Any]]: Backend-ready documents.

    """
    prepared_docs: list[dict[str, Any]] = []
    for item in result_payload.get("documents", []):
        path = Path(str(item["path"]))
        try:
            raw_text = extract_text(
                path,
                pdf_ocr=pdf_ocr_options_from_args(args),
                msg_options=msg_options_from_args(args),
            )
        except Exception as exc:
            _logger.warning("Skipping backend indexing for unreadable file '%s': %s", path, exc)
            continue

        cleaned_text = apply_cleaning_options(
            raw_text,
            options=cleaning_flag_from_string(str(args.cleaning_options)),
        )
        tokens_all = tokenize_for_ingestion(cleaned_text)
        filtered_tokens = [token for token in tokens_all if token not in stopwords]

        prepared_docs.append(
            {
                "_id": str(item["document_id"]),
                "_source": {
                    "path": str(path),
                    "filename": path.name,
                    "extension": path.suffix.lower(),
                    "content": " ".join(filtered_tokens),
                    "content_raw": raw_text,
                    "content_clean": cleaned_text,
                    "tokens": filtered_tokens,
                    "tokens_all": tokens_all,
                    "removed_stopword_count": len(tokens_all) - len(filtered_tokens),
                },
            },
        )

    return prepared_docs


def bulk_index_documents(
    *,
    backend_url: str,
    index: str,
    docs: list[dict[str, Any]],
) -> None:
    """Bulk index prepared documents.

    Args:
        backend_url (str): Backend base URL.
        index (str): Target index name.
        docs (list[dict[str, Any]]): Documents to index.

    Raises:
        RuntimeError: If backend bulk response reports indexing errors.

    """
    if not docs:
        return

    lines: list[str] = []
    for doc in docs:
        lines.extend(
            [
                json.dumps({"index": {"_index": index, "_id": doc["_id"]}}, ensure_ascii=False),
                json.dumps(doc["_source"], ensure_ascii=False),
            ],
        )

    payload = "\n".join(lines) + "\n"
    bulk_url = f"{backend_url.rstrip('/')}/_bulk?refresh=true"
    with httpx.Client(timeout=60.0) as client:
        response = client.post(
            bulk_url,
            content=payload.encode("utf-8"),
            headers={"Content-Type": "application/x-ndjson"},
        )
        response.raise_for_status()
        response_payload = response.json()

    if response_payload.get("errors"):
        raise RuntimeError(_BULK_INDEX_ERRORS_MESSAGE)
