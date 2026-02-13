"""Deterministic embedding cache helpers."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.lib.format import read_array, write_array

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


def build_embedding_cache_key(
    *,
    strategy: str,
    model_name: str,
    cache_version: str,
    documents: list[str],
) -> str:
    """Build deterministic cache key for one embedding run.

    Args:
        strategy (str): Strategy identifier (for example `bertopic`).
        model_name (str): Resolved embedding model name/path.
        cache_version (str): User-controlled cache namespace version.
        documents (list[str]): Ordered input documents.

    Returns:
        str: SHA-256 cache key.

    """
    digest = hashlib.sha256()
    digest.update(strategy.encode("utf-8", errors="ignore"))
    digest.update(b"\0")
    digest.update(model_name.encode("utf-8", errors="ignore"))
    digest.update(b"\0")
    digest.update(cache_version.encode("utf-8", errors="ignore"))
    digest.update(b"\0")
    digest.update(str(len(documents)).encode("utf-8", errors="ignore"))
    digest.update(b"\0")
    for document in documents:
        encoded = document.encode("utf-8", errors="ignore")
        digest.update(str(len(encoded)).encode("utf-8", errors="ignore"))
        digest.update(b":")
        digest.update(encoded)
        digest.update(b"\0")
    return digest.hexdigest()


def load_embeddings_from_cache(
    *,
    cache_dir: Path,
    cache_key: str,
) -> NDArray | None:
    """Load embeddings from cache directory if available.

    Args:
        cache_dir (Path): Cache directory.
        cache_key (str): Deterministic cache key.

    Returns:
        NDArray | None: Cached vectors when available, else `None`.

    """
    cache_path = _cache_vectors_path(cache_dir=cache_dir, cache_key=cache_key)
    if not cache_path.exists():
        return None
    with cache_path.open("rb") as handle:
        loaded = read_array(handle, allow_pickle=False)
    return np.asarray(loaded, dtype=np.float32)


def store_embeddings_in_cache(
    *,
    cache_dir: Path,
    cache_key: str,
    vectors: NDArray,
    metadata: dict[str, Any],
) -> None:
    """Store embeddings and metadata under one deterministic cache key.

    Args:
        cache_dir (Path): Cache directory.
        cache_key (str): Deterministic cache key.
        vectors (NDArray): Embedding vectors to store.
        metadata (dict[str, Any]): Additional metadata to persist.

    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_vectors_path(cache_dir=cache_dir, cache_key=cache_key)
    metadata_path = _cache_metadata_path(cache_dir=cache_dir, cache_key=cache_key)

    with cache_path.open("wb") as handle:
        write_array(handle, np.asarray(vectors, dtype=np.float32), allow_pickle=False)

    merged_metadata = dict(metadata)
    merged_metadata["cache_key"] = cache_key
    merged_metadata["cached_at"] = datetime.now(tz=UTC).isoformat()
    merged_metadata["shape"] = list(np.asarray(vectors).shape)
    merged_metadata["dtype"] = str(np.asarray(vectors).dtype)
    metadata_path.write_text(json.dumps(merged_metadata, ensure_ascii=True, indent=2), encoding="utf-8")


def _cache_vectors_path(*, cache_dir: Path, cache_key: str) -> Path:
    """Build vectors cache path from key.

    Args:
        cache_dir (Path): Cache directory.
        cache_key (str): Deterministic cache key.

    Returns:
        Path: Vectors cache path.

    """
    return cache_dir / f"{cache_key}.npy"


def _cache_metadata_path(*, cache_dir: Path, cache_key: str) -> Path:
    """Build metadata cache path from key.

    Args:
        cache_dir (Path): Cache directory.
        cache_key (str): Deterministic cache key.

    Returns:
        Path: Metadata cache path.

    """
    return cache_dir / f"{cache_key}.json"
