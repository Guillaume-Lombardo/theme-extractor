from __future__ import annotations

import numpy as np

from theme_extractor.extraction.embedding_cache import (
    build_embedding_cache_key,
    load_embeddings_from_cache,
    store_embeddings_in_cache,
)


def test_build_embedding_cache_key_is_deterministic() -> None:
    first = build_embedding_cache_key(
        strategy="bertopic",
        model_name="bge-m3",
        cache_version="v1",
        documents=["alpha", "beta"],
    )
    second = build_embedding_cache_key(
        strategy="bertopic",
        model_name="bge-m3",
        cache_version="v1",
        documents=["alpha", "beta"],
    )
    different_version = build_embedding_cache_key(
        strategy="bertopic",
        model_name="bge-m3",
        cache_version="v2",
        documents=["alpha", "beta"],
    )
    assert first == second
    assert first != different_version


def test_store_and_load_embeddings_roundtrip(tmp_path) -> None:
    vectors = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    cache_key = "unit-test-key"
    cache_dir = tmp_path / "cache"

    store_embeddings_in_cache(
        cache_dir=cache_dir,
        cache_key=cache_key,
        vectors=vectors,
        metadata={"strategy": "bertopic", "cache_version": "v1"},
    )
    loaded = load_embeddings_from_cache(cache_dir=cache_dir, cache_key=cache_key)
    assert loaded is not None
    assert np.array_equal(loaded, vectors)


def test_load_embeddings_from_cache_missing_returns_none(tmp_path) -> None:
    loaded = load_embeddings_from_cache(cache_dir=tmp_path / "cache", cache_key="missing")
    assert loaded is None
