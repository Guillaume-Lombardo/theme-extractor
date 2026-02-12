from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass
from typing import Any

import numpy as np

from theme_extractor.cli import main

_EXPECTED_METHOD_COUNT = 7


@dataclass
class _BackendStub:
    backend_name: str = "stub"

    def search_documents(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:  # noqa: PLR6301
        _ = index
        _ = body
        return {
            "hits": {
                "hits": [
                    {"_id": "doc-a", "_source": {"content": "alpha beta"}},
                    {"_id": "doc-b", "_source": {"content": "alpha gamma"}},
                ],
            },
        }

    def terms_aggregation(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:  # noqa: PLR6301
        _ = index
        _ = body
        return {
            "aggregations": {
                "terms": {
                    "buckets": [{"key": "alpha", "doc_count": 2}],
                },
            },
        }

    def significant_terms_aggregation(  # noqa: PLR6301
        self,
        *,
        index: str,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        _ = index
        _ = body
        return {
            "aggregations": {
                "themes": {
                    "buckets": [{"key": "beta", "doc_count": 1, "score": 4.2}],
                },
            },
        }

    def significant_text_aggregation(  # noqa: PLR6301
        self,
        *,
        index: str,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        _ = index
        _ = body
        return {
            "aggregations": {
                "themes": {
                    "buckets": [{"key": "gamma", "doc_count": 1, "score": 3.3}],
                },
            },
        }


@dataclass
class _EmptyBackendStub:
    backend_name: str = "stub-empty"

    def search_documents(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:  # noqa: PLR6301
        _ = index
        _ = body
        return {"hits": {"hits": []}}

    def terms_aggregation(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:  # noqa: PLR6301
        _ = index
        _ = body
        return {"aggregations": {"terms": {"buckets": []}}}

    def significant_terms_aggregation(  # noqa: PLR6301
        self,
        *,
        index: str,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        _ = index
        _ = body
        return {"aggregations": {"themes": {"buckets": []}}}

    def significant_text_aggregation(  # noqa: PLR6301
        self,
        *,
        index: str,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        _ = index
        _ = body
        return {"aggregations": {"themes": {"buckets": []}}}


def test_benchmark_supports_combined_methods_focus_and_backend(monkeypatch, capsys) -> None:
    monkeypatch.setattr("theme_extractor.cli.build_search_backend", lambda **_kwargs: _BackendStub())

    exit_code = main(
        [
            "benchmark",
            "--methods",
            "baseline_tfidf,terms,significant_terms,significant_text,keybert,bertopic,llm",
            "--focus",
            "topics",
            "--backend",
            "opensearch",
            "--index",
            "legal_docs",
            "--offline-policy",
            "preload_or_first_run",
        ],
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)

    assert len(payload["methods"]) == _EXPECTED_METHOD_COUNT
    assert payload["outputs"]["bertopic"]["metadata"]["backend"] == "opensearch"
    assert payload["outputs"]["llm"]["metadata"]["index"] == "legal_docs"
    assert payload["outputs"]["llm"]["metadata"]["offline_policy"] == "preload_or_first_run"


def test_extract_supports_strict_offline_policy_and_elasticsearch_backend(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr("theme_extractor.cli.build_search_backend", lambda **_kwargs: _BackendStub())

    exit_code = main(
        [
            "extract",
            "--method",
            "terms",
            "--focus",
            "both",
            "--backend",
            "elasticsearch",
            "--offline-policy",
            "strict",
            "--index",
            "theme_idx",
        ],
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["focus"] == "both"
    assert payload["document_topics"] == []
    assert payload["metadata"]["backend"] == "elasticsearch"
    assert payload["metadata"]["offline_policy"] == "strict"
    assert payload["metadata"]["index"] == "theme_idx"


def test_extract_bertopic_supports_matrix_flags_and_notes(monkeypatch, capsys) -> None:
    monkeypatch.setattr("theme_extractor.cli.build_search_backend", lambda **_kwargs: _BackendStub())
    monkeypatch.setattr(
        "theme_extractor.extraction.bertopic._make_embeddings_if_enabled",
        lambda **_kwargs: (None, "embedding fallback"),
    )
    monkeypatch.setattr(
        "theme_extractor.extraction.bertopic._apply_reduction",
        lambda **kwargs: (kwargs["matrix"], "reduction fallback"),
    )
    monkeypatch.setattr(
        "theme_extractor.extraction.bertopic._cluster_labels",
        lambda **_kwargs: (np.array([0, 0]), "cluster fallback"),
    )

    exit_code = main(
        [
            "extract",
            "--method",
            "bertopic",
            "--focus",
            "both",
            "--bertopic-use-embeddings",
            "--bertopic-dim-reduction",
            "umap",
            "--bertopic-clustering",
            "hdbscan",
            "--bertopic-min-topic-size",
            "1",
            "--bertopic-nr-topics",
            "2",
        ],
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["topics"]
    assert payload["document_topics"]
    assert "embedding fallback" in payload["notes"]
    assert "reduction fallback" in payload["notes"]
    assert "cluster fallback" in payload["notes"]


def test_extract_llm_strict_offline_uses_fallback(monkeypatch, capsys) -> None:
    monkeypatch.setattr("theme_extractor.cli.build_search_backend", lambda **_kwargs: _BackendStub())

    exit_code = main(
        [
            "extract",
            "--method",
            "llm",
            "--focus",
            "both",
            "--offline-policy",
            "strict",
            "--topn",
            "3",
        ],
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["topics"]
    assert payload["document_topics"]
    assert "strict offline mode" in "\n".join(payload["notes"])


def test_extract_llm_preload_mode_provider_path(monkeypatch, capsys) -> None:
    monkeypatch.setattr("theme_extractor.cli.build_search_backend", lambda **_kwargs: _BackendStub())
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    monkeypatch.setattr(
        "theme_extractor.extraction.llm._extract_keywords_with_openai",
        lambda **_kwargs: [("alpha", 0.91), ("beta", 0.87)],
    )

    exit_code = main(
        [
            "extract",
            "--method",
            "llm",
            "--focus",
            "topics",
            "--offline-policy",
            "preload_or_first_run",
            "--llm-provider",
            "openai",
            "--llm-model",
            "gpt-4o-mini",
            "--topn",
            "2",
        ],
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["topics"]
    assert payload["document_topics"] is None
    assert "provider response" in "\n".join(payload["notes"])


def test_extract_llm_preload_mode_runtime_failure_falls_back(monkeypatch, capsys) -> None:
    monkeypatch.setattr("theme_extractor.cli.build_search_backend", lambda **_kwargs: _BackendStub())
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    monkeypatch.setattr(
        "theme_extractor.extraction.llm._extract_keywords_with_openai",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    exit_code = main(
        [
            "extract",
            "--method",
            "llm",
            "--focus",
            "topics",
            "--offline-policy",
            "preload_or_first_run",
            "--query",
            "facture",
        ],
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["topics"]
    assert "runtime failed" in "\n".join(payload["notes"])


def test_extract_llm_empty_corpus_with_topic_focus(monkeypatch, capsys) -> None:
    monkeypatch.setattr("theme_extractor.cli.build_search_backend", lambda **_kwargs: _EmptyBackendStub())

    exit_code = main(
        [
            "extract",
            "--method",
            "llm",
            "--focus",
            "topics",
        ],
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["topics"] == []
    assert payload["document_topics"] is None
    assert "empty corpus" in "\n".join(payload["notes"])


def test_extract_bertopic_local_models_dir_option(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.setattr("theme_extractor.cli.build_search_backend", lambda **_kwargs: _BackendStub())
    local_models_dir = tmp_path / "models"
    (local_models_dir / "bge-m3").mkdir(parents=True)

    captured: dict[str, str] = {}

    class _SentenceTransformer:
        def __init__(self, model_name: str) -> None:
            captured["model_name"] = model_name

        @staticmethod
        def encode(
            _documents: list[str],
            *,
            convert_to_numpy: bool,
            normalize_embeddings: bool,
        ) -> np.ndarray:
            _ = convert_to_numpy
            _ = normalize_embeddings
            return np.array([[0.1, 0.2], [0.2, 0.3]])

    class _SentenceTransformersModule:
        SentenceTransformer = _SentenceTransformer

    monkeypatch.setattr(
        "theme_extractor.extraction.bertopic.import_module",
        lambda _name: _SentenceTransformersModule(),
    )

    exit_code = main(
        [
            "extract",
            "--method",
            "bertopic",
            "--focus",
            "topics",
            "--bertopic-use-embeddings",
            "--bertopic-embedding-model",
            "bge-m3",
            "--bertopic-local-models-dir",
            str(local_models_dir),
            "--bertopic-min-topic-size",
            "1",
            "--bertopic-nr-topics",
            "2",
        ],
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["topics"]
    assert captured["model_name"] == str((local_models_dir / "bge-m3").resolve())


def test_extract_keybert_local_models_dir_option(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.setattr("theme_extractor.cli.build_search_backend", lambda **_kwargs: _BackendStub())
    local_models_dir = tmp_path / "models"
    (local_models_dir / "bge-m3").mkdir(parents=True)
    captured: dict[str, str] = {}

    class _SentenceTransformer:
        def __init__(self, model_name: str) -> None:
            captured["model_name"] = model_name

    class _KeyBERT:
        def __init__(self, model: object | None = None) -> None:
            _ = model

        @staticmethod
        def extract_keywords(
            _corpus_text: str,
            *,
            keyphrase_ngram_range: tuple[int, int],
            stop_words: str,
            top_n: int,
        ) -> list[tuple[str, float]]:
            _ = keyphrase_ngram_range
            _ = stop_words
            _ = top_n
            return [("alpha", 0.9), ("beta", 0.8)]

    keybert_module = types.ModuleType("keybert")
    keybert_module.KeyBERT = _KeyBERT  # type: ignore[attr-defined]
    sentence_transformers_module = types.ModuleType("sentence_transformers")
    sentence_transformers_module.SentenceTransformer = _SentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "keybert", keybert_module)
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers_module)

    exit_code = main(
        [
            "extract",
            "--method",
            "keybert",
            "--focus",
            "topics",
            "--keybert-use-embeddings",
            "--keybert-embedding-model",
            "bge-m3",
            "--keybert-local-models-dir",
            str(local_models_dir),
            "--topn",
            "2",
        ],
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["topics"]
    assert captured["model_name"] == str((local_models_dir / "bge-m3").resolve())
