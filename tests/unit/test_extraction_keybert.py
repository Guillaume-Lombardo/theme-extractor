from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from theme_extractor.domain import (
    BackendName,
    CommandName,
    ExtractionRunMetadata,
    ExtractMethod,
    OfflinePolicy,
    OutputFocus,
    UnifiedExtractionOutput,
)
from theme_extractor.extraction import BaselineExtractionConfig
from theme_extractor.extraction.keybert import (
    KeyBertExtractionConfig,
    KeyBertRunRequest,
    run_keybert_method,
)

_EXPECTED_DOC_TOPIC_COUNT = 2


@dataclass
class _BackendStub:
    search_response: dict[str, Any]
    backend_name: str = "stub"

    def search_documents(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        _ = index
        _ = body
        return self.search_response

    def terms_aggregation(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:  # noqa: PLR6301
        _ = index
        _ = body
        return {}

    def significant_terms_aggregation(  # noqa: PLR6301
        self,
        *,
        index: str,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        _ = index
        _ = body
        return {}

    def significant_text_aggregation(  # noqa: PLR6301
        self,
        *,
        index: str,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        _ = index
        _ = body
        return {}


def _make_output(*, focus: OutputFocus) -> UnifiedExtractionOutput:
    metadata = ExtractionRunMetadata(
        command=CommandName.EXTRACT,
        method=ExtractMethod.KEYBERT,
        offline_policy=OfflinePolicy.STRICT,
        backend=BackendName.ELASTICSEARCH,
        index="idx",
    )
    return UnifiedExtractionOutput(focus=focus, metadata=metadata)


def test_run_keybert_uses_keybert_keywords_when_available(monkeypatch) -> None:
    backend = _BackendStub(
        search_response={
            "hits": {
                "hits": [
                    {"_id": "doc-1", "_source": {"content": "french tax invoice"}},
                    {"_id": "doc-2", "_source": {"content": "invoice payment due date"}},
                ],
            },
        },
    )
    monkeypatch.setattr(
        "theme_extractor.extraction.keybert._extract_keywords_with_keybert",
        lambda **_kwargs: [("invoice", 0.9), ("tax", 0.8)],
    )

    output = run_keybert_method(
        backend=backend,
        request=KeyBertRunRequest(
            index="idx",
            focus=OutputFocus.BOTH,
            config=BaselineExtractionConfig(top_n=2),
            keybert_config=KeyBertExtractionConfig(),
        ),
        output=_make_output(focus=OutputFocus.BOTH),
    )

    assert output.topics
    assert output.topics[0].label == "keybert"
    assert [keyword.term for keyword in output.topics[0].keywords] == ["invoice", "tax"]
    assert output.document_topics is not None
    assert len(output.document_topics) == _EXPECTED_DOC_TOPIC_COUNT
    assert "KeyBERT strategy executed with keybert dependency." in output.notes


def test_run_keybert_falls_back_to_tfidf_when_keybert_missing(monkeypatch) -> None:
    backend = _BackendStub(
        search_response={
            "hits": {
                "hits": [
                    {"_id": "doc-1", "_source": {"content": "alpha beta gamma"}},
                ],
            },
        },
    )

    def _raise_import_error(**_kwargs):
        raise ImportError

    monkeypatch.setattr(
        "theme_extractor.extraction.keybert._extract_keywords_with_keybert",
        _raise_import_error,
    )

    output = run_keybert_method(
        backend=backend,
        request=KeyBertRunRequest(
            index="idx",
            focus=OutputFocus.TOPICS,
            config=BaselineExtractionConfig(top_n=2),
            keybert_config=KeyBertExtractionConfig(),
        ),
        output=_make_output(focus=OutputFocus.TOPICS),
    )

    assert output.topics
    assert output.topics[0].label == "keybert"
    assert output.topics[0].keywords
    assert output.document_topics is None
    assert "KeyBERT dependency missing; TF-IDF fallback was used." in output.notes


def test_run_keybert_empty_corpus_with_topics_focus() -> None:
    backend = _BackendStub(search_response={"hits": {"hits": []}})

    output = run_keybert_method(
        backend=backend,
        request=KeyBertRunRequest(
            index="idx",
            focus=OutputFocus.TOPICS,
            config=BaselineExtractionConfig(top_n=2),
            keybert_config=KeyBertExtractionConfig(),
        ),
        output=_make_output(focus=OutputFocus.TOPICS),
    )

    assert output.topics == []
    assert output.document_topics is None
    assert "KeyBERT executed with empty corpus from backend search." in output.notes


def test_run_keybert_empty_corpus_with_both_focus() -> None:
    backend = _BackendStub(search_response={"hits": {"hits": []}})

    output = run_keybert_method(
        backend=backend,
        request=KeyBertRunRequest(
            index="idx",
            focus=OutputFocus.BOTH,
            config=BaselineExtractionConfig(top_n=2),
            keybert_config=KeyBertExtractionConfig(),
        ),
        output=_make_output(focus=OutputFocus.BOTH),
    )

    assert output.topics == []
    assert output.document_topics == []
    assert "KeyBERT executed with empty corpus from backend search." in output.notes


def test_run_keybert_uses_local_model_alias(monkeypatch, tmp_path) -> None:
    backend = _BackendStub(
        search_response={
            "hits": {
                "hits": [
                    {"_id": "doc-1", "_source": {"content": "french tax invoice"}},
                    {"_id": "doc-2", "_source": {"content": "invoice payment due date"}},
                ],
            },
        },
    )
    local_models_dir = tmp_path / "models"
    alias_path = local_models_dir / "bge-m3"
    alias_path.mkdir(parents=True)
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
            return [("invoice", 0.9), ("tax", 0.8)]

    keybert_module = types.ModuleType("keybert")
    keybert_module.KeyBERT = _KeyBERT  # type: ignore[attr-defined]
    sentence_transformers_module = types.ModuleType("sentence_transformers")
    sentence_transformers_module.SentenceTransformer = _SentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "keybert", keybert_module)
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers_module)

    output = run_keybert_method(
        backend=backend,
        request=KeyBertRunRequest(
            index="idx",
            focus=OutputFocus.TOPICS,
            config=BaselineExtractionConfig(top_n=2),
            keybert_config=KeyBertExtractionConfig(
                use_embeddings=True,
                embedding_model="bge-m3",
                local_models_dir=Path(local_models_dir),
            ),
        ),
        output=_make_output(focus=OutputFocus.TOPICS),
    )

    assert output.topics
    assert captured["model_name"] == str(alias_path.resolve())
