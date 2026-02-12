from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from theme_extractor.domain import (
    BackendName,
    BertopicClustering,
    BertopicDimReduction,
    CommandName,
    ExtractionRunMetadata,
    ExtractMethod,
    OfflinePolicy,
    OutputFocus,
    UnifiedExtractionOutput,
)
from theme_extractor.extraction import BaselineExtractionConfig
from theme_extractor.extraction import bertopic as bertopic_mod
from theme_extractor.extraction.bertopic import (
    BertopicExtractionConfig,
    BertopicRunRequest,
    run_bertopic_method,
)

_EXPECTED_BERTOPIC_SEARCH_SIZE_CAP = 1000
_EXPECTED_BERTOPIC_CAP_NOTE = "BERTopic search_size was capped to 1000 to limit memory usage."


@dataclass
class _BackendStub:
    search_response: dict[str, Any]
    backend_name: str = "stub"
    last_body: dict[str, Any] | None = None

    def search_documents(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        _ = index
        self.last_body = body
        return self.search_response

    def terms_aggregation(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:  # noqa: PLR6301
        _ = index
        _ = body
        return {}

    def significant_terms_aggregation(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:  # noqa: PLR6301
        _ = index
        _ = body
        return {}

    def significant_text_aggregation(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:  # noqa: PLR6301
        _ = index
        _ = body
        return {}


def _make_output(*, focus: OutputFocus) -> UnifiedExtractionOutput:
    metadata = ExtractionRunMetadata(
        command=CommandName.EXTRACT,
        method=ExtractMethod.BERTOPIC,
        offline_policy=OfflinePolicy.STRICT,
        backend=BackendName.ELASTICSEARCH,
        index="idx",
    )
    return UnifiedExtractionOutput(focus=focus, metadata=metadata)


def test_run_bertopic_builds_topics_and_doc_links() -> None:
    backend = _BackendStub(
        search_response={
            "hits": {
                "hits": [
                    {"_id": "doc-1", "_source": {"content": "invoice tax payment declaration"}},
                    {"_id": "doc-2", "_source": {"content": "invoice due date payment tax"}},
                    {"_id": "doc-3", "_source": {"content": "copropriete reglement assemblée syndic"}},
                    {"_id": "doc-4", "_source": {"content": "syndic copropriete charges assemblée"}},
                ],
            },
        },
    )

    output = run_bertopic_method(
        backend=backend,
        request=BertopicRunRequest(
            index="idx",
            focus=OutputFocus.BOTH,
            baseline_config=BaselineExtractionConfig(top_n=3, search_size=20),
            bertopic_config=BertopicExtractionConfig(
                reduce_dim=BertopicDimReduction.SVD,
                clustering=BertopicClustering.KMEANS,
                min_topic_size=1,
                nr_topics=2,
            ),
        ),
        output=_make_output(focus=OutputFocus.BOTH),
    )

    assert output.topics
    assert output.document_topics
    assert "BERTopic strategy executed." in output.notes


def test_run_bertopic_empty_corpus_sets_empty_doc_topics() -> None:
    backend = _BackendStub(search_response={"hits": {"hits": []}})
    output = run_bertopic_method(
        backend=backend,
        request=BertopicRunRequest(
            index="idx",
            focus=OutputFocus.BOTH,
            baseline_config=BaselineExtractionConfig(),
            bertopic_config=BertopicExtractionConfig(),
        ),
        output=_make_output(focus=OutputFocus.BOTH),
    )
    assert output.topics == []
    assert output.document_topics == []
    assert "BERTopic executed with empty corpus from backend search." in output.notes


def test_run_bertopic_single_document_returns_empty_output() -> None:
    backend = _BackendStub(
        search_response={
            "hits": {
                "hits": [
                    {"_id": "doc-1", "_source": {"content": "single document only"}},
                ],
            },
        },
    )

    output = run_bertopic_method(
        backend=backend,
        request=BertopicRunRequest(
            index="idx",
            focus=OutputFocus.BOTH,
            baseline_config=BaselineExtractionConfig(),
            bertopic_config=BertopicExtractionConfig(),
        ),
        output=_make_output(focus=OutputFocus.BOTH),
    )

    assert output.topics == []
    assert output.document_topics == []
    assert "BERTopic requires at least 2 usable documents; output returned empty topics." in output.notes


def test_run_bertopic_caps_search_size_for_memory() -> None:
    backend = _BackendStub(
        search_response={
            "hits": {
                "hits": [
                    {"_id": "doc-1", "_source": {"content": "invoice tax payment declaration"}},
                    {"_id": "doc-2", "_source": {"content": "invoice due date payment tax"}},
                ],
            },
        },
    )

    output = run_bertopic_method(
        backend=backend,
        request=BertopicRunRequest(
            index="idx",
            focus=OutputFocus.TOPICS,
            baseline_config=BaselineExtractionConfig(search_size=10_000),
            bertopic_config=BertopicExtractionConfig(min_topic_size=1, nr_topics=2),
        ),
        output=_make_output(focus=OutputFocus.TOPICS),
    )

    assert backend.last_body is not None
    assert backend.last_body["size"] == _EXPECTED_BERTOPIC_SEARCH_SIZE_CAP
    assert _EXPECTED_BERTOPIC_CAP_NOTE in output.notes


def test_run_bertopic_adds_note_when_topics_filtered_out() -> None:
    backend = _BackendStub(
        search_response={
            "hits": {
                "hits": [
                    {"_id": "doc-1", "_source": {"content": "invoice tax payment declaration"}},
                    {"_id": "doc-2", "_source": {"content": "invoice due date payment tax"}},
                    {"_id": "doc-3", "_source": {"content": "copropriete reglement assemblee syndic"}},
                    {"_id": "doc-4", "_source": {"content": "syndic copropriete charges assemblee"}},
                ],
            },
        },
    )
    output = run_bertopic_method(
        backend=backend,
        request=BertopicRunRequest(
            index="idx",
            focus=OutputFocus.BOTH,
            baseline_config=BaselineExtractionConfig(top_n=3, search_size=20),
            bertopic_config=BertopicExtractionConfig(
                reduce_dim=BertopicDimReduction.SVD,
                clustering=BertopicClustering.KMEANS,
                min_topic_size=10,
                nr_topics=2,
            ),
        ),
        output=_make_output(focus=OutputFocus.BOTH),
    )
    assert output.topics == []
    assert "BERTopic produced no topics after min_topic_size/topic filtering." in output.notes


def test_make_embeddings_uses_local_model_alias_from_data_models(tmp_path, monkeypatch) -> None:
    model_dir = tmp_path / "data" / "models" / "bge-m3"
    model_dir.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

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

    class _Module:
        SentenceTransformer = _SentenceTransformer

    monkeypatch.setattr(bertopic_mod, "import_module", lambda _name: _Module())

    vectors, note = bertopic_mod._make_embeddings_if_enabled(
        use_embeddings=True,
        embedding_model="bge-m3",
        local_models_dir=Path("data/models"),
        documents=["doc one", "doc two"],
    )

    assert note is None
    assert vectors is not None
    assert captured["model_name"] == str(model_dir.resolve())
