from __future__ import annotations

from dataclasses import dataclass
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
from theme_extractor.extraction import llm as llm_mod
from theme_extractor.extraction.llm import (
    LlmExtractionConfig,
    LlmRunRequest,
    run_llm_method,
)


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
        method=ExtractMethod.LLM,
        offline_policy=OfflinePolicy.STRICT,
        backend=BackendName.ELASTICSEARCH,
        index="idx",
    )
    return UnifiedExtractionOutput(focus=focus, metadata=metadata)


def test_run_llm_strict_offline_uses_tfidf_fallback() -> None:
    backend = _BackendStub(
        search_response={
            "hits": {
                "hits": [
                    {"_id": "doc-1", "_source": {"content": "invoice payment tax"}},
                    {"_id": "doc-2", "_source": {"content": "tax declaration invoice"}},
                ],
            },
        },
    )
    output = run_llm_method(
        backend=backend,
        request=LlmRunRequest(
            index="idx",
            focus=OutputFocus.BOTH,
            offline_policy=OfflinePolicy.STRICT,
            baseline_config=BaselineExtractionConfig(top_n=3),
            llm_config=LlmExtractionConfig(),
        ),
        output=_make_output(focus=OutputFocus.BOTH),
    )
    assert output.topics
    assert output.document_topics
    assert "strict offline mode" in "\n".join(output.notes)


def test_run_llm_with_provider_success(monkeypatch) -> None:
    backend = _BackendStub(
        search_response={
            "hits": {
                "hits": [
                    {"_id": "doc-1", "_source": {"content": "copropriete syndic assemblee"}},
                    {"_id": "doc-2", "_source": {"content": "charges copropriete syndic"}},
                ],
            },
        },
    )
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    monkeypatch.setattr(
        "theme_extractor.extraction.llm._extract_keywords_with_openai",
        lambda **_kwargs: [("copropriete", 0.9), ("syndic", 0.8)],
    )

    output = run_llm_method(
        backend=backend,
        request=LlmRunRequest(
            index="idx",
            focus=OutputFocus.TOPICS,
            offline_policy=OfflinePolicy.PRELOAD_OR_FIRST_RUN,
            baseline_config=BaselineExtractionConfig(top_n=2),
            llm_config=LlmExtractionConfig(),
        ),
        output=_make_output(focus=OutputFocus.TOPICS),
    )
    assert output.topics
    assert output.document_topics is None
    assert "provider response" in "\n".join(output.notes)


def test_run_llm_missing_api_key_falls_back_to_tfidf(monkeypatch) -> None:
    backend = _BackendStub(
        search_response={
            "hits": {
                "hits": [
                    {"_id": "doc-1", "_source": {"content": "assurance sinistre habitation"}},
                    {"_id": "doc-2", "_source": {"content": "declaration sinistre expert"}},
                ],
            },
        },
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    output = run_llm_method(
        backend=backend,
        request=LlmRunRequest(
            index="idx",
            focus=OutputFocus.TOPICS,
            offline_policy=OfflinePolicy.PRELOAD_OR_FIRST_RUN,
            baseline_config=BaselineExtractionConfig(top_n=2),
            llm_config=LlmExtractionConfig(),
        ),
        output=_make_output(focus=OutputFocus.TOPICS),
    )
    assert output.topics
    assert "credentials were not provided" in "\n".join(output.notes)


def test_run_llm_empty_corpus_sets_empty_document_topics() -> None:
    backend = _BackendStub(search_response={"hits": {"hits": []}})
    output = run_llm_method(
        backend=backend,
        request=LlmRunRequest(
            index="idx",
            focus=OutputFocus.BOTH,
            offline_policy=OfflinePolicy.STRICT,
            baseline_config=BaselineExtractionConfig(),
            llm_config=LlmExtractionConfig(),
        ),
        output=_make_output(focus=OutputFocus.BOTH),
    )
    assert output.topics == []
    assert output.document_topics == []
    assert "empty corpus" in "\n".join(output.notes)


def test_parse_keyword_payload_supports_dict_and_list_shapes() -> None:
    dict_payload = '{"keywords":[{"term":"tax","score":0.9},{"term":"invoice","score":0.8}]}'
    list_payload = '["budget","macro"]'

    dict_out = llm_mod._parse_keyword_payload(payload_text=dict_payload, top_n=2)
    list_out = llm_mod._parse_keyword_payload(payload_text=list_payload, top_n=2)

    assert dict_out == [("tax", 0.9), ("invoice", 0.8)]
    assert list_out == [("budget", 0.0), ("macro", 0.0)]


def test_extract_keywords_with_llm_runtime_error_uses_fallback(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    monkeypatch.setattr(
        "theme_extractor.extraction.llm._extract_keywords_with_openai",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    request = LlmRunRequest(
        index="idx",
        focus=OutputFocus.TOPICS,
        offline_policy=OfflinePolicy.PRELOAD_OR_FIRST_RUN,
        baseline_config=BaselineExtractionConfig(top_n=2),
        llm_config=LlmExtractionConfig(),
    )

    keywords, note = llm_mod._extract_keywords_with_llm(
        corpus_text="invoice tax payment\ninvoice declaration",
        request=request,
    )
    assert keywords
    assert "runtime failed" in note


def test_extract_keywords_with_openai_parses_provider_response(monkeypatch) -> None:
    class _Message:
        content = '{"keywords":[{"term":"impot","score":0.91}]}'

    class _Choice:
        message = _Message()

    class _Completions:
        @staticmethod
        def create(**_kwargs) -> object:
            class _Response:
                def __init__(self) -> None:
                    self.choices = [_Choice()]

            return _Response()

    class _Chat:
        completions = _Completions()

    class _Client:
        def __init__(self, **_kwargs) -> None:
            self.chat = _Chat()

    class _OpenAiModule:
        OpenAI = _Client

    monkeypatch.setattr("theme_extractor.extraction.llm.import_module", lambda _name: _OpenAiModule())
    out = llm_mod._extract_keywords_with_openai(
        corpus_text="impot revenu fiscal",
        config=LlmExtractionConfig(),
        top_n=3,
        api_key="fake-key",
    )
    assert out == [("impot", 0.91)]
