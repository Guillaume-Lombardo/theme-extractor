from __future__ import annotations

from docker import index_corpus

_EXPECTED_REMOVED_STOPWORDS = 2


def test_parse_manual_stopwords_normalizes_accents() -> None:
    parsed = index_corpus._parse_manual_stopwords("été, la ,THE")
    assert parsed == {"ete", "la", "the"}


def test_build_stopwords_merges_all_sources(monkeypatch) -> None:
    monkeypatch.setattr(index_corpus, "discover_auto_stopwords", lambda *_args, **_kwargs: {"auto_term"})
    monkeypatch.setattr(index_corpus, "load_stopwords_from_files", lambda _paths: {"file_term"})
    monkeypatch.setattr(index_corpus, "get_default_stopwords", lambda: {"default_term"})

    merged = index_corpus._build_stopwords(
        tokenized_documents=[["alpha", "beta"], ["alpha", "gamma"]],
        manual_stopwords={"manual_term"},
        manual_stopwords_files=[],
        default_stopwords_enabled=True,
        auto_stopwords_enabled=True,
        auto_stopwords_min_doc_ratio=0.7,
        auto_stopwords_min_corpus_ratio=0.01,
        auto_stopwords_max_terms=200,
    )

    assert merged == {"auto_term", "default_term", "file_term", "manual_term"}


def test_mapping_body_contains_raw_and_clean_content_fields() -> None:
    mapping = index_corpus._mapping_body()
    properties = mapping["mappings"]["properties"]
    assert "content_raw" in properties
    assert "content_clean" in properties
    assert "content" in properties


def test_build_index_documents_filters_stopwords_from_index_fields() -> None:
    prepared_documents = [
        {
            "_id": "doc-1",
            "_source": {
                "path": "/work/doc.txt",
                "filename": "doc.txt",
                "extension": ".txt",
                "content_raw": "LA maison and roof",
                "content_clean": "la maison and roof",
                "tokens_all": ["la", "maison", "and", "roof"],
            },
        },
    ]

    docs = index_corpus._build_index_documents(
        prepared_documents=prepared_documents,
        stopwords={"la", "and"},
    )

    source = docs[0]["_source"]
    assert source["content"] == "maison roof"
    assert source["content_raw"] == "LA maison and roof"
    assert source["tokens"] == ["maison", "roof"]
    assert source["content_clean"] == "la maison and roof"
    assert source["tokens_all"] == ["la", "maison", "and", "roof"]
    assert source["removed_stopword_count"] == _EXPECTED_REMOVED_STOPWORDS
