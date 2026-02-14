"""BERTopic-like extraction strategy with configurable matrix options."""

from __future__ import annotations

import math
from contextlib import suppress
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from pydantic import BaseModel, ConfigDict
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from theme_extractor.domain import (
    BertopicClustering,
    BertopicDimReduction,
    DocumentTopicLink,
    OutputFocus,
    TopicKeyword,
    TopicResult,
    UnifiedExtractionOutput,
)
from theme_extractor.extraction.baselines import BaselineExtractionConfig  # noqa: TC001
from theme_extractor.extraction.embedding_cache import (
    build_embedding_cache_key,
    load_embeddings_from_cache,
    store_embeddings_in_cache,
)
from theme_extractor.extraction.utils import resolve_embedding_model_name
from theme_extractor.ingestion.cleaning import get_default_stopwords

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from theme_extractor.search.protocols import SearchBackend

    type Matrix = NDArray | csr_matrix


_EMPTY_CORPUS_NOTE = "BERTopic executed with empty corpus from backend search."
_MIN_DOCS_FOR_CLUSTERING = 2
_SMALL_CORPUS_NOTE = "BERTopic requires at least 2 usable documents; output returned empty topics."
_UMAP_FALLBACK_NOTE = "UMAP unavailable or failed; dimensionality reduction fell back to none."
_HDBSCAN_FALLBACK_NOTE = "HDBSCAN dependency missing; clustering fell back to kmeans."
_EMBEDDINGS_FALLBACK_NOTE = "Embedding dependency missing; BERTopic fell back to TF-IDF vectors."
_EMBEDDINGS_RUNTIME_FALLBACK_NOTE = "Embedding runtime failed; BERTopic fell back to TF-IDF vectors."
_SEARCH_SIZE_LIMIT = 1000
_SEARCH_SIZE_LIMIT_NOTE = f"BERTopic search_size was capped to {_SEARCH_SIZE_LIMIT} to limit memory usage."
_DEFAULT_LOCAL_MODELS_DIR = Path("data/models")
_DEFAULT_EMBEDDING_CACHE_DIR = Path("data/cache/embeddings")
_NO_TOPICS_AFTER_FILTER_NOTE = "BERTopic produced no topics after min_topic_size/topic filtering."
_RELAXED_TOPIC_SIZE_NOTE = (
    "BERTopic relaxed min_topic_size to 1 because the configured threshold removed all clusters."
)
_EMBEDDING_CACHE_HIT_NOTE = "BERTopic embeddings loaded from local cache."
_KEYWORD_STOPWORDS = get_default_stopwords()


class BertopicExtractionConfig(BaseModel):
    """Represent BERTopic extraction runtime options.

    Args:
        use_embeddings (bool): Whether to compute embedding vectors.
        embedding_model (str): Embedding model name.
        reduce_dim (BertopicDimReduction): Dimensionality reduction method.
        clustering (BertopicClustering): Clustering method.
        nr_topics (int | None): Fixed number of topics for KMeans.
        min_topic_size (int): Minimum accepted topic size.
        seed (int): Random seed.
        local_models_dir (Path | None): Local directory used to resolve embedding model aliases.
        embedding_cache_enabled (bool): Enable local embedding cache.
        embedding_cache_dir (Path): Local cache directory for embeddings.
        embedding_cache_version (str): Cache namespace version.

    """

    model_config = ConfigDict(frozen=True)

    use_embeddings: bool = False
    embedding_model: str = "bge-m3"
    reduce_dim: BertopicDimReduction = BertopicDimReduction.SVD
    clustering: BertopicClustering = BertopicClustering.KMEANS
    nr_topics: int | None = None
    min_topic_size: int = 10
    seed: int = 42
    local_models_dir: Path | None = _DEFAULT_LOCAL_MODELS_DIR
    embedding_cache_enabled: bool = True
    embedding_cache_dir: Path = _DEFAULT_EMBEDDING_CACHE_DIR
    embedding_cache_version: str = "v1"


class BertopicRunRequest(BaseModel):
    """Represent one BERTopic execution request.

    Args:
        index (str): Target backend index.
        focus (OutputFocus): Output focus mode.
        baseline_config (BaselineExtractionConfig): Backend corpus retrieval options.
        bertopic_config (BertopicExtractionConfig): BERTopic strategy options.

    """

    model_config = ConfigDict(frozen=True)

    index: str
    focus: OutputFocus
    baseline_config: BaselineExtractionConfig
    bertopic_config: BertopicExtractionConfig


def _normalized_query(query: str, fields: Sequence[str]) -> dict[str, Any]:
    normalized = query.strip()
    if normalized in {"", "*", "match_all"}:
        return {"match_all": {}}
    return {
        "simple_query_string": {
            "query": normalized,
            "fields": list(fields),
            "default_operator": "and",
        },
    }


def _search_body(config: BaselineExtractionConfig) -> dict[str, Any]:
    effective_size = max(1, min(config.search_size, _SEARCH_SIZE_LIMIT))
    return {
        "size": effective_size,
        "query": _normalized_query(config.query, config.fields),
        "_source": {"includes": [config.source_field]},
    }


def _choose_k(n_docs: int) -> int:
    if n_docs <= 2:  # noqa: PLR2004
        return 2
    return max(2, min(20, int(math.sqrt(n_docs)) + 1))


def _build_tfidf_matrix(documents: list[str]) -> tuple[csr_matrix, NDArray]:
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=1, max_features=50_000)
    sparse_tfidf = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()
    return sparse_tfidf, feature_names


def _make_embeddings_if_enabled(  # noqa: PLR0913
    *,
    use_embeddings: bool,
    embedding_model: str,
    local_models_dir: Path | None,
    embedding_cache_enabled: bool,
    embedding_cache_dir: Path,
    embedding_cache_version: str,
    documents: list[str],
) -> tuple[NDArray | None, str | None]:
    if not use_embeddings:
        return None, None
    try:
        resolved_model = resolve_embedding_model_name(
            embedding_model=embedding_model,
            local_models_dir=local_models_dir,
        )
        if embedding_cache_enabled:
            cache_key = build_embedding_cache_key(
                strategy="bertopic",
                model_name=resolved_model,
                cache_version=embedding_cache_version,
                documents=documents,
            )
            try:
                cached_vectors = load_embeddings_from_cache(
                    cache_dir=embedding_cache_dir.expanduser(),
                    cache_key=cache_key,
                )
            except Exception:
                cached_vectors = None
            if cached_vectors is not None:
                return cached_vectors, _EMBEDDING_CACHE_HIT_NOTE

        sentence_transformers_module = import_module("sentence_transformers")
        sentence_transformer_cls = sentence_transformers_module.SentenceTransformer
        model = sentence_transformer_cls(resolved_model)
        vectors = model.encode(documents, convert_to_numpy=True, normalize_embeddings=True)

        if embedding_cache_enabled:
            with suppress(Exception):
                store_embeddings_in_cache(
                    cache_dir=embedding_cache_dir.expanduser(),
                    cache_key=cache_key,
                    vectors=np.asarray(vectors, dtype=np.float32),
                    metadata={
                        "strategy": "bertopic",
                        "model_name": resolved_model,
                        "cache_version": embedding_cache_version,
                        "document_count": len(documents),
                    },
                )
    except ImportError:
        return None, _EMBEDDINGS_FALLBACK_NOTE
    except Exception as exc:
        return None, f"{_EMBEDDINGS_RUNTIME_FALLBACK_NOTE} Root error: {exc}"
    return vectors, None


def _apply_reduction(
    *,
    matrix: Matrix,
    reduce_dim: BertopicDimReduction,
    seed: int,
) -> tuple[Matrix, str | None]:
    if reduce_dim == BertopicDimReduction.NONE:
        return matrix, None

    n_docs, n_features = matrix.shape
    n_components = min(100, max(2, n_features - 1), max(2, n_docs - 1))

    if reduce_dim == BertopicDimReduction.SVD:
        return _apply_svd_reduction(matrix=matrix, n_components=n_components, seed=seed)

    if reduce_dim == BertopicDimReduction.NMF:
        return _apply_nmf_reduction(matrix=matrix, n_components=n_components, seed=seed)

    if reduce_dim == BertopicDimReduction.UMAP:
        return _apply_umap_reduction(matrix=matrix, n_components=n_components, seed=seed)

    return matrix, None


def _apply_svd_reduction(
    *,
    matrix: Matrix,
    n_components: int,
    seed: int,
) -> tuple[Matrix, str | None]:
    try:
        reduced = TruncatedSVD(
            n_components=n_components,
            random_state=seed,
        ).fit_transform(matrix)
    except Exception:
        return matrix, "SVD reduction failed; dimensionality reduction fell back to none."
    return normalize(reduced), None


def _apply_nmf_reduction(
    *,
    matrix: Matrix,
    n_components: int,
    seed: int,
) -> tuple[Matrix, str | None]:
    try:
        reduced = NMF(
            n_components=n_components,
            random_state=seed,
            init="nndsvda",
        ).fit_transform(matrix)
    except Exception:
        return matrix, "NMF reduction failed; dimensionality reduction fell back to none."
    return normalize(reduced), None


def _apply_umap_reduction(
    *,
    matrix: Matrix,
    n_components: int,
    seed: int,
) -> tuple[Matrix, str | None]:
    try:
        umap_module = import_module("umap")
    except ImportError:
        return matrix, _UMAP_FALLBACK_NOTE

    umap_class = umap_module.UMAP
    reducer = umap_class(
        n_neighbors=15,
        n_components=min(20, max(2, n_components)),
        min_dist=0.0,
        metric="cosine",
        random_state=seed,
    )
    try:
        reduced = reducer.fit_transform(matrix)
    except Exception:
        return matrix, _UMAP_FALLBACK_NOTE
    return normalize(reduced), None


def _cluster_labels(
    *,
    matrix: NDArray,
    clustering: BertopicClustering,
    nr_topics: int | None,
    seed: int,
    n_docs: int,
) -> tuple[NDArray, str | None]:
    if clustering == BertopicClustering.KMEANS:
        n_clusters = nr_topics if nr_topics is not None else _choose_k(n_docs)
        n_clusters = max(2, min(n_clusters, n_docs))
        model = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
        labels = model.fit_predict(matrix)
        return labels, None

    if clustering == BertopicClustering.HDBSCAN:
        try:
            hdbscan_module = import_module("hdbscan")
        except ImportError:
            n_clusters = nr_topics if nr_topics is not None else _choose_k(n_docs)
            n_clusters = max(2, min(n_clusters, n_docs))
            model = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
            labels = model.fit_predict(matrix)
            return labels, _HDBSCAN_FALLBACK_NOTE

        hdbscan_class = hdbscan_module.HDBSCAN
        model = hdbscan_class(min_cluster_size=2)
        labels = model.fit_predict(matrix)
        non_noise = labels[labels >= 0]
        if non_noise.size == 0:
            n_clusters = nr_topics if nr_topics is not None else _choose_k(n_docs)
            n_clusters = max(2, min(n_clusters, n_docs))
            kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
            labels = kmeans.fit_predict(matrix)
            return labels, _HDBSCAN_FALLBACK_NOTE
        return labels, None

    n_clusters = nr_topics if nr_topics is not None else _choose_k(n_docs)
    n_clusters = max(2, min(n_clusters, n_docs))
    model = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
    labels = model.fit_predict(matrix)
    return labels, None


def _to_document_topics(
    *,
    focus: OutputFocus,
    labels: NDArray,
    document_ids: list[str],
    topic_id_map: dict[int, int],
) -> list[DocumentTopicLink] | None:
    if focus not in {OutputFocus.DOCUMENTS, OutputFocus.BOTH}:
        return None
    doc_topics: list[DocumentTopicLink] = []
    for index, cluster_id in enumerate(labels):
        cluster_id_int = int(cluster_id)
        if cluster_id_int not in topic_id_map:
            continue
        doc_topics.append(
            DocumentTopicLink(
                document_id=document_ids[index],
                topic_id=topic_id_map[cluster_id_int],
                rank=index + 1,
            ),
        )
    return doc_topics


def _collect_documents_from_hits(
    *,
    hits: list[dict[str, Any]],
    source_field: str,
) -> tuple[list[str], list[str]]:
    documents: list[str] = []
    document_ids: list[str] = []
    for hit in hits:
        source = hit.get("_source") or {}
        value = source.get(source_field)
        if not isinstance(value, str) or not value.strip():
            continue
        documents.append(value)
        document_ids.append(str(hit.get("_id", f"doc-{len(document_ids)}")))
    return documents, document_ids


def _set_empty_output(
    *,
    output: UnifiedExtractionOutput,
    focus: OutputFocus,
    note: str,
) -> UnifiedExtractionOutput:
    output.notes.append(note)
    output.document_topics = [] if focus in {OutputFocus.DOCUMENTS, OutputFocus.BOTH} else None
    return output


def _build_topics_from_clusters(  # noqa: PLR0913
    *,
    labels: NDArray,
    sparse_tfidf: csr_matrix,
    feature_names: NDArray,
    request: BertopicRunRequest,
    document_ids: list[str],
    min_topic_size: int,
) -> tuple[list[TopicResult], dict[int, int]]:
    clusters = sorted({int(label) for label in labels if int(label) >= 0})
    topics: list[TopicResult] = []
    topic_id_map: dict[int, int] = {}
    next_topic_id = 0

    for cluster_id in clusters:
        doc_indexes = np.where(labels == cluster_id)[0]
        size = int(doc_indexes.size)
        if size < max(1, min_topic_size):
            continue
        mean_tfidf = np.asarray(sparse_tfidf[doc_indexes].mean(axis=0)).ravel()
        top_indexes = np.argsort(mean_tfidf)[::-1][: max(1, request.baseline_config.top_n)]
        keywords = [
            TopicKeyword(term=str(feature_names[i]), score=float(mean_tfidf[i]))
            for i in top_indexes
            if float(mean_tfidf[i]) > 0.0 and str(feature_names[i]).strip().lower() not in _KEYWORD_STOPWORDS
        ]
        if not keywords:
            continue

        topic_id_map[cluster_id] = next_topic_id
        topics.append(
            TopicResult(
                topic_id=next_topic_id,
                label=f"bertopic-{next_topic_id}",
                score=float(size),
                keywords=keywords,
                document_ids=[document_ids[i] for i in doc_indexes],
            ),
        )
        next_topic_id += 1

    return topics, topic_id_map


def _compute_clustering_inputs(
    *,
    request: BertopicRunRequest,
    documents: list[str],
    output: UnifiedExtractionOutput,
) -> tuple[csr_matrix, NDArray, NDArray]:
    sparse_tfidf, feature_names = _build_tfidf_matrix(documents)
    embedding_vectors, embedding_note = _make_embeddings_if_enabled(
        use_embeddings=request.bertopic_config.use_embeddings,
        embedding_model=request.bertopic_config.embedding_model,
        local_models_dir=request.bertopic_config.local_models_dir,
        embedding_cache_enabled=request.bertopic_config.embedding_cache_enabled,
        embedding_cache_dir=request.bertopic_config.embedding_cache_dir,
        embedding_cache_version=request.bertopic_config.embedding_cache_version,
        documents=documents,
    )
    if embedding_note:
        output.notes.append(embedding_note)

    base_matrix: Matrix
    base_matrix = embedding_vectors if embedding_vectors is not None else sparse_tfidf
    reduced_matrix, reduction_note = _apply_reduction(
        matrix=base_matrix,
        reduce_dim=request.bertopic_config.reduce_dim,
        seed=request.bertopic_config.seed,
    )
    if reduction_note:
        output.notes.append(reduction_note)

    clustering_matrix = _to_dense_matrix(reduced_matrix)

    labels, clustering_note = _cluster_labels(
        matrix=clustering_matrix,
        clustering=request.bertopic_config.clustering,
        nr_topics=request.bertopic_config.nr_topics,
        seed=request.bertopic_config.seed,
        n_docs=len(documents),
    )
    if clustering_note:
        output.notes.append(clustering_note)
    return sparse_tfidf, feature_names, labels


def _to_dense_matrix(matrix: Matrix) -> NDArray:
    if isinstance(matrix, csr_matrix):
        return cast("NDArray", matrix.toarray())
    return cast("NDArray", np.asarray(matrix))


def run_bertopic_method(
    *,
    backend: SearchBackend,
    request: BertopicRunRequest,
    output: UnifiedExtractionOutput,
) -> UnifiedExtractionOutput:
    """Run BERTopic-like extraction and populate unified output.

    Args:
        backend (SearchBackend): Search backend adapter.
        request (BertopicRunRequest): BERTopic runtime request.
        output (UnifiedExtractionOutput): Mutable output payload.

    Returns:
        UnifiedExtractionOutput: Updated output payload.

    """
    response = backend.search_documents(index=request.index, body=_search_body(request.baseline_config))
    hits = (response.get("hits") or {}).get("hits") or []
    documents, document_ids = _collect_documents_from_hits(
        hits=hits,
        source_field=request.baseline_config.source_field,
    )

    if request.baseline_config.search_size > _SEARCH_SIZE_LIMIT:
        output.notes.append(_SEARCH_SIZE_LIMIT_NOTE)

    if not documents:
        return _set_empty_output(output=output, focus=request.focus, note=_EMPTY_CORPUS_NOTE)
    if len(documents) < _MIN_DOCS_FOR_CLUSTERING:
        return _set_empty_output(output=output, focus=request.focus, note=_SMALL_CORPUS_NOTE)

    sparse_tfidf, feature_names, labels = _compute_clustering_inputs(
        request=request,
        documents=documents,
        output=output,
    )
    if sparse_tfidf.shape[1] < 2:  # noqa: PLR2004
        return _set_empty_output(output=output, focus=request.focus, note=_EMPTY_CORPUS_NOTE)

    topics, topic_id_map = _build_topics_from_clusters(
        labels=labels,
        sparse_tfidf=sparse_tfidf,
        feature_names=feature_names,
        request=request,
        document_ids=document_ids,
        min_topic_size=request.bertopic_config.min_topic_size,
    )
    if not topics and np.any(labels >= 0):
        topics, topic_id_map = _build_topics_from_clusters(
            labels=labels,
            sparse_tfidf=sparse_tfidf,
            feature_names=feature_names,
            request=request,
            document_ids=document_ids,
            min_topic_size=1,
        )
        if topics:
            output.notes.append(_RELAXED_TOPIC_SIZE_NOTE)

    if not topics:
        output.notes.append(_NO_TOPICS_AFTER_FILTER_NOTE)

    output.topics = topics
    output.document_topics = _to_document_topics(
        focus=request.focus,
        labels=labels,
        document_ids=document_ids,
        topic_id_map=topic_id_map,
    )
    output.notes.append("BERTopic strategy executed.")
    return output
