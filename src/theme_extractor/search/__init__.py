"""Search backend interfaces and adapters."""

from theme_extractor.search.adapters import ElasticClientAdapter, OpenSearchClientAdapter
from theme_extractor.search.factory import build_search_backend, supported_backends
from theme_extractor.search.protocols import SearchBackend

__all__ = [
    "ElasticClientAdapter",
    "OpenSearchClientAdapter",
    "SearchBackend",
    "build_search_backend",
    "supported_backends",
]
