"""Factory helpers to instantiate the configured search backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

from theme_extractor.domain import BackendName
from theme_extractor.errors import MissingBackendUrlError, UnsupportedBackendError
from theme_extractor.search.adapters import ElasticClientAdapter, OpenSearchClientAdapter

if TYPE_CHECKING:
    from theme_extractor.search.protocols import SearchBackend


def supported_backends() -> tuple[BackendName, BackendName]:
    """Return backend names supported by the project.

    Returns:
        tuple[BackendName, BackendName]: Supported backend identifiers.

    """
    return (BackendName.ELASTICSEARCH, BackendName.OPENSEARCH)


def build_search_backend(
    *,
    backend: BackendName | str,
    client: object | None = None,
    url: str | None = None,
    timeout_s: float = 30.0,
    verify_certs: bool = True,
) -> SearchBackend:
    """Build a concrete backend adapter from user options.

    Args:
        backend (BackendName | str): Backend identifier.
        client (object | None): Optional pre-configured backend client.
        url (str | None): Optional backend URL when no client is injected.
        timeout_s (float): Request timeout in seconds.
        verify_certs (bool): Whether TLS certificates are verified.

    Raises:
        MissingBackendUrlError: If `url` is missing when `client` is absent.
        UnsupportedBackendError: If the backend identifier is unsupported.

    Returns:
        SearchBackend: Search backend adapter.

    """
    backend_name = backend.value if isinstance(backend, BackendName) else backend.strip().lower()

    if backend_name == BackendName.ELASTICSEARCH.value:
        if client is not None:
            return ElasticClientAdapter(client=client)
        if url is None:
            raise MissingBackendUrlError
        return ElasticClientAdapter.from_connection(
            url=url,
            timeout_s=timeout_s,
            verify_certs=verify_certs,
        )

    if backend_name == BackendName.OPENSEARCH.value:
        if client is not None:
            return OpenSearchClientAdapter(client=client)
        if url is None:
            raise MissingBackendUrlError
        return OpenSearchClientAdapter.from_connection(
            url=url,
            timeout_s=timeout_s,
            verify_certs=verify_certs,
        )

    supported = ", ".join(item.value for item in supported_backends())
    raw_backend = backend.value if isinstance(backend, BackendName) else backend
    raise UnsupportedBackendError(backend=raw_backend, supported=supported)
