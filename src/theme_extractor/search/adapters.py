"""Adapters implementing the thin SearchBackend interface."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

from theme_extractor.errors import MissingOptionalDependencyError

_ELASTIC_MISSING_DEP_MSG = "Install the optional dependency group 'elasticsearch' to use this backend."
_OPENSEARCH_MISSING_DEP_MSG = "Install the optional dependency group 'opensearch' to use this backend."


@dataclass(frozen=True, slots=True)
class ElasticClientAdapter:
    """Thin adapter around the Elasticsearch Python client."""

    client: Any
    backend_name: str = "elasticsearch"

    @classmethod
    def from_connection(
        cls,
        *,
        url: str,
        timeout_s: float,
        verify_certs: bool,
    ) -> ElasticClientAdapter:
        """Build an adapter from connection settings.

        Args:
            url (str): Backend URL.
            timeout_s (float): Request timeout in seconds.
            verify_certs (bool): Whether TLS certificates are verified.

        Raises:
            MissingOptionalDependencyError: If `elasticsearch` is not installed.

        Returns:
            ElasticClientAdapter: Configured adapter.

        """
        try:
            module = import_module("elasticsearch")
        except ImportError as exc:
            raise MissingOptionalDependencyError(_ELASTIC_MISSING_DEP_MSG) from exc

        elasticsearch_class = module.Elasticsearch
        client = elasticsearch_class(
            hosts=[url],
            request_timeout=timeout_s,
            verify_certs=verify_certs,
        )
        return cls(client=client)

    def _search(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        """Run a backend search operation.

        Args:
            index (str): Target index name.
            body (dict[str, Any]): Search body payload.

        Returns:
            dict[str, Any]: Raw backend response payload.

        """
        response = self.client.search(index=index, body=body)
        return dict(response)

    def search_documents(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        """Execute a standard search query.

        Args:
            index (str): Target index name.
            body (dict[str, Any]): Search body payload.

        Returns:
            dict[str, Any]: Backend response payload.

        """
        return self._search(index=index, body=body)

    def terms_aggregation(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        """Execute a terms aggregation query.

        Args:
            index (str): Target index name.
            body (dict[str, Any]): Aggregation body payload.

        Returns:
            dict[str, Any]: Backend response payload.

        """
        return self._search(index=index, body=body)

    def significant_terms_aggregation(
        self,
        *,
        index: str,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a significant_terms aggregation query.

        Args:
            index (str): Target index name.
            body (dict[str, Any]): Aggregation body payload.

        Returns:
            dict[str, Any]: Backend response payload.

        """
        return self._search(index=index, body=body)

    def significant_text_aggregation(
        self,
        *,
        index: str,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a significant_text aggregation query.

        Args:
            index (str): Target index name.
            body (dict[str, Any]): Aggregation body payload.

        Returns:
            dict[str, Any]: Backend response payload.

        """
        return self._search(index=index, body=body)


@dataclass(frozen=True, slots=True)
class OpenSearchClientAdapter:
    """Thin adapter around the OpenSearch Python client."""

    client: Any
    backend_name: str = "opensearch"

    @classmethod
    def from_connection(
        cls,
        *,
        url: str,
        timeout_s: float,
        verify_certs: bool,
    ) -> OpenSearchClientAdapter:
        """Build an adapter from connection settings.

        Args:
            url (str): Backend URL.
            timeout_s (float): Request timeout in seconds.
            verify_certs (bool): Whether TLS certificates are verified.

        Raises:
            MissingOptionalDependencyError: If `opensearch-py` is not installed.

        Returns:
            OpenSearchClientAdapter: Configured adapter.

        """
        try:
            module = import_module("opensearchpy")
        except ImportError as exc:
            raise MissingOptionalDependencyError(_OPENSEARCH_MISSING_DEP_MSG) from exc

        opensearch_class = module.OpenSearch
        client = opensearch_class(
            hosts=[url],
            timeout=timeout_s,
            use_ssl=url.startswith("https://"),
            verify_certs=verify_certs,
        )
        return cls(client=client)

    def _search(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        """Run a backend search operation.

        Args:
            index (str): Target index name.
            body (dict[str, Any]): Search body payload.

        Returns:
            dict[str, Any]: Raw backend response payload.

        """
        response = self.client.search(index=index, body=body)
        return dict(response)

    def search_documents(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        """Execute a standard search query.

        Args:
            index (str): Target index name.
            body (dict[str, Any]): Search body payload.

        Returns:
            dict[str, Any]: Backend response payload.

        """
        return self._search(index=index, body=body)

    def terms_aggregation(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        """Execute a terms aggregation query.

        Args:
            index (str): Target index name.
            body (dict[str, Any]): Aggregation body payload.

        Returns:
            dict[str, Any]: Backend response payload.

        """
        return self._search(index=index, body=body)

    def significant_terms_aggregation(
        self,
        *,
        index: str,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a significant_terms aggregation query.

        Args:
            index (str): Target index name.
            body (dict[str, Any]): Aggregation body payload.

        Returns:
            dict[str, Any]: Backend response payload.

        """
        return self._search(index=index, body=body)

    def significant_text_aggregation(
        self,
        *,
        index: str,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a significant_text aggregation query.

        Args:
            index (str): Target index name.
            body (dict[str, Any]): Aggregation body payload.

        Returns:
            dict[str, Any]: Backend response payload.

        """
        return self._search(index=index, body=body)
