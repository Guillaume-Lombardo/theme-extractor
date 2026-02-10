"""Protocols for search backends used by extraction baselines."""

from __future__ import annotations

from typing import Any, Protocol


class SearchBackend(Protocol):
    """Define a thin interface for Elasticsearch/OpenSearch operations."""

    backend_name: str

    def search_documents(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        """Execute a standard search query.

        Args:
            index (str): Target index name.
            body (dict[str, Any]): Search body payload.

        Returns:
            dict[str, Any]: Raw backend response.

        """

    def terms_aggregation(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        """Execute a terms aggregation query.

        Args:
            index (str): Target index name.
            body (dict[str, Any]): Aggregation body payload.

        Returns:
            dict[str, Any]: Raw backend response.

        """

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
            dict[str, Any]: Raw backend response.

        """

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
            dict[str, Any]: Raw backend response.

        """
