"""Project-specific exceptions for theme-extractor."""


class ThemeExtractorError(Exception):
    """Base exception for the project."""


class MissingOptionalDependencyError(ImportError, ThemeExtractorError):
    """Raised when an optional dependency is not installed."""


class UnsupportedBackendError(ValueError, ThemeExtractorError):
    """Raised when the user asks for an unsupported backend."""

    def __init__(self, backend: str, supported: str) -> None:
        """Build exception payload for unsupported backend values."""
        super().__init__(f"Unsupported backend '{backend}'. Supported values: {supported}.")


class UnsupportedMethodError(ValueError, ThemeExtractorError):
    """Raised when the user asks for an unsupported extraction method."""

    def __init__(self, method: str) -> None:
        """Build exception payload for unsupported extraction methods."""
        super().__init__(f"Unsupported extraction method: {method}")


class MissingBackendUrlError(ValueError, ThemeExtractorError):
    """Raised when no backend URL is supplied and no client is injected."""

    def __init__(self) -> None:
        """Build exception payload for missing backend URLs."""
        super().__init__("Backend URL is required when no client instance is provided.")
