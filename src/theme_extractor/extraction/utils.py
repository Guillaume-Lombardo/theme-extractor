"""Shared extraction helpers."""

from __future__ import annotations

from pathlib import Path


def resolve_embedding_model_name(
    *,
    embedding_model: str,
    local_models_dir: Path | None,
) -> str:
    """Resolve an embedding model argument to a local path when possible.

    Args:
        embedding_model (str): Raw user-provided model value.
        local_models_dir (Path | None): Optional local model directory.

    Returns:
        str: Resolved model identifier or absolute local path.

    """
    normalized = embedding_model.strip()
    if not normalized:
        return normalized

    direct_path = Path(normalized).expanduser()
    if direct_path.exists():
        return str(direct_path.resolve())

    if local_models_dir is not None:
        local_candidate = (local_models_dir / normalized).expanduser()
        if local_candidate.exists():
            return str(local_candidate.resolve())

    return normalized
