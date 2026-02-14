"""Shared CLI runtime primitives (environment, payload emission, proxy)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel

_PROXY_ENV_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")
_DEFAULT_BACKEND_URL = "http://localhost:9200"
_DEFAULT_INDEX = "theme_extractor"


def env_bool(name: str, *, default_value: bool) -> bool:
    """Read a boolean value from environment variables.

    Args:
        name (str): Environment variable name.
        default_value (bool): Fallback value when missing or invalid.

    Returns:
        bool: Parsed boolean value.

    """
    raw = os.getenv(name)
    if raw is None:
        return default_value
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default_value


def emit_payload(payload: dict[str, Any] | BaseModel | str, output: str) -> None:
    """Emit payload to stdout or to a file.

    Args:
        payload (dict[str, Any] | BaseModel | str): Data payload to serialize.
        output (str): Output path or "-" for stdout.

    """
    if isinstance(payload, str):
        serialized = payload
    else:
        payload_dict = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
        serialized = json.dumps(payload_dict, ensure_ascii=False, indent=2)

    if output == "-":
        print(serialized)
        return

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if serialized.endswith("\n"):
        output_path.write_text(serialized, encoding="utf-8")
    else:
        output_path.write_text(serialized + "\n", encoding="utf-8")


def apply_proxy_environment(proxy_url: str | None) -> None:
    """Apply proxy url to standard proxy environment variables.

    Args:
        proxy_url (str | None): Proxy URL when provided.

    """
    if not proxy_url:
        return

    for key in _PROXY_ENV_KEYS:
        os.environ[key] = proxy_url
