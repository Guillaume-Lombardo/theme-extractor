"""Sphinx configuration for theme-extractor documentation."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

_DOCS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _DOCS_DIR.parent
_SRC_DIR = _REPO_ROOT / "src"
_GENERATED_DIR = _DOCS_DIR / "_generated"

sys.path.insert(0, str(_SRC_DIR))

project = "theme-extractor"
author = "theme-extractor contributors"
release = "0.1.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]

autosummary_generate = True
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "alabaster"
html_static_path = ["_static"]


def _copy_external_markdown() -> None:
    """Copy repository README/howto files into Sphinx source tree."""
    _GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_REPO_ROOT / "README.md", _GENERATED_DIR / "README.md")

    howto_src_dir = _REPO_ROOT / "howto"
    for source_file in sorted(howto_src_dir.glob("*.md")):
        shutil.copy2(source_file, _GENERATED_DIR / source_file.name)


_copy_external_markdown()
