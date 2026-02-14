"""Unified CLI package exports."""

from theme_extractor.cli.argument_parser import build_parser
from theme_extractor.cli.entrypoint import main

__all__ = ["build_parser", "main"]
