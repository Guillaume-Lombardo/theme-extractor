---
name: code-style
description: Apply Python coding standards, typing discipline, and readability rules in theme-extractor. Use when writing or reviewing production code, especially around interfaces, CLI, and domain models.
---

# Code Style Skill

## Purpose
Keep code consistent, typed, and easy to maintain.

## Standards
- Use Python 3.13+ idioms.
- Type public interfaces explicitly.
- Keep functions small and focused.
- Use clear domain-oriented names.
- Add comments only when logic is non-obvious.

## Conventions
- Separate models/config/protocols from runtime services.
- Use `pydantic` for input and config validation.
- Raise explicit domain exceptions.
- Keep CLI parsing separate from business logic.
- Keep output schemas explicit and versionable.
- Prefer `enum.StrEnum` for single-choice options.
- Prefer `enum.Flag`/`enum.IntFlag` for combinable options, with conversion helpers (`str -> flag`, `flag -> str`).
- Use Google-style docstrings with explicit argument and return types to keep Sphinx generation reliable.

## Static Checks
- Formatting: Ruff formatter.
- Linting: Ruff.
- Type checking: Ty.

## Commands
- `uv run ruff format .`
- `uv run ruff check .`
- `uv run ty check src tests`
