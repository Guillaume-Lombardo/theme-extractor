---
name: tooling
description: Standardize developer tooling and validation workflow in theme-extractor. Use when setting up the environment, running quality checks, preparing a PR, or troubleshooting local tooling drift.
---

# Tooling Skill

## Purpose
Standardize setup, checks, and local developer workflows.

## Setup
- Run `uv sync --group dev`.
- Add optional groups when needed:
  - `uv sync --group bert`
  - `uv sync --group elasticsearch`
  - `uv sync --group opensearch`

## Hooks
- Run `uv run pre-commit install`.
- Run `uv run pre-commit run --all-files`.

## Standard Validation Pipeline
1. Run `uv run ruff format .`.
2. Run `uv run ruff check .`.
3. Run `uv run ty check src tests`.
4. Run `uv run pytest -m unit`.
5. Run `uv run pytest -m integration`.
6. Run `uv run pytest -m end2end`.
7. Run one dead-code cleanup pass and remove obsolete/unused code before pushing.

## PR Monitoring Rule
When a PR has just been created or updated:
1. Wait 60 seconds before the first GitHub check.
2. Then poll every 60 seconds until BOTH are true:
   - CI status is available and passing,
   - Copilot review has arrived.
3. Once both are available, analyze Copilot comments and apply changes according to relevance (`valid`, `partially valid`, `not needed` with rationale).

## Offline and Proxy Rules
- Keep model and backend usage configurable for offline execution.
- Respect proxy variables when network is required:
  - `HTTP_PROXY`
  - `HTTPS_PROXY`
  - `ALL_PROXY`
- Expose proxy and offline toggles in config or CLI, never hardcode them.
