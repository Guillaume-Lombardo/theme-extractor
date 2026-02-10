---
name: testing
description: Build and maintain a complete test strategy for theme-extractor across unit, integration, and end2end scopes. Use when implementing, refactoring, or validating CLI, ingestion, and extraction behavior.
---

# Testing Skill

## Purpose
Guarantee correctness and regression safety across all test scopes.

## Test Topology
- `tests/unit`: default fast scope.
- `tests/integration`: component and backend interactions.
- `tests/end2end`: full pipeline and CLI journeys.

Markers are auto-assigned by directory in `tests/conftest.py`.

## Workflow
1. Write or update unit tests first.
2. Add integration tests for boundaries and adapters.
3. Add end2end tests for user-visible workflows.
4. Run unit tests during iteration.
5. Run integration and end2end suites before PR completion.

## Commands
- `uv run pytest -m unit`
- `uv run pytest -m integration`
- `uv run pytest -m end2end`

## Quality Rules
- Make tests deterministic with explicit seeds and stable fixtures.
- Keep unit tests free of hidden external dependencies.
- Add proxy/offline scenarios when relevant.
- Add representative fixtures for each supported document type.
