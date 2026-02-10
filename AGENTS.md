# AGENTS.md

## Mission
Build a robust Python package to benchmark multiple theme/topic extraction strategies on the same corpus, with a coherent CLI, unified outputs, and strong ingestion/cleaning quality.

## Current Stage
Stage 1 focuses on AI delivery tooling:
- agent governance (`agent.md`)
- execution roadmap (`plan.md`)
- reusable project skills (`skills/*`)

## Working Rules
- Use English as the default language for docstrings, README, and core project artifacts.
- Allow French only as a secondary translation or complementary version when needed.
- Keep architecture modular and strategy-driven.
- Keep implementation offline-first.
- Proxy support must be explicit and configurable.
- Do not couple business logic to a specific backend client.
- Prefer typed enums for user-facing choices:
  - use `enum.StrEnum` for single-choice values
  - use `enum.Flag`/`enum.IntFlag` for combinable choices
  - provide explicit conversions from `str` to enum/flag and back
- Write Google-style docstrings with explicit types in `Args` and `Returns` (and `Raises` when relevant).

## Mandatory Clarification Method (MBAD)
Use MBAD before major design or implementation decisions:
- `M` Mission: what business/functional outcome is targeted?
- `B` Boundaries: scope, constraints, and non-goals.
- `A` Alternatives: candidate options with tradeoffs.
- `D` Decisions: final choice, rationale, and open questions.

## Quality Gates
- Unit tests are the default run target.
- Before closing any PR, run all tests from `tests/unit`, `tests/integration`, and `tests/end2end`.
- Test markers are auto-applied by `tests/conftest.py`.
- Require one end-to-end test for every main feature.
- Require integration tests for all supported feature combinations.
- When a bug is reported, write a failing test first, then implement the fix.

## Delivery Workflow
- Implement each run, phase, and feature in a dedicated branch created for that specific scope.
- Do not develop features directly on the main branch.
- End every run, phase, and feature delivery with a GitHub Pull Request.
- Use PR review and CI as mandatory validation before merge.

## Pre-PR Checklist
Run locally:
- `uv run ruff format .`
- `uv run ruff check .`
- `uv run ty check src tests`
- `uv run pytest -m unit`
- `uv run pytest -m integration`
- `uv run pytest -m end2end`
- `uv run pre-commit run --all-files`

## Skills
Project skills live in `skills/`:
- `skills/architecture/SKILL.md`
- `skills/testing/SKILL.md`
- `skills/code-style/SKILL.md`
- `skills/tooling/SKILL.md`

## Architecture Constraint (Search Backend)
Elasticsearch and OpenSearch access must go through a thin interface:
- one shared protocol for required operations
- one adapter for Elasticsearch Python client
- one adapter for OpenSearch Python client
