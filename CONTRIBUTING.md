# Contributing to theme-extractor

Thanks for contributing.

## Prerequisites
- Python 3.13+
- `uv`
- Optional: Docker (for local Elasticsearch/OpenSearch stacks)

## Local setup
```bash
uv sync --group dev
```

## Development workflow
1. Create a dedicated branch from `main`.
2. Implement the change with tests.
3. Keep docs/config in sync when behavior changes:
   - `README.md`
   - `.env.template`
4. Open a Pull Request.

## Required checks before PR
```bash
uv run ruff format .
uv run ruff check .
uv run ty check src tests
uv run pytest -m unit
uv run pytest -m integration --no-cov
uv run pytest -m end2end --no-cov
uv run pre-commit run --all-files
```

## Testing policy
- Add unit tests for new logic.
- Add integration tests for supported combinations.
- Add at least one end-to-end test for each main feature.
- For bug fixes, write a failing test first, then implement the fix.

## Commit and PR guidance
- Keep commits focused and atomic.
- Use clear PR titles and explain:
  - problem
  - root cause
  - fix
  - validation evidence

## Code style
- Use typed enums for user-facing choices (`StrEnum`, `Flag`/`IntFlag`).
- Keep docstrings in Google style with typed `Args`, `Returns`, and `Raises` where relevant.
- Prefer modular strategy-based architecture and backend abstractions.
