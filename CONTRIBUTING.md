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

## Quality standards (Definition of Done for contributions)

A change is considered acceptable if:

- It follows existing repository patterns (naming, structure, errors, types, config).
- It does not introduce dead code or unreachable branches.
- It does not introduce obvious duplication without justification.
- Public contracts (CLI/API/schemas/config) remain consistent and documented.
- New or changed behavior is covered by tests.
- Logs and errors are actionable and do not leak sensitive data.
- Any accepted technical debt is explicitly documented (issue or referenced TODO).

If a deviation is necessary, it must be explained in the PR description.

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

All checks must pass before requesting review.

## Testing policy

- Add unit tests for new logic.
- Add integration tests for supported combinations.
- Add at least one end-to-end test for each main feature.
- For bug fixes, write a failing test first, then implement the fix.
- Prefer tests that validate behavior and invariants rather than internal implementation details.

## Commit and PR guidance

- Keep commits focused and atomic.
- Use clear PR titles and explain:
  - the problem
  - the root cause
  - the fix
  - the validation evidence (tests, logs, screenshots, etc.)

In the PR description, also state:

- Whether any public contract changes (CLI/API/config/schema).
- Whether any technical debt is introduced or removed.
- Any known risks and the rollback strategy if applicable.

## Code review expectations

Reviews focus on:

- Correctness and safety (including edge cases and error handling).
- Repository coherence (patterns, naming, structure, abstractions).
- Detection and removal of dead code.
- Avoiding or reducing duplication via appropriate factorization.
- Maintainability (complexity, clarity, test coverage).

Be prepared to:

- Justify new patterns or abstractions.
- Accept small refactors requested for coherence.
- Split changes if a PR becomes too broad.

## Code style

- Use typed enums for user-facing choices (`StrEnum`, `Flag`/`IntFlag`).
- Keep docstrings in Google style with typed `Args`, `Returns`, and `Raises` where relevant.
- Prefer modular, strategy-based architecture and backend abstractions.
- Keep functions and modules focused with clear responsibilities.
- Avoid introducing ad-hoc one-off patterns when a shared abstraction already exists.

## Technical debt and refactors

- Do not leave dead code behind.
- If duplication or debt is temporarily accepted, document it explicitly and, when possible, create a follow-up issue.
- Prefer small, incremental refactors over large, risky rewrites.
