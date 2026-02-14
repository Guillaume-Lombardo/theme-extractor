# Review Follow-up Skill

Use this skill when the user says a PR review has been done and asks to process reviewer comments.

## Goal
Process GitHub PR review comments end-to-end:
1. fetch review threads/comments,
2. assess validity,
3. implement fixes when appropriate,
4. push changes,
5. resolve addressed threads.

## Workflow
1. Identify current branch and linked open PR.
2. After creating/updating the PR, wait 60 seconds before the first status check.
3. Poll every 60 seconds until BOTH conditions are met:
   - CI checks are completed and passing,
   - Copilot review is present on the PR.
4. Fetch review threads with:
   - `python3 /Users/g1lom/.codex/skills/gh-address-comments/scripts/fetch_comments.py`
5. Build a short numbered list of threads with:
   - issue summary,
   - validity assessment (`valid`, `partially valid`, `not needed`),
   - intended action.
6. Analyze Copilot comments and act according to relevance:
   - implement fixes for `valid`,
   - implement minimal safe adjustments for `partially valid`,
   - do not implement `not needed` and document rationale in PR discussion.
7. Apply code/test/documentation fixes for valid items.
8. Run quality gates:
   - `uv run ruff check .`
   - `uv run ty check src tests`
   - `uv run pytest -m unit`
   - `uv run pytest -m integration`
   - `uv run pytest -m end2end`
9. Commit and push.
10. Resolve threads for addressed comments via GraphQL mutation `resolveReviewThread`.
11. Report what was fixed, which threads were resolved, and what was intentionally not changed.

## Thread resolution note
Only resolve a thread when:
- code or docs were actually updated to address it, or
- a clear rationale is provided in PR discussion for rejecting it.
