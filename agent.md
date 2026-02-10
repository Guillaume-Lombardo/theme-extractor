# agent.md

## Role
Pragmatic software agent for the `theme-extractor` package.

## Objective
Deliver a CLI-centric benchmark framework for theme/topic extraction strategies:
- baseline lexical/statistical methods (terms, significant terms/text, tf-idf)
- KeyBERT
- BERTopic variants
- LLM-based strategy

## Key Principles
- Same user experience across strategies.
- Same normalized output contract across strategies.
- Reproducible runs (explicit config, deterministic seeds when possible).
- Offline-first behavior, optional proxy usage.
- Explicit separation between ingestion, preprocessing, extraction, and restitution.

## Collaboration Contract
- Use MBAD to clarify unclear points before coding critical parts.
- Surface assumptions explicitly when requirements are incomplete.
- Prefer small, testable increments.
- Keep docs, skills, and plan synchronized with implementation.

## Definition Of Done (feature level)
A feature is done only if:
- implementation is complete and typed
- tests exist at relevant levels (unit/integration/end2end as needed)
- lint/format/type checks pass
- docs/plan updates are applied when architecture or behavior changes

## Non-Goals (for now)
- No UI/web app in first iterations.
- No hidden network dependency in core extraction logic.
