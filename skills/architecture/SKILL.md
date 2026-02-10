---
name: architecture
description: Design and evolve the software architecture of theme-extractor with explicit boundaries, strategy interfaces, and backend abstractions. Use when defining module structure, introducing extraction strategies, or implementing Elasticsearch/OpenSearch interoperability.
---

# Architecture Skill

## Purpose
Design a maintainable architecture with clear contracts and low coupling.

## Workflow
1. Run MBAD before deciding architecture changes.
2. Define or update typed contracts first (models, protocols, interfaces).
3. Apply strategy pattern for extraction methods.
4. Decouple ingestion, preprocessing, extraction, and output serialization.
5. Record architecture tradeoffs in `plan.md`.

## Mandatory Decisions
- Define one thin search backend protocol with:
  - `search_documents(...)`
  - `terms_aggregation(...)`
  - `significant_terms_aggregation(...)`
  - `significant_text_aggregation(...)`
- Implement one Elasticsearch adapter.
- Implement one OpenSearch adapter.
- Keep one normalized result schema shared by all extractors.

## Deliverables
- Update interfaces and protocols.
- Update module boundaries.
- Update plan entries for architecture decisions.
