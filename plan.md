# plan.md

## Vision
Create a production-grade toolkit to compare topic extraction strategies on heterogeneous document corpora with a unified CLI and normalized outputs.

## Phased Plan

### Phase 0 - Agent Tooling (current)
- [x] Define agent charter (`agent.md`)
- [x] Define execution roadmap (`plan.md`)
- [x] Define project operating rules (`AGENTS.md`)
- [x] Create reusable project skills (`skills/*`)

### Phase 1 - Core Architecture
- [x] Define domain contracts:
  - corpus document model
  - normalized topic output model
  - extraction strategy interface
  - benchmark run metadata model
- [x] Define backend search abstraction for Elasticsearch/OpenSearch
- [x] Define CLI root and subcommands skeleton

### Phase 2 - Ingestion and Cleaning
- [x] Implement ingestion pipeline for:
  - pdf, doc/docx, xls/xlsx, ppt/pptx, msg, md, html, txt
- [x] Implement normalization/cleaning pipeline:
  - header/footer suppression
  - boilerplate removal
  - whitespace and encoding normalization
  - French accent normalization (`é/è/ê -> e`, etc.) with configurable behavior
  - language-aware token cleanup
- [x] Implement stopwords enrichment capabilities:
  - manual stopwords injection by user input/config
  - automatic stopwords discovery from corpus statistics
  - explicit toggle to disable automatic stopwords generation
- [x] Add ingestion quality tests on representative fixtures

### Phase 3 - Strategy Implementations
- [x] Baselines:
  - tf-idf
  - Elasticsearch/OpenSearch terms
  - Elasticsearch/OpenSearch significant_terms
  - Elasticsearch/OpenSearch significant_text
- [x] KeyBERT strategy
- [x] BERTopic strategy matrix:
  - embedding on/off
  - dimensionality reduction: none/svd/nmf/umap
  - clustering: kmeans/hdbscan
  - embedding model parameterized by user (default suggestion: bge-m3)
- [x] LLM strategy with offline-compatible fallback behavior

### Phase 4 - Unified Restitution and Topic Characterization
- [x] Standardize all outputs to one schema
- [x] Add topic characterization layer:
  - labels
  - representative keywords
  - representative documents
  - optional short summary/explanation
- [x] Add comparison report across strategies

### Phase 5 - Hardening
- [x] Expand integration/end2end scenarios
- [x] Validate offline and proxy execution modes
- [x] Finalize CLI docs and usage recipes
- [x] Pre-PR gate: all test suites green

### Phase 6 - Productization and Evaluation
- [x] Add `theme-extractor doctor` command:
  - validate runtime prerequisites (Python/package/env)
  - validate optional dependency groups availability (`elasticsearch`, `opensearch`, `bert`, `llm`)
  - validate local model directory and declared aliases for offline runs
  - optional backend connectivity check (opt-in)
- [x] Add CI workflow for pull requests:
  - `ruff format --check` + `ruff check`
  - `ty check src tests`
  - `pytest -m unit`
  - `pytest -m integration --no-cov`
  - `pytest -m end2end`
  - `pre-commit run --all-files`
- [x] Add markdown report generation from unified JSON outputs:
  - strategy summary
  - top topics with representative keywords/documents
  - overlap/comparison insights
- [x] Add quantitative evaluation utilities:
  - topic coherence proxies
  - inter-topic diversity
  - run-to-run stability (seed sensitivity)
- [ ] Improve ingestion robustness:
  - OCR fallback strategy for scanned PDFs
  - stronger header/footer detection on multipage docs
  - richer `.msg` support (metadata and attachment text extraction policy)
- [ ] Improve performance/scalability:
  - [x] streaming mode for large corpora (avoid retaining full text payloads)
  - [x] optional embedding cache with deterministic keys/versioning

### Phase 7 - Rationalize documentation
- [x] Make the README clearer for external users.
- [x] Make a folder `howto` containing markdown files (main commands, variant examples, and their ups and downs):
  - [x] How to ingest a corpus
  - [x] How to run theme extraction and interpret its report
  - [x] How to run a benchmark and choose the best strategy
  - [x] How to generate a report
- [x] Make a Sphinx documentation including docstrings, README, and how-to guides.

## Branch and PR Policy
- [ ] Execute each run/phase/feature in its own dedicated branch.
- [ ] Close each run/phase/feature with a GitHub Pull Request.

## MBAD Decisions (2026-02-10)
- [x] Primary deliverable: raw JSON output first.
- [x] Keep markdown report as a possible later phase.
- [x] Target strict offline mode, with optional preload/first-run model download mode.
- [x] Accept optional Python dependency for `.msg` parsing.
- [x] Keep a single CLI with subcommands.
- [x] Start BERTopic without embeddings using KMeans first (HDBSCAN optional extension).
- [x] Prioritize topic-first unified outputs, then document-topic associations.

## Risks To Address Early
- file format parsing variability (especially `.msg` and scanned PDFs)
- performance/memory footprint on large corpora
- ensuring consistent scoring semantics across heterogeneous methods
- dependency complexity for BERTopic stack
