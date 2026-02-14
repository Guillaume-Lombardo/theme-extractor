# theme-extractor

Python toolkit to compare theme/topic extraction strategies on the same corpus, with:

- one unified CLI
- one unified JSON output schema
- offline-first execution options
- Elasticsearch/OpenSearch backend support

## Why this project

`theme-extractor` helps you answer one practical question:
"Which extraction strategy works best for my corpus and constraints?"

It lets you run baseline lexical methods, embedding-based methods, and LLM-assisted methods with consistent outputs for easier comparison.

## Core Commands

- `theme-extractor ingest`
- `theme-extractor extract`
- `theme-extractor benchmark`
- `theme-extractor evaluate`
- `theme-extractor report`
- `theme-extractor doctor`

## Quickstart

Prerequisite: start one backend first (local Docker guide: [`/howto/docker-local.md`](howto/docker-local.md)).

```bash
uv sync --group elasticsearch
uv sync --group ingestion
uv run theme-extractor doctor --output data/out/doctor.json
uv run theme-extractor ingest --input data/raw --output data/out/ingest.json
uv run theme-extractor ingest \
  --input data/raw \
  --reset-index \
  --index-backend \
  --cleaning-options all \
  --default-stopwords \
  --manual-stopwords "de,le,la,the,and,of" \
  --output data/out/ingest_indexed.json
uv run theme-extractor benchmark \
  --methods baseline_tfidf,terms,significant_terms,keybert,bertopic \
  --backend elasticsearch \
  --backend-url http://localhost:9200 \
  --index theme_extractor \
  --focus both \
  --output data/out/benchmark.json
uv run theme-extractor evaluate \
  --input data/out/benchmark.json \
  --output data/out/evaluation.json
uv run theme-extractor report \
  --input data/out/benchmark.json \
  --output data/out/report_benchmark.md
```

`ingest` always generates a local JSON QA payload; with `--index-backend`, it also indexes cleaned/filtered text into Elasticsearch/OpenSearch.

Run `significant_text` separately with `--agg-field content` (see [`/howto/benchmark.md`](howto/benchmark.md)).

## Methods Available

- Baselines:
  - `baseline_tfidf`
  - `terms`
  - `significant_terms`
  - `significant_text`
- Semantic:
  - `keybert`
  - `bertopic` (embedding on/off, reduction `none/svd/nmf/umap`, clustering `kmeans/hdbscan`)
- Generative:
  - `llm` (offline fallback behavior supported)

## What You Get

- Unified extraction schema:
  - topic-first output (`topics`)
  - optional document-topic links (`document_topics`)
  - execution notes and metadata (`notes`, `metadata`)
- Benchmark output:
  - per-method outputs
  - comparison block
- Quantitative proxies via `evaluate`:
  - topic coherence proxy
  - inter-topic diversity
  - run-to-run stability

## Documentation Map (How-To)

Detailed operations are intentionally kept in `/howto`:

- [`/howto/ingest.md`](howto/ingest.md): ingestion, cleaning, stopwords, streaming mode
- [`/howto/extract.md`](howto/extract.md): single-method extraction and interpretation
- [`/howto/benchmark.md`](howto/benchmark.md): multi-method comparison workflow
- [`/howto/report.md`](howto/report.md): markdown reporting workflow
- [`/howto/release.md`](howto/release.md): PyPI/TestPyPI release workflow
- [`/howto/docker-local.md`](howto/docker-local.md): local Docker stacks (Elasticsearch/OpenSearch)
- [`/howto/troubleshooting.md`](howto/troubleshooting.md): common failures and fixes

Sphinx documentation is available under `/docs` (includes README, how-to pages, and API docstrings).

CLI code is now organized under `src/theme_extractor/cli/`:

- `__init__.py`: CLI entrypoint (`main`)
- `argument_parser.py`: parser and flags
- `command_handlers.py`: command execution logic
- `ingest_backend_indexing.py`: ingest-side backend indexing helpers
- `common_runtime.py`: shared runtime helpers

Build locally:

```bash
uv run sphinx-build -b html docs docs/_build/html
```

## Configuration

Use `.env.template` as bootstrap:

```bash
cp .env.template .env
set -a; source .env; set +a
```

Important variable groups:

- backend/runtime (`THEME_EXTRACTOR_BACKEND*`, `THEME_EXTRACTOR_PROXY_URL`)
- ingestion stopwords (`THEME_EXTRACTOR_DEFAULT_STOPWORDS_ENABLED`, `THEME_EXTRACTOR_AUTO_STOPWORDS_*`)
- PDF OCR fallback (`THEME_EXTRACTOR_PDF_OCR_*`) for scanned PDFs
- `.msg` extraction (`THEME_EXTRACTOR_MSG_*`) for metadata and attachment policy
- local model resolution (`THEME_EXTRACTOR_LOCAL_MODELS_DIR`)
- BERTopic embedding cache (`THEME_EXTRACTOR_BERTOPIC_EMBEDDING_CACHE_*`)

## Dependency Groups

Install optional runtime groups with `uv`:

```bash
uv sync --group ingestion
```

`ingestion` includes parsers for PDF/Office/MSG formats (`pymupdf`, `python-docx`, `openpyxl`, `python-pptx`, `extract-msg`).

## Project Governance

- [`AGENTS.md`](AGENTS.md): operating rules
- [`plan.md`](plan.md): phased roadmap
- [`agent.md`](agent.md): agent charter

## Community Standards

- [`CONTRIBUTING.md`](CONTRIBUTING.md)
- [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md)
- [`SECURITY.md`](SECURITY.md)
- [`SUPPORT.md`](SUPPORT.md)
