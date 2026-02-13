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

```bash
uv sync --group elasticsearch
uv run theme-extractor doctor --output data/out/doctor.json
uv run theme-extractor ingest --input data/raw --output data/out/ingest.json
uv run theme-extractor benchmark \
  --methods baseline_tfidf,terms,significant_terms,significant_text,keybert,bertopic \
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
- [`/howto/docker-local.md`](howto/docker-local.md): local Docker stacks (Elasticsearch/OpenSearch)
- [`/howto/troubleshooting.md`](howto/troubleshooting.md): common failures and fixes

## Configuration

Use `.env.template` as bootstrap:

```bash
cp .env.template .env
set -a; source .env; set +a
```

Important variable groups:
- backend/runtime (`THEME_EXTRACTOR_BACKEND*`, `THEME_EXTRACTOR_PROXY_URL`)
- ingestion stopwords (`THEME_EXTRACTOR_DEFAULT_STOPWORDS_ENABLED`, `THEME_EXTRACTOR_AUTO_STOPWORDS_*`)
- local model resolution (`THEME_EXTRACTOR_LOCAL_MODELS_DIR`)
- BERTopic embedding cache (`THEME_EXTRACTOR_BERTOPIC_EMBEDDING_CACHE_*`)

## Project Governance

- [`AGENTS.md`](AGENTS.md): operating rules
- [`plan.md`](plan.md): phased roadmap
- [`agent.md`](agent.md): agent charter
