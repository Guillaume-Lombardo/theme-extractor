# theme-extractor

AI-ready Python package to benchmark multiple theme/topic extraction strategies on heterogeneous corpora.

## CLI

Use one unified CLI with subcommands:

- `theme-extractor ingest`
- `theme-extractor extract`
- `theme-extractor benchmark`

Current phase exposes stable command contracts and raw JSON outputs.

## Project Agent Tooling

- `/AGENTS.md`: project operating rules
- `/agent.md`: agent charter
- `/plan.md`: phased delivery roadmap
- `/skills/architecture/SKILL.md`
- `/skills/testing/SKILL.md`
- `/skills/code-style/SKILL.md`
- `/skills/tooling/SKILL.md`

## Current State

This repository currently provides:

- agent governance and delivery skills
- phase 1 architecture contracts
- one unified CLI with subcommands
- normalized JSON output schemas (topic-first, with optional document-topic links)

## Local Baseline Testing With Docker

This project includes Docker Compose stacks to quickly test baseline extraction on a small local corpus in `data/raw/`.

### 1) Put Sample Documents In `data/raw/`

```bash
mkdir -p data/raw data/out
# Copy around a dozen files into data/raw/ (pdf/docx/xlsx/pptx/msg/md/html/txt...)
```

### 2) Start Elasticsearch (optionally with PostgreSQL)

Elasticsearch only:

```bash
docker compose -f docker/compose.elasticsearch.yaml up -d
```

Elasticsearch + PostgreSQL:

```bash
docker compose -f docker/compose.elasticsearch.yaml --profile postgres up -d
```

### 3) Index Corpus Into Backend

```bash
uv run python docker/index_corpus.py \
  --input data/raw \
  --backend-url http://localhost:9200 \
  --index theme_extractor \
  --cleaning-options all \
  --reset-index
```

### 4) Run Baseline Extractions

TF-IDF:

```bash
uv run theme-extractor extract \
  --method baseline_tfidf \
  --backend elasticsearch \
  --backend-url http://localhost:9200 \
  --index theme_extractor \
  --focus topics \
  --query "match_all" \
  --output data/out/extract_tfidf.json
```

Terms:

```bash
uv run theme-extractor extract \
  --method terms \
  --backend elasticsearch \
  --backend-url http://localhost:9200 \
  --index theme_extractor \
  --agg-field tokens \
  --focus topics \
  --output data/out/extract_terms.json
```

Significant Terms:

```bash
uv run theme-extractor extract \
  --method significant_terms \
  --backend elasticsearch \
  --backend-url http://localhost:9200 \
  --index theme_extractor \
  --agg-field tokens \
  --focus topics \
  --output data/out/extract_significant_terms.json
```

Significant Text:

```bash
uv run theme-extractor extract \
  --method significant_text \
  --backend elasticsearch \
  --backend-url http://localhost:9200 \
  --index theme_extractor \
  --agg-field content \
  --focus topics \
  --output data/out/extract_significant_text.json
```

All baselines in one benchmark run:

```bash
uv run theme-extractor benchmark \
  --methods baseline_tfidf,terms,significant_terms,significant_text \
  --backend elasticsearch \
  --backend-url http://localhost:9200 \
  --index theme_extractor \
  --focus both \
  --output data/out/benchmark_baselines.json
```

### OpenSearch Variant

Start:

```bash
docker compose -f docker/compose.opensearch.yaml up -d
```

Index (note port `9201`):

```bash
uv run python docker/index_corpus.py \
  --input data/raw \
  --backend-url http://localhost:9201 \
  --index theme_extractor \
  --cleaning-options all \
  --reset-index
```

Run extraction with OpenSearch:

```bash
uv run theme-extractor extract \
  --method baseline_tfidf \
  --backend opensearch \
  --backend-url http://localhost:9201 \
  --index theme_extractor \
  --focus topics \
  --output data/out/extract_tfidf_opensearch.json
```

### SQLite vs PostgreSQL

- For current quick baseline tests, SQLite/no SQL container is enough.
- PostgreSQL container is available in compose files via `--profile postgres` for future workflows.

### Cleanup

Stop and remove all containers + volumes:

```bash
docker compose -f docker/compose.elasticsearch.yaml down -v --remove-orphans
docker compose -f docker/compose.opensearch.yaml down -v --remove-orphans
```

Optionally remove generated outputs:

```bash
rm -f data/out/*.json
```
