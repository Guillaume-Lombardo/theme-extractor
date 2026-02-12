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

### 0) Optional `.env` Bootstrap

You do not strictly need a `.env` file, but it helps keep commands shorter and consistent.

```bash
cp .env.template .env
set -a; source .env; set +a
```

### 1) Put Sample Documents In `data/raw/`

```bash
mkdir -p data/raw data/out
# Copy around a dozen files into data/raw/ (pdf/docx/xlsx/pptx/msg/md/html/txt...)
```

Install optional ingestion dependencies for office/pdf parsing:

```bash
uv pip install pymupdf python-docx openpyxl python-pptx extract-msg
```

### 2) Start Elasticsearch (optionally with PostgreSQL)

Install backend client dependency first (required for `extract`/`benchmark` commands):

```bash
# For Elasticsearch backend
uv sync --group elasticsearch

# For OpenSearch backend (optional alternative)
# uv sync --group opensearch

# Optional BERTopic/embedding stack
# uv sync --group bert
```

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
  --input "${THEME_EXTRACTOR_INPUT_DIR:-data/raw}" \
  --backend-url "${THEME_EXTRACTOR_BACKEND_URL:-http://localhost:9200}" \
  --index "${THEME_EXTRACTOR_INDEX:-theme_extractor}" \
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
  --query "facture OR impot OR copropriete" \
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
  --query "facture OR impot OR copropriete" \
  --agg-field content \
  --focus topics \
  --output data/out/extract_significant_text.json
```

BERTopic (matrix options):

```bash
uv run theme-extractor extract \
  --method bertopic \
  --backend elasticsearch \
  --backend-url http://localhost:9200 \
  --index theme_extractor \
  --focus both \
  --query "match_all" \
  --bertopic-use-embeddings \
  --bertopic-embedding-model bge-m3 \
  --bertopic-dim-reduction svd \
  --bertopic-clustering kmeans \
  --bertopic-nr-topics 8 \
  --bertopic-min-topic-size 5 \
  --output data/out/extract_bertopic.json
```

Notes:
- If `sentence-transformers`, `umap-learn`, or `hdbscan` are missing, the CLI falls back to safe defaults and reports it in `notes`.
- For strict offline runs, preload all optional models/dependencies before execution.

Important note for `significant_terms` and `significant_text`:

- Do not use `--query "match_all"` if you want meaningful significant results.
- Significant aggregations compare a foreground subset (your query) against the corpus background.
- With `match_all`, foreground and background are effectively the same, so empty or weak results are expected.
- Use a focused business query (for example `facture`, `impot`, `copropriete`, `sinistre`, `contrat`) to surface truly distinctive terms.

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

### 5) Recommended "Quality Benchmark" Preset

For better interpretation quality on mixed corpora:

1. Run broad baselines (`baseline_tfidf`, `terms`) on `match_all`.
2. Run significant baselines with a focused query.

Broad corpus run:

```bash
uv run theme-extractor benchmark \
  --methods baseline_tfidf,terms \
  --backend elasticsearch \
  --backend-url http://localhost:9200 \
  --index theme_extractor \
  --query "match_all" \
  --focus both \
  --output data/out/benchmark_baselines_broad.json
```

Focused significant runs (recommended to keep method-specific aggregation fields):

```bash
uv run theme-extractor benchmark \
  --methods significant_terms \
  --backend elasticsearch \
  --backend-url http://localhost:9200 \
  --index theme_extractor \
  --query "impot" \
  --agg-field tokens \
  --focus both \
  --output data/out/benchmark_significant_terms.json
```

```bash
uv run theme-extractor benchmark \
  --methods significant_text \
  --backend elasticsearch \
  --backend-url http://localhost:9200 \
  --index theme_extractor \
  --query "impot" \
  --agg-field content \
  --focus both \
  --output data/out/benchmark_significant_text.json
```

### OpenSearch Variant

Start:

```bash
docker compose -f docker/compose.opensearch.yaml up -d
```

Index (note port `9201`):

```bash
uv run python docker/index_corpus.py \
  --input "${THEME_EXTRACTOR_INPUT_DIR:-data/raw}" \
  --backend-url "http://localhost:9201" \
  --index "${THEME_EXTRACTOR_INDEX:-theme_extractor}" \
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

## Troubleshooting

### `significant_terms` or `significant_text` returns empty topics

Likely cause:

- Query is too broad (`match_all`) so foreground ~= background.

What to do:

- Use a focused business query:
  - `--query "facture OR impot OR copropriete"`
- Keep method-specific aggregation fields:
  - `significant_terms` with `--agg-field tokens`
  - `significant_text` with `--agg-field content`

### Baseline results are dominated by stopwords (`de`, `le`, `the`, ...)

Likely causes:

- Mixed-language corpus without enough stopword filtering.
- Ingestion noise (OCR artifacts, technical markdown files mixed with business docs).

What to do:

- Re-index with full cleaning:
  - `--cleaning-options all`
- Exclude non-business files from `data/raw/` for benchmark runs.
- Add manual stopwords during ingest:
  - `--manual-stopwords "de,le,la,the,and,of"`
- Optionally enable automatic corpus-based stopwords:
  - `--auto-stopwords --auto-stopwords-min-doc-ratio 0.7`

### Terms results contain many one-character tokens (`a`, `b`, `c`, ...)

Likely causes:

- OCR/text extraction noise.
- Token field contains low-value tokens.

What to do:

- Ensure ingestion cleaning is enabled (`--cleaning-options all`).
- Re-index after cleanup (`--reset-index`).
- Verify you aggregate on `tokens` (not raw content) for `terms`.

### `ModuleNotFoundError: No module named 'theme_extractor'`

Likely cause:

- Editable package not correctly installed after local changes.

What to do:

```bash
uv pip install -e .
uv run theme-extractor --help
```

### `ConnectError: [Errno 61] Connection refused`

Likely cause:

- Elasticsearch/OpenSearch container is not up.

What to do:

```bash
docker compose -f docker/compose.elasticsearch.yaml up -d
docker ps
curl -s http://localhost:9200
```

### Elasticsearch 9 fails to start with upgrade error from 8.x data

Typical error:

- `cannot upgrade a node from version [8.x] directly to version [9.x]`

What to do for local disposable test stacks:

```bash
docker compose -f docker/compose.elasticsearch.yaml down -v --remove-orphans
docker compose -f docker/compose.elasticsearch.yaml up -d
```
