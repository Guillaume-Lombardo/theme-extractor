# theme-extractor

AI-ready Python package to benchmark multiple theme/topic extraction strategies on heterogeneous corpora.

## CLI

Use one unified CLI with subcommands:

- `theme-extractor ingest`
- `theme-extractor extract`
- `theme-extractor benchmark`
- `theme-extractor doctor`
- `theme-extractor report`
- `theme-extractor evaluate`

Current phase exposes stable command contracts and raw JSON outputs.

## Runtime Diagnostics (`doctor`)

Use `doctor` to quickly validate your local runtime before extraction runs.

Basic check:

```bash
uv run theme-extractor doctor --output data/out/doctor.json
```

With backend connectivity check:

```bash
uv run theme-extractor doctor \
  --backend elasticsearch \
  --backend-url http://localhost:9200 \
  --index theme_extractor \
  --check-backend \
  --output data/out/doctor_backend.json
```

What it checks:
- runtime context (python/platform/offline policy/backend/proxy)
- optional dependency groups (`elasticsearch`, `opensearch`, `bert`, `llm`)
- expected local model aliases in `THEME_EXTRACTOR_LOCAL_MODELS_DIR`
- optional backend connectivity probe (`--check-backend`)

## Markdown Reports (`report`)

Use `report` to convert one extract/benchmark JSON into a markdown summary.

From extraction output:

```bash
uv run theme-extractor report \
  --input data/out/extract_tfidf.json \
  --output data/out/report_extract.md
```

From benchmark output:

```bash
uv run theme-extractor report \
  --input data/out/benchmark_baselines.json \
  --title "Baseline Benchmark Report" \
  --output data/out/report_benchmark.md
```

## Quantitative Evaluation (`evaluate`)

Use `evaluate` to compute quantitative proxy metrics from one or multiple
extract/benchmark JSON payloads.

Single extract evaluation:

```bash
uv run theme-extractor evaluate \
  --input data/out/extract_keybert.json \
  --output data/out/evaluation_keybert.json
```

Multiple payload evaluation:

```bash
uv run theme-extractor evaluate \
  --input data/out/extract_keybert_seed42.json \
  --input data/out/extract_keybert_seed123.json \
  --input data/out/benchmark_baselines.json \
  --output data/out/evaluation_all.json
```

Provided proxies:
- topic coherence proxy (keyword-mass concentration / compactness fallback)
- inter-topic diversity (1 - mean pairwise keyword Jaccard)
- run-to-run stability per method (pairwise keyword Jaccard across runs)

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
- topic characterization layer on extraction outputs:
  - inferred labels
  - representative keywords
  - representative documents
  - short summary per topic
- benchmark comparison block with pairwise keyword overlap across strategies

## Local Baseline Testing With Docker

This project includes Docker Compose stacks to quickly test baseline extraction on a small local corpus in `data/raw/`.

### 0) Optional `.env` Bootstrap

You do not strictly need a `.env` file, but it helps keep commands shorter and consistent.

```bash
cp .env.template .env
set -a; source .env; set +a
```

Ingestion-related env vars available in `.env.template`:

- `THEME_EXTRACTOR_DEFAULT_STOPWORDS_ENABLED`
- `THEME_EXTRACTOR_AUTO_STOPWORDS_ENABLED`
- `THEME_EXTRACTOR_AUTO_STOPWORDS_MIN_DOC_RATIO`
- `THEME_EXTRACTOR_AUTO_STOPWORDS_MIN_CORPUS_RATIO`
- `THEME_EXTRACTOR_AUTO_STOPWORDS_MAX_TERMS`
- `THEME_EXTRACTOR_PROXY_URL` (optional runtime proxy default for CLI commands)

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

# Optional LLM provider client
# uv sync --group llm
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

KeyBERT:

```bash
uv run theme-extractor extract \
  --method keybert \
  --backend elasticsearch \
  --backend-url http://localhost:9200 \
  --index theme_extractor \
  --focus both \
  --query "match_all" \
  --keybert-use-embeddings \
  --keybert-embedding-model bge-m3 \
  --keybert-local-models-dir data/models \
  --fields content,filename,path \
  --source-field content \
  --topn 25 \
  --search-size 200 \
  --output data/out/extract_keybert.json
```

KeyBERT benchmark example:

```bash
uv run theme-extractor benchmark \
  --methods keybert,baseline_tfidf \
  --backend elasticsearch \
  --backend-url http://localhost:9200 \
  --index theme_extractor \
  --focus both \
  --output data/out/benchmark_keybert.json
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

Alternative with NMF reduction:

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
  --bertopic-dim-reduction nmf \
  --bertopic-clustering kmeans \
  --bertopic-nr-topics 8 \
  --bertopic-min-topic-size 5 \
  --output data/out/extract_bertopic-nmf-kmeans.json
```

Alternative with UMAP reduction:

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
  --bertopic-dim-reduction umap \
  --bertopic-clustering kmeans \
  --bertopic-nr-topics 8 \
  --bertopic-min-topic-size 5 \
  --output data/out/extract_bertopic-umap-kmeans.json
```

Notes:

- If `sentence-transformers`, `umap-learn`, or `hdbscan` are missing, the CLI falls back to safe defaults and reports it in `notes`.
- For embeddings, `--bertopic-embedding-model` accepts either:
  - a model id, or
  - a local path.
- `--bertopic-dim-reduction` supports `none`, `svd`, `nmf`, and `umap`.
- Convenience behavior: if you pass `--bertopic-embedding-model bge-m3` and `data/models/bge-m3` exists, the local path is used automatically.
- You can override the local alias directory with `--bertopic-local-models-dir` or `THEME_EXTRACTOR_LOCAL_MODELS_DIR`.
- For strict offline runs, preload all optional models/dependencies before execution.

LLM strategy (strict offline fallback by default):

```bash
uv run theme-extractor extract \
  --method llm \
  --backend elasticsearch \
  --backend-url http://localhost:9200 \
  --index theme_extractor \
  --focus both \
  --query "match_all" \
  --offline-policy strict \
  --output data/out/extract_llm.json
```

LLM strategy with provider call enabled (`preload_or_first_run`):

```bash
export OPENAI_API_KEY="<OPENAI_API_KEY>"
uv run theme-extractor extract \
  --method llm \
  --backend elasticsearch \
  --backend-url http://localhost:9200 \
  --index theme_extractor \
  --focus both \
  --query "match_all" \
  --offline-policy preload_or_first_run \
  --llm-provider openai \
  --llm-model "${THEME_EXTRACTOR_LLM_MODEL:-gpt-4o-mini}" \
  --output data/out/extract_llm_online.json
```

LLM notes:

- In `strict` mode, the strategy never performs network calls and falls back to TF-IDF.
- In `preload_or_first_run`, if API credentials are missing or provider runtime fails, the strategy still falls back to TF-IDF and records the reason in `notes`.
- If `keybert` is missing at runtime, the `keybert` method falls back to TF-IDF and records the fallback reason in `notes`.
- KeyBERT embedding flags mirror BERTopic behavior: `--keybert-use-embeddings`, `--keybert-embedding-model`, and `--keybert-local-models-dir`.

Ingestion stopwords notes:

- Default FR/EN stopwords are enabled during `ingest`.
- Disable them if you need raw token behavior: `--no-default-stopwords`.
- Add project-specific stopwords with files:
  - `--manual-stopwords-file config/stopwords.yaml`
  - `--manual-stopwords-file config/stopwords.csv`
- Auto stopwords use both document coverage and corpus frequency:
  - `--auto-stopwords --auto-stopwords-min-doc-ratio 0.7 --auto-stopwords-min-corpus-ratio 0.01`
- Large corpus memory guard:
  - compact mode is enabled by default (`--streaming-mode`)
  - disable only for debugging/comparison (`--no-streaming-mode`)

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

### 6) Validate Offline and Proxy Modes

Strict offline validation (LLM fallback without network):

```bash
uv run theme-extractor extract \
  --method llm \
  --backend elasticsearch \
  --backend-url http://localhost:9200 \
  --index theme_extractor \
  --offline-policy strict \
  --focus topics \
  --output data/out/extract_llm_strict_offline.json
```

Proxy validation (single-run explicit proxy):

```bash
uv run theme-extractor extract \
  --method terms \
  --backend elasticsearch \
  --backend-url http://localhost:9200 \
  --index theme_extractor \
  --proxy-url http://proxy.company.local:8080 \
  --focus topics \
  --output data/out/extract_terms_proxy.json
```

Notes:

- `--proxy-url` sets runtime `HTTP_PROXY/HTTPS_PROXY` for the command execution.
- You can define a default proxy with `THEME_EXTRACTOR_PROXY_URL` in `.env`.
- For one-off overrides, CLI flag value takes precedence.

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
- Keep default FR/EN stopwords enabled (nltk if available, otherwise fallback lists):
  - enabled by default, disable only if needed with `--no-default-stopwords`
- Exclude non-business files from `data/raw/` for benchmark runs.
- Add manual stopwords during ingest:
  - inline: `--manual-stopwords "de,le,la,the,and,of"`
  - file-based: `--manual-stopwords-file config/stopwords.yaml`
  - file-based CSV: `--manual-stopwords-file config/stopwords.csv`
- Enable corpus-based automatic stopwords with stronger filtering:
  - `--auto-stopwords --auto-stopwords-min-doc-ratio 0.7 --auto-stopwords-min-corpus-ratio 0.01`

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

### Proxy was provided but network calls still fail

Likely causes:

- Proxy URL is malformed or unreachable from your host.
- Corporate proxy requires authentication not included in URL.
- Backend URL is blocked by network policy.

What to do:

```bash
# 1) Verify proxy setting is exported in current shell
echo "$THEME_EXTRACTOR_PROXY_URL"

# 2) Run command with explicit proxy to avoid env ambiguity
uv run theme-extractor extract --method terms --proxy-url http://<proxy-host>:<port> --focus topics

# 3) Test backend reachability directly
curl -x http://<proxy-host>:<port> -s http://localhost:9200
```

### Elasticsearch 9 fails to start with upgrade error from 8.x data

Typical error:

- `cannot upgrade a node from version [8.x] directly to version [9.x]`

What to do for local disposable test stacks:

```bash
docker compose -f docker/compose.elasticsearch.yaml down -v --remove-orphans
docker compose -f docker/compose.elasticsearch.yaml up -d
```
