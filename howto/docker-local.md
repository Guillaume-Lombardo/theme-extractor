# How To Run Local Docker Backends

## Goal
Start Elasticsearch or OpenSearch quickly to test ingestion/extraction workflows on local files.

## Prerequisites

```bash
uv sync --group elasticsearch
# Optional alternative:
# uv sync --group opensearch
```

For office/pdf ingestion support:

```bash
uv pip install pymupdf python-docx openpyxl python-pptx extract-msg
```

## Elasticsearch Stack

Start:

```bash
docker compose -f docker/compose.elasticsearch.yaml up -d
```

With PostgreSQL profile:

```bash
docker compose -f docker/compose.elasticsearch.yaml --profile postgres up -d
```

Index local corpus:

```bash
uv run python docker/index_corpus.py \
  --input "${THEME_EXTRACTOR_INPUT_DIR:-data/raw}" \
  --backend-url "${THEME_EXTRACTOR_BACKEND_URL:-http://localhost:9200}" \
  --index "${THEME_EXTRACTOR_INDEX:-theme_extractor}" \
  --cleaning-options all \
  --default-stopwords \
  --manual-stopwords "de,le,la,the,and,of" \
  --auto-stopwords \
  --auto-stopwords-min-doc-ratio 0.7 \
  --auto-stopwords-min-corpus-ratio 0.01 \
  --auto-stopwords-max-terms 200 \
  --reset-index
```

## OpenSearch Stack

Start:

```bash
docker compose -f docker/compose.opensearch.yaml up -d
```

Index corpus (default local port in this setup: `9201`):

```bash
uv run python docker/index_corpus.py \
  --input "${THEME_EXTRACTOR_INPUT_DIR:-data/raw}" \
  --backend-url "http://localhost:9201" \
  --index "${THEME_EXTRACTOR_INDEX:-theme_extractor}" \
  --cleaning-options all \
  --default-stopwords \
  --manual-stopwords "de,le,la,the,and,of" \
  --auto-stopwords \
  --auto-stopwords-min-doc-ratio 0.7 \
  --auto-stopwords-min-corpus-ratio 0.01 \
  --auto-stopwords-max-terms 200 \
  --reset-index
```

## Cleanup

```bash
docker compose -f docker/compose.elasticsearch.yaml down -v --remove-orphans
docker compose -f docker/compose.opensearch.yaml down -v --remove-orphans
```

Optional output cleanup:

```bash
rm -f data/out/*.json
```
