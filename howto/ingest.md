# How To Ingest a Corpus

## Goal
Create a normalized ingestion JSON payload from local files.

## Minimal Run

```bash
uv run theme-extractor ingest \
  --input data/raw \
  --output data/out/ingest.json
```

## Recommended Run (mixed corpus)

```bash
uv run theme-extractor ingest \
  --input data/raw \
  --recursive \
  --cleaning-options all \
  --manual-stopwords "de,le,la,the,and,of" \
  --manual-stopwords-file config/stopwords.yaml \
  --auto-stopwords \
  --auto-stopwords-min-doc-ratio 0.7 \
  --auto-stopwords-min-corpus-ratio 0.01 \
  --auto-stopwords-max-terms 200 \
  --streaming-mode \
  --output data/out/ingest.json
```

## Important Options

- `--cleaning-options`: choose cleaning steps (`none`, `all`, `accent_normalization`, `token_cleanup`, etc.).
- `--manual-stopwords`: inline comma-separated stopwords.
- `--manual-stopwords-file`: extra stopwords from YAML/CSV/TXT.
- `--auto-stopwords*`: corpus-driven stopwords generation.
- `--streaming-mode`: compact ingestion mode for large corpora (default enabled).

## Output Validation

Check these fields in `data/out/ingest.json`:

- `processed_documents` and `skipped_documents`
- `manual_stopwords` and `auto_stopwords`
- `streaming_mode`
- `documents[*].token_count` and `documents[*].removed_stopword_count`

## Common Pitfalls

- Missing parser dependency for PDF/Office formats:
  - install: `uv pip install pymupdf python-docx openpyxl python-pptx extract-msg`
- Empty ingestion:
  - verify supported file extensions and `--input` path.
