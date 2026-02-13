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
  --manual-stopwords-file path/to/stopwords.yaml \
  --auto-stopwords \
  --auto-stopwords-min-doc-ratio 0.7 \
  --auto-stopwords-min-corpus-ratio 0.01 \
  --auto-stopwords-max-terms 200 \
  --pdf-ocr-fallback \
  --pdf-ocr-languages fra+eng \
  --pdf-ocr-dpi 200 \
  --pdf-ocr-min-chars 32 \
  --streaming-mode \
  --output data/out/ingest.json
```

Create your stopwords file yourself (YAML/CSV/TXT), then pass its path with `--manual-stopwords-file`.

## Important Options

- `--cleaning-options`: choose cleaning steps (`none`, `all`, `accent_normalization`, `token_cleanup`, etc.).
- `--manual-stopwords`: inline comma-separated stopwords.
- `--manual-stopwords-file`: extra stopwords from YAML/CSV/TXT.
- `--auto-stopwords*`: corpus-driven stopwords generation.
- `--pdf-ocr-fallback`: OCR fallback for scanned PDFs when embedded text is too low.
- `--pdf-ocr-languages`: OCR language codes (default `fra+eng`).
- `--pdf-ocr-dpi`: OCR rendering DPI (default `200`).
- `--pdf-ocr-min-chars`: minimum alphanumeric characters in embedded text required to skip OCR fallback (pages with fewer characters trigger OCR).
- `--pdf-ocr-tessdata`: optional tessdata directory path.
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
