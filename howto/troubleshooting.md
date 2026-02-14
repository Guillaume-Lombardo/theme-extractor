# Troubleshooting

## `significant_terms` / `significant_text` returns empty topics

Likely cause:

- Query too broad (`match_all`), so foreground and background are almost identical.

Fix:

- Use a focused query, for example:
  - `--query "facture OR impot OR copropriete"`
- Keep method-specific aggregation fields (requires separate runs, because `benchmark` accepts one shared `--agg-field`):
  - `significant_terms` with `--agg-field tokens`
  - `significant_text` with `--agg-field content`

## Topics dominated by stopwords (`de`, `le`, `the`, ...)

Likely causes:

- weak cleaning / stopword setup
- noisy non-business files in corpus

Fix:

- Re-index with full cleaning:
  - `--cleaning-options all`
- Keep default stopwords enabled (default behavior).
- Add manual stopwords:
  - inline: `--manual-stopwords "de,le,la,the,and,of"`
  - file-based: `--manual-stopwords-file path/to/stopwords.yaml` (create this file yourself)
- Enable automatic stopwords discovery:
  - `--auto-stopwords --auto-stopwords-min-doc-ratio 0.7 --auto-stopwords-min-corpus-ratio 0.01`
- Apply these options when indexing backend data (the `ingest` command only produces local QA JSON):
  - `uv run python docker/index_corpus.py --input data/raw --backend-url http://localhost:9200 --index theme_extractor --cleaning-options all --default-stopwords --manual-stopwords "de,le,la,the,and,of" --auto-stopwords --reset-index`

## Inspect cleaned text stored in backend

Use Elasticsearch/OpenSearch `_search` with source filtering:

```bash
curl -s "http://localhost:9200/theme_extractor/_search?pretty" \
  -H "Content-Type: application/json" \
  -d '{"size": 3, "_source": ["path", "content", "content_clean", "removed_stopword_count"], "query": {"match_all": {}}}'
```

Notes:

- `content`: cleaned text after stopword filtering (used by extraction defaults).
- `content_clean`: cleaned text before stopword filtering (debug/inspection field).

## Missing optional dependencies during ingestion

Symptom:

- files are skipped with messages like "Install 'pymupdf' ...".

Fix:

```bash
uv sync --group ingestion
```

## `.msg` ingestion misses expected context

Likely causes:

- metadata extraction disabled
- attachment policy too restrictive

Fix:

- include email metadata:
  - `--msg-include-metadata`
- include attachment names:
  - `--msg-attachments-policy names`
- include attachment textual payloads when useful:
  - `--msg-attachments-policy text`

## Scanned PDFs produce empty/near-empty content

Likely cause:

- PDF pages are image-only (no embedded text layer).

Fix:

- Enable OCR fallback during ingestion:
  - `--pdf-ocr-fallback --pdf-ocr-languages fra+eng --pdf-ocr-dpi 200`
- Tune OCR trigger threshold:
  - `--pdf-ocr-min-chars 32`
- If your OCR runtime needs an explicit tessdata path:
  - `--pdf-ocr-tessdata /path/to/tessdata`
- If you see warnings like `Failed loading language 'fra'`:
  - install the matching language pack in your tesseract runtime, or
  - run with available language(s), for example:
    - `--pdf-ocr-languages eng`

## Tesseract warning: missing `fra.traineddata` / `TESSDATA_PREFIX`

Symptom (example):

- `Error opening data file .../tessdata/fra.traineddata`
- `Please make sure the TESSDATA_PREFIX environment variable is set...`
- `Failed loading language 'fra'`

Meaning:

- OCR is enabled and `fra` is requested, but French language data is not found by tesseract.

Fix (macOS with Homebrew):

```bash
brew install tesseract tesseract-lang
export TESSDATA_PREFIX="$(brew --prefix)/share/tessdata"
```

Fix (Debian/Ubuntu):

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-fra tesseract-ocr-eng
export TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata
```

Checks:

```bash
tesseract --list-langs
```

Runtime alternatives:

- point directly to tessdata: `--pdf-ocr-tessdata /path/to/tessdata`
- use only installed languages: `--pdf-ocr-languages eng`
- disable OCR fallback: `--no-pdf-ocr-fallback`

## Legacy `.doc` files fail with OpenXML relationship errors

Symptom (example):

- `no relationship of type 'http://schemas.openxmlformats.org/.../officeDocument'`

Meaning:

- the file is legacy binary `.doc` (not OOXML `.docx`), and `python-docx` cannot parse it.

Fix:

- convert `.doc` to `.docx` before ingestion.

Example conversion with LibreOffice:

```bash
soffice --headless --convert-to docx --outdir /tmp /path/to/file.doc
```

## Proxy and offline behavior

Tips:

- one-shot proxy: `--proxy-url http://proxy.company.local:8080`
- default proxy from env: `THEME_EXTRACTOR_PROXY_URL`
- strict offline mode for LLM:
  - `--offline-policy strict`

## Evaluation report looks empty

Likely cause:

- evaluation input only contains benchmark payload(s), so `extract_count` is `0`.

Expected behavior:

- `Extract Metrics` may contain placeholder row.
- benchmark results should still be visible in:
  - `Benchmark Metrics`
  - `Benchmark Method Details` (per-method table).

Fix:

- confirm `evaluation_*.json` contains `benchmarks[*].metrics.per_method`.
- regenerate markdown report:
  - `uv run theme-extractor report --input data/out/evaluation_benchmark_all.json --output data/out/report_evaluation_benchmark_all.md`
