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

## Missing optional dependencies during ingestion

Symptom:
- files are skipped with messages like "Install 'pymupdf' ...".

Fix:

```bash
uv pip install pymupdf python-docx openpyxl python-pptx extract-msg
```

## Proxy and offline behavior

Tips:
- one-shot proxy: `--proxy-url http://proxy.company.local:8080`
- default proxy from env: `THEME_EXTRACTOR_PROXY_URL`
- strict offline mode for LLM:
  - `--offline-policy strict`
