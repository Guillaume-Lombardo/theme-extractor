# How To Run Extraction and Interpret Results

## Goal

Run one method with `extract` and read the unified topic output.

## Baseline TF-IDF

```bash
uv run theme-extractor extract \
  --method baseline_tfidf \
  --backend elasticsearch \
  --backend-url http://localhost:9200 \
  --index theme_extractor \
  --focus both \
  --query "match_all" \
  --output data/out/extract_tfidf.json
```

## BERTopic with embeddings

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
  --bertopic-clustering hdbscan \
  --bertopic-min-topic-size 5 \
  --bertopic-embedding-cache-enabled \
  --bertopic-embedding-cache-dir data/cache/embeddings \
  --bertopic-embedding-cache-version v1 \
  --output data/out/extract_bertopic.json
```

## How To Read Output

Main sections in extraction JSON:

- `topics`: topic-first result list.
- `document_topics`: optional document-to-topic links.
- `notes`: runtime fallback/execution notes.
- `metadata`: run context (method, backend, index, policy).

For each topic, prioritize:

- `label`: readable theme name.
- `keywords`: top descriptive terms with scores.
- `document_ids` / `representative_documents`: evidence pointers.
- `summary`: optional short explanation.

## Quality Tips

- For `significant_terms` / `significant_text`, use a focused query, not `match_all`.
- Compare `notes` across runs to detect fallback behavior.
- Use `--focus both` during exploration, then narrow later if needed.
