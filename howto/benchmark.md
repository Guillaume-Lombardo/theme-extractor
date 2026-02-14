# How To Benchmark Methods and Choose a Strategy

## Goal

Run several methods in one command and compare outputs on the same corpus slice.

## Example Run

Broad + semantic methods:

```bash
uv run theme-extractor benchmark \
  --methods baseline_tfidf,terms,keybert,bertopic,llm \
  --backend elasticsearch \
  --backend-url http://localhost:9200 \
  --index theme_extractor \
  --focus both \
  --query "match_all" \
  --bertopic-min-topic-size 5 \
  --search-size 200 \
  --output data/out/benchmark_all.json
```

Significant terms run:

```bash
uv run theme-extractor benchmark \
  --methods significant_terms \
  --backend elasticsearch \
  --backend-url http://localhost:9200 \
  --index theme_extractor \
  --query "facture OR copropriete OR impot" \
  --agg-field tokens \
  --focus both \
  --output data/out/benchmark_significant_terms.json
```

Significant text run:

```bash
uv run theme-extractor benchmark \
  --methods significant_text \
  --backend elasticsearch \
  --backend-url http://localhost:9200 \
  --index theme_extractor \
  --query "facture OR copropriete OR impot" \
  --agg-field content \
  --focus both \
  --output data/out/benchmark_significant_text.json
```

## Recommended Comparison Workflow

1. Run broad lexical baselines on `match_all`:
   - `baseline_tfidf`, `terms`.
2. Run significant methods with focused query:
   - `significant_terms`, `significant_text`.
3. Run semantic methods:
   - `keybert`, `bertopic`, optional `llm`.
4. Evaluate consistency and quality:
   - representative keywords quality
   - topic diversity
   - document-topic coherence
   - stability across seeds/runs

## Choosing a Best Candidate

Use these practical criteria:

- relevance: keywords are interpretable by domain users
- coverage: major document groups are represented
- separability: topics are distinct (low overlap)
- operational cost: runtime + dependencies + offline constraints

## Follow-Up Evaluation Command

```bash
uv run theme-extractor evaluate \
  --input data/out/benchmark_all.json \
  --output data/out/evaluation_benchmark_all.json
```

Use `evaluate` as a quantitative proxy, then confirm with manual domain review.
For detailed interpretation of metrics (including Jaccard), see `howto/evaluation.md`.
