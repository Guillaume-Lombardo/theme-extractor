# How To Run and Interpret Evaluation

## Goal

Compute quantitative proxy metrics from `extract` or `benchmark` outputs, then interpret them correctly.

## Run Evaluation

From one benchmark file:

```bash
uv run theme-extractor evaluate \
  --input data/out/benchmark_all.json \
  --output data/out/evaluation_benchmark_all.json
```

From multiple inputs (mixed extract + benchmark):

```bash
uv run theme-extractor evaluate \
  --input data/out/extract_tfidf.json data/out/benchmark_all.json \
  --output data/out/evaluation_mixed.json
```

## Metric Definitions

Main fields in `evaluation_*.json`:

- `summary`: counts of evaluated inputs (`input_count`, `extract_count`, `benchmark_count`).
- `extracts[*].metrics`: one metric block per evaluated extract payload.
- `benchmarks[*].metrics`: one metric block per evaluated benchmark payload.

Per-method metrics (in benchmark evaluation):

- `topic_count`: number of topics produced by the method.
- `document_topic_count`: number of document-topic links.
- `avg_keywords_per_topic`: average keyword count per topic.
- `topic_coherence_proxy`: lexical coherence proxy (higher usually means more internally consistent topics).
- `inter_topic_diversity`: topic separability proxy (higher means less overlap between topics).
- `inter_topic_mean_jaccard`: average topic overlap based on keyword sets (lower is better for separability).
- `cross_method_mean_jaccard`: average overlap between methods.

## What Is Jaccard?

Jaccard similarity between two keyword sets `A` and `B`:

`J(A, B) = |A ∩ B| / |A ∪ B|`

Interpretation:

- `0.0`: no shared keywords.
- `1.0`: exactly the same keyword sets.

In this project:

- lower `inter_topic_mean_jaccard` is generally better (topics are more distinct),
- moderate `cross_method_mean_jaccard` can be healthy (some agreement without collapse into duplicates).

## Interpretation Guidelines

Read metrics together, not in isolation:

- High coherence + low diversity can mean repeated/near-duplicate themes.
- High diversity + very low coherence can mean fragmented/noisy themes.
- Very low `topic_count` can indicate restrictive query/filters or sparse corpus.
- Very high `topic_count` with weak coherence can indicate over-segmentation.

Recommended process:

1. Start with `benchmark` on a representative query/corpus.
2. Run `evaluate`.
3. Review per-method metrics (`coherence`, `diversity`, Jaccard).
4. Validate top topics manually with business/domain users before final choice.

## Common Pitfalls

- `extract_count = 0` is normal when input only contains benchmark payload(s).
- Empty metrics usually mean empty extraction results upstream; inspect `notes` in extract/benchmark outputs first.
- Comparing methods with different query scopes can produce misleading Jaccard values.
