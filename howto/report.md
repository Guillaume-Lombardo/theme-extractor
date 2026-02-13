# How To Generate and Use Reports

## Goal
Convert extraction/benchmark JSON output into markdown for easier sharing.

## From `extract` output

```bash
uv run theme-extractor report \
  --input data/out/extract_tfidf.json \
  --output data/out/report_extract.md
```

## From `benchmark` output

```bash
uv run theme-extractor report \
  --input data/out/benchmark_all.json \
  --title "Benchmark Report - DSJ Corpus" \
  --output data/out/report_benchmark.md
```

## Interpretation Checklist

When reading report markdown:

- verify top topics are aligned with expected domain themes
- verify representative documents are relevant evidence
- look for overlap warnings in comparison sections
- cross-check with `evaluate` metrics when available

## Suggested Shareable Bundle

For one run, keep these files together:

- `extract_*.json` or `benchmark_*.json`
- `evaluation_*.json` (optional but recommended)
- `report_*.md`

This bundle gives traceability from raw output to human-readable synthesis.
