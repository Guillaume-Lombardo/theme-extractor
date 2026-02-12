"""Quantitative proxy evaluation utilities for extract/benchmark outputs."""

from __future__ import annotations

import itertools
import json
from typing import TYPE_CHECKING, Any

from theme_extractor.domain import BenchmarkOutput, UnifiedExtractionOutput
from theme_extractor.errors import ThemeExtractorError

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

_MAX_TOP_KEYWORDS = 5


class UnsupportedEvaluationPayloadError(ValueError, ThemeExtractorError):
    """Raised when one JSON payload is not a supported extract/benchmark shape."""

    def __init__(self) -> None:
        """Build unsupported evaluation payload exception."""
        super().__init__("Input JSON is not a supported extract/benchmark payload.")


def _topic_keyword_terms(output: UnifiedExtractionOutput) -> list[set[str]]:
    """Collect normalized keyword-term sets per topic.

    Args:
        output (UnifiedExtractionOutput): Unified extraction output.

    Returns:
        list[set[str]]: One keyword set per topic.

    """
    return [
        {keyword.term.lower().strip() for keyword in topic.keywords if keyword.term and keyword.term.strip()}
        for topic in output.topics
    ]


def _safe_mean(values: Iterable[float]) -> float | None:
    """Compute mean on finite list values.

    Args:
        values (Iterable[float]): Numeric values.

    Returns:
        float | None: Mean value, or None when empty.

    """
    values_list = list(values)
    if not values_list:
        return None
    return float(sum(values_list) / len(values_list))


def _pairwise_jaccard(sets: Sequence[set[str]]) -> list[float]:
    """Compute pairwise Jaccard coefficients between sets.

    Args:
        sets (Sequence[set[str]]): Input sets.

    Returns:
        list[float]: Pairwise Jaccard values.

    """
    scores: list[float] = []
    for left, right in itertools.combinations(sets, 2):
        union = left | right
        if not union:
            continue
        scores.append(float(len(left & right) / len(union)))
    return scores


def _topic_coherence_proxy(output: UnifiedExtractionOutput) -> float | None:
    """Compute one lightweight topic coherence proxy.

    Proxy definition:
    - for each topic, keep top keyword scores (when available)
    - coherence is top-keyword-mass concentration in [0, 1]
    - if no scores exist, fallback to compactness proxy `min(1, 3/num_keywords)`

    Args:
        output (UnifiedExtractionOutput): Unified extraction output.

    Returns:
        float | None: Coherence proxy in [0, 1], or None when unavailable.

    """
    per_topic_scores: list[float] = []
    for topic in output.topics:
        scored = [
            float(keyword.score)
            for keyword in topic.keywords
            if isinstance(keyword.score, (int, float)) and keyword.score > 0
        ]
        if scored:
            ranked = sorted(scored, reverse=True)
            top_mass = sum(ranked[:_MAX_TOP_KEYWORDS])
            total_mass = sum(ranked)
            if total_mass > 0:
                per_topic_scores.append(float(top_mass / total_mass))
            continue

        keyword_count = len([keyword for keyword in topic.keywords if keyword.term])
        if keyword_count > 0:
            per_topic_scores.append(float(min(1.0, 3.0 / keyword_count)))

    return _safe_mean(per_topic_scores)


def evaluate_extract_output(output: UnifiedExtractionOutput) -> dict[str, Any]:
    """Evaluate one extraction output with proxy quantitative metrics.

    Args:
        output (UnifiedExtractionOutput): Extraction output.

    Returns:
        dict[str, Any]: Evaluation metrics payload.

    """
    topic_sets = _topic_keyword_terms(output)
    pairwise_jaccard = _pairwise_jaccard(topic_sets)
    diversity = None
    mean_jaccard = _safe_mean(pairwise_jaccard)
    if mean_jaccard is not None:
        diversity = float(1.0 - mean_jaccard)

    avg_keywords_per_topic = _safe_mean(len(topic_set) for topic_set in topic_sets)

    return {
        "method": output.metadata.method.value,
        "topic_count": len(output.topics),
        "document_topic_count": len(output.document_topics or []),
        "avg_keywords_per_topic": avg_keywords_per_topic,
        "topic_coherence_proxy": _topic_coherence_proxy(output),
        "inter_topic_diversity": diversity,
        "inter_topic_mean_jaccard": mean_jaccard,
    }


def evaluate_benchmark_output(output: BenchmarkOutput) -> dict[str, Any]:
    """Evaluate one benchmark output.

    Args:
        output (BenchmarkOutput): Benchmark output payload.

    Returns:
        dict[str, Any]: Benchmark evaluation metrics.

    """
    per_method: dict[str, dict[str, Any]] = {}
    aggregate_sets: list[set[str]] = []

    for method in output.methods:
        method_key = method.value
        method_output = output.outputs.get(method_key)
        if method_output is None:
            continue

        metrics = evaluate_extract_output(method_output)
        per_method[method_key] = metrics
        aggregate_sets.append(
            {
                keyword.term.lower().strip()
                for topic in method_output.topics
                for keyword in topic.keywords
                if keyword.term and keyword.term.strip()
            },
        )

    cross_method_jaccard_values = _pairwise_jaccard(aggregate_sets)
    cross_method_mean = _safe_mean(cross_method_jaccard_values)

    return {
        "method_count": len(per_method),
        "per_method": per_method,
        "cross_method_mean_jaccard": cross_method_mean,
    }


def _load_payload(path: Path) -> BenchmarkOutput | UnifiedExtractionOutput:
    """Load one extract/benchmark payload from JSON file.

    Args:
        path (Path): Input JSON file path.

    Raises:
        UnsupportedEvaluationPayloadError: If payload shape is unsupported.

    Returns:
        BenchmarkOutput | UnifiedExtractionOutput: Parsed payload.

    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and payload.get("command") == "benchmark":
        return BenchmarkOutput.model_validate(payload)
    if isinstance(payload, dict) and "topics" in payload and "metadata" in payload:
        return UnifiedExtractionOutput.model_validate(payload)
    raise UnsupportedEvaluationPayloadError


def evaluate_payload_files(input_paths: Sequence[Path]) -> dict[str, Any]:
    """Evaluate one or many JSON payload files.

    Args:
        input_paths (Sequence[Path]): Input JSON file paths.

    Returns:
        dict[str, Any]: Aggregated quantitative evaluation payload.

    """
    extracts: list[dict[str, Any]] = []
    benchmarks: list[dict[str, Any]] = []
    stability_groups: dict[str, list[set[str]]] = {}

    for path in input_paths:
        payload = _load_payload(path)
        if isinstance(payload, BenchmarkOutput):
            benchmarks.append(
                {
                    "path": str(path),
                    "metrics": evaluate_benchmark_output(payload),
                },
            )
            continue

        extract_metrics = evaluate_extract_output(payload)
        extracts.append(
            {
                "path": str(path),
                "metrics": extract_metrics,
            },
        )

        method_key = payload.metadata.method.value
        topic_terms = {
            keyword.term.lower().strip()
            for topic in payload.topics
            for keyword in topic.keywords
            if keyword.term and keyword.term.strip()
        }
        stability_groups.setdefault(method_key, []).append(topic_terms)

    stability_by_method: dict[str, Any] = {}
    for method_key, run_keyword_sets in stability_groups.items():
        pairwise = _pairwise_jaccard(run_keyword_sets)
        stability_by_method[method_key] = {
            "run_count": len(run_keyword_sets),
            "pairwise_jaccard_mean": _safe_mean(pairwise),
            "pairwise_comparisons": len(pairwise),
        }

    return {
        "schema_version": "1.0",
        "command": "evaluate",
        "summary": {
            "input_count": len(input_paths),
            "extract_count": len(extracts),
            "benchmark_count": len(benchmarks),
        },
        "extracts": extracts,
        "benchmarks": benchmarks,
        "stability": {
            "by_method": stability_by_method,
        },
    }
