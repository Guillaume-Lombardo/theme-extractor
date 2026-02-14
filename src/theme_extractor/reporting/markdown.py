"""Markdown report generation from extraction, benchmark, and evaluation payloads."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from theme_extractor.domain import BenchmarkOutput, UnifiedExtractionOutput
from theme_extractor.errors import ThemeExtractorError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

_MAX_TOPIC_LINES = 10
_MAX_DOC_LINES = 15
_EMPTY_TABLE_DATA_ROW_INDEX = 2


class UnsupportedReportPayloadError(ValueError, ThemeExtractorError):
    """Raised when input JSON cannot be parsed as extract/benchmark payload."""

    def __init__(self) -> None:
        """Build unsupported report payload exception."""
        super().__init__("Input JSON is not a supported extract/benchmark/evaluate payload.")


def _fmt_score(score: float | None) -> str:
    """Format optional numeric score for markdown rendering.

    Args:
        score (float | None): Optional score value.

    Returns:
        str: Formatted score string.

    """
    if score is None:
        return "-"
    return f"{score:.4f}"


def _keywords_preview(keywords: Sequence[Any], *, top_n: int = 5) -> str:
    """Build one compact keywords preview string.

    Args:
        keywords (Sequence[Any]): Keyword objects exposing `term`.
        top_n (int): Maximum number of terms in preview.

    Returns:
        str: Comma-separated keyword preview.

    """
    terms = [str(keyword.term) for keyword in keywords if getattr(keyword, "term", "")]
    if not terms:
        return "-"
    return ", ".join(terms[: max(1, top_n)])


def _render_topic_lines(output: UnifiedExtractionOutput, *, top_n: int = _MAX_TOPIC_LINES) -> list[str]:
    """Render markdown table lines for one extraction output topics.

    Args:
        output (UnifiedExtractionOutput): Unified extraction output.
        top_n (int): Maximum number of topics to render.

    Returns:
        list[str]: Markdown lines.

    """
    lines = [
        "| Topic ID | Label | Score | Keywords | Documents |",
        "| --- | --- | ---: | --- | ---: |",
    ]
    lines.extend(
        [
            "| "
            f"{topic.topic_id} | {topic.label or '-'} | {_fmt_score(topic.score)} | "
            f"{_keywords_preview(topic.keywords)} | {len(topic.document_ids)} |"
            for topic in output.topics[: max(1, top_n)]
        ],
    )

    if len(lines) == _EMPTY_TABLE_DATA_ROW_INDEX:
        lines.append("| - | - | - | - | - |")
    return lines


def _render_document_links(output: UnifiedExtractionOutput, *, top_n: int = _MAX_DOC_LINES) -> list[str]:
    """Render markdown table lines for document-topic links.

    Args:
        output (UnifiedExtractionOutput): Unified extraction output.
        top_n (int): Maximum number of links to render.

    Returns:
        list[str]: Markdown lines.

    """
    lines = [
        "| Document ID | Topic ID | Rank | Probability |",
        "| --- | ---: | ---: | ---: |",
    ]

    links = output.document_topics or []
    for link in links[: max(1, top_n)]:
        probability = "-" if link.probability is None else f"{link.probability:.4f}"
        rank = "-" if link.rank is None else str(link.rank)
        lines.append(
            f"| {link.document_id} | {link.topic_id} | {rank} | {probability} |",
        )

    if len(lines) == _EMPTY_TABLE_DATA_ROW_INDEX:
        lines.append("| - | - | - | - |")
    return lines


def render_extract_markdown(output: UnifiedExtractionOutput, *, title: str | None = None) -> str:
    """Render one extraction output as markdown report.

    Args:
        output (UnifiedExtractionOutput): Extraction output payload.
        title (str | None): Optional report title override.

    Returns:
        str: Markdown report content.

    """
    header = title or "Theme Extractor Report"
    lines = [
        f"# {header}",
        "",
        "## Run Metadata",
        "",
        f"- `command`: `{output.metadata.command.value}`",
        f"- `method`: `{output.metadata.method.value}`",
        f"- `backend`: `{output.metadata.backend.value if output.metadata.backend else '-'}`",
        f"- `index`: `{output.metadata.index or '-'}`",
        f"- `offline_policy`: `{output.metadata.offline_policy.value}`",
        f"- `generated_at`: `{output.metadata.generated_at.isoformat()}`",
        "",
        "## Topics",
        "",
        *_render_topic_lines(output),
        "",
    ]

    if output.document_topics is not None:
        lines.extend(
            [
                "## Document Associations",
                "",
                *_render_document_links(output),
                "",
            ],
        )

    if output.notes:
        lines.extend(
            [
                "## Notes",
                "",
                *[f"- {note}" for note in output.notes],
                "",
            ],
        )

    return "\n".join(lines).rstrip() + "\n"


def render_benchmark_markdown(output: BenchmarkOutput, *, title: str | None = None) -> str:
    """Render one benchmark output as markdown report.

    Args:
        output (BenchmarkOutput): Benchmark payload.
        title (str | None): Optional report title override.

    Returns:
        str: Markdown report content.

    """
    header = title or "Theme Extractor Benchmark Report"
    lines = [
        f"# {header}",
        "",
        "## Run Metadata",
        "",
        f"- `command`: `{output.command.value}`",
        f"- `generated_at`: `{output.generated_at.isoformat()}`",
        f"- `methods`: `{', '.join(method.value for method in output.methods)}`",
        "",
    ]

    pairwise_raw = output.comparison.get("pairwise_overlap", []) if output.comparison else []
    pairwise: list[dict[str, Any]] = []
    if isinstance(pairwise_raw, list):
        for item in pairwise_raw:
            if not isinstance(item, dict):
                continue
            pairwise.append({str(key): value for key, value in item.items()})
    lines.extend(
        [
            "## Method Comparison",
            "",
            "| Left | Right | Overlap Count | Jaccard |",
            "| --- | --- | ---: | ---: |",
        ],
    )
    lines.extend(
        [
            f"| {item.get('left_method', '-')} | {item.get('right_method', '-')} | "
            f"{item.get('overlap_count', 0)} | {float(item.get('jaccard', 0.0)):.4f} |"
            for item in pairwise
        ],
    )
    if not pairwise:
        lines.append("| - | - | - | - |")

    lines.extend(["", "## Strategy Details", ""])
    for method in output.methods:
        method_key = method.value
        strategy_output = output.outputs.get(method_key)
        if strategy_output is None:
            continue

        lines.extend(
            [
                f"### `{method_key}`",
                "",
                *_render_topic_lines(strategy_output, top_n=5),
                "",
            ],
        )

    return "\n".join(lines).rstrip() + "\n"


def _render_evaluation_extract_rows(payload: dict[str, Any]) -> list[str]:
    """Render evaluation extract metrics table rows.

    Args:
        payload (dict[str, Any]): Evaluation payload.

    Returns:
        list[str]: Markdown table rows.

    """
    rows = [
        "| Path | Method | Topics | Documents | Avg Keywords/Topic | Coherence Proxy |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for extract in payload.get("extracts", []):
        if not isinstance(extract, dict):
            continue
        metrics = extract.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        rows.append(
            f"| {extract.get('path', '-')} | {metrics.get('method', '-')} | "
            f"{metrics.get('topic_count', 0)} | {metrics.get('document_topic_count', 0)} | "
            f"{float(metrics.get('avg_keywords_per_topic', 0.0)):.4f} | "
            f"{float(metrics.get('topic_coherence_proxy', 0.0)):.4f} |",
        )
    if len(rows) == _EMPTY_TABLE_DATA_ROW_INDEX:
        rows.append("| - | - | - | - | - | - |")
    return rows


def _render_evaluation_benchmark_rows(payload: dict[str, Any]) -> list[str]:
    """Render evaluation benchmark metrics table rows.

    Args:
        payload (dict[str, Any]): Evaluation payload.

    Returns:
        list[str]: Markdown table rows.

    """
    rows = [
        "| Path | Methods | Cross-method Mean Jaccard |",
        "| --- | ---: | ---: |",
    ]
    for benchmark in payload.get("benchmarks", []):
        if not isinstance(benchmark, dict):
            continue
        metrics = benchmark.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        rows.append(
            f"| {benchmark.get('path', '-')} | {metrics.get('method_count', 0)} | "
            f"{_fmt_score(metrics.get('cross_method_mean_jaccard'))} |",
        )
    if len(rows) == _EMPTY_TABLE_DATA_ROW_INDEX:
        rows.append("| - | - | - |")
    return rows


def _render_evaluation_per_method_rows(metrics: dict[str, Any]) -> list[str]:
    """Render per-method rows from one benchmark evaluation metrics payload.

    Args:
        metrics (dict[str, Any]): Benchmark metrics object.

    Returns:
        list[str]: Markdown table rows.

    """
    rows = [
        "| Method | Topics | Documents | Avg Keywords/Topic | Coherence | Diversity | Mean Jaccard |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    per_method = metrics.get("per_method", {})
    if not isinstance(per_method, dict):
        per_method = {}
    for method_name, method_metrics in per_method.items():
        if not isinstance(method_metrics, dict):
            continue
        rows.append(
            f"| {method_name} | "
            f"{method_metrics.get('topic_count', 0)} | "
            f"{method_metrics.get('document_topic_count', 0)} | "
            f"{float(method_metrics.get('avg_keywords_per_topic', 0.0)):.4f} | "
            f"{_fmt_score(method_metrics.get('topic_coherence_proxy'))} | "
            f"{_fmt_score(method_metrics.get('inter_topic_diversity'))} | "
            f"{_fmt_score(method_metrics.get('inter_topic_mean_jaccard'))} |",
        )
    if len(rows) == _EMPTY_TABLE_DATA_ROW_INDEX:
        rows.append("| - | - | - | - | - | - | - |")
    return rows


def render_evaluation_markdown(payload: dict[str, Any], *, title: str | None = None) -> str:
    """Render one evaluation output as markdown report.

    Args:
        payload (dict[str, Any]): Evaluation payload.
        title (str | None): Optional report title override.

    Returns:
        str: Markdown report content.

    """
    summary = payload.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}
    lines = [
        f"# {title or 'Theme Extractor Evaluation Report'}",
        "",
        "## Summary",
        "",
        f"- `input_count`: `{summary.get('input_count', 0)}`",
        f"- `extract_count`: `{summary.get('extract_count', 0)}`",
        f"- `benchmark_count`: `{summary.get('benchmark_count', 0)}`",
        "",
        "## Extract Metrics",
        "",
        *_render_evaluation_extract_rows(payload),
        "",
        "## Benchmark Metrics",
        "",
        *_render_evaluation_benchmark_rows(payload),
        "",
    ]
    benchmarks = payload.get("benchmarks", [])
    if isinstance(benchmarks, list) and benchmarks:
        lines.extend(["## Benchmark Method Details", ""])
        for benchmark in benchmarks:
            if not isinstance(benchmark, dict):
                continue
            metrics = benchmark.get("metrics", {})
            if not isinstance(metrics, dict):
                continue
            lines.extend(
                [
                    f"### `{benchmark.get('path', '-')}`",
                    "",
                    *_render_evaluation_per_method_rows(metrics),
                    "",
                ],
            )
    return "\n".join(lines).rstrip() + "\n"


def load_report_payload(input_path: Path) -> BenchmarkOutput | UnifiedExtractionOutput | dict[str, Any]:
    """Load one reportable payload from JSON file.

    Args:
        input_path (Path): Input JSON path.

    Raises:
        UnsupportedReportPayloadError: If JSON shape is not report-compatible.

    Returns:
        BenchmarkOutput | UnifiedExtractionOutput | dict[str, Any]: Parsed payload model.

    """
    raw_payload = json.loads(input_path.read_text(encoding="utf-8"))

    if isinstance(raw_payload, dict) and raw_payload.get("command") == "evaluate":
        return raw_payload

    if isinstance(raw_payload, dict) and raw_payload.get("command") == "benchmark":
        return BenchmarkOutput.model_validate(raw_payload)

    if isinstance(raw_payload, dict) and "topics" in raw_payload and "metadata" in raw_payload:
        return UnifiedExtractionOutput.model_validate(raw_payload)

    raise UnsupportedReportPayloadError


def render_report_markdown(
    input_path: Path,
    *,
    title: str | None = None,
) -> str:
    """Render markdown report from one JSON extract, benchmark, or evaluation payload.

    Args:
        input_path (Path): Input extract/benchmark/evaluation JSON file.
        title (str | None): Optional report title override.

    Raises:
        UnsupportedReportPayloadError: If payload command/type is unsupported.

    Returns:
        str: Markdown report content.

    """
    payload = load_report_payload(input_path)
    if isinstance(payload, BenchmarkOutput):
        return render_benchmark_markdown(payload, title=title)
    if isinstance(payload, dict) and payload.get("command") == "evaluate":
        return render_evaluation_markdown(payload, title=title)
    if isinstance(payload, UnifiedExtractionOutput):
        return render_extract_markdown(payload, title=title)

    raise UnsupportedReportPayloadError
