"""Report helpers for PangolinBench."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


REQUIRED_TOP_LEVEL_FIELDS = [
    "benchmark",
    "benchmark_version",
    "tokenizer_name",
    "baseline_tokenizer_name",
    "ok",
    "quality_gate_failures",
    "summary",
    "subset_metrics",
    "lexicon_metrics",
    "script_metrics",
    "rich_transcription_metrics",
    "baseline_comparison",
    "examples",
    "config",
]


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    return value


def json_dumps_report(report: dict[str, Any], *, indent: int = 2) -> str:
    sanitized = sanitize_for_json(report)
    return json.dumps(sanitized, ensure_ascii=False, indent=indent, allow_nan=False, sort_keys=True)


def write_json_report(report: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_dumps_report(report) + "\n", encoding="utf-8")


def markdown_summary(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# PangolinBench Summary",
        "",
        f"- tokenizer: {report.get('tokenizer_name')}",
        f"- ok: {report.get('ok')}",
        f"- examples: {summary.get('num_examples')}",
        f"- roundtrip_ok_rate: {summary.get('roundtrip_ok_rate')}",
        f"- total_tokens: {summary.get('total_tokens')}",
        f"- tokens_per_char: {summary.get('tokens_per_char')}",
        f"- pangolin_score: {summary.get('pangolin_score')}",
    ]
    failures = report.get("quality_gate_failures") or []
    if failures:
        lines.append("")
        lines.append("## Quality Gate Failures")
        for failure in failures:
            lines.append(f"- {failure}")
    return "\n".join(lines) + "\n"


def write_markdown_summary(report: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown_summary(report), encoding="utf-8")


def validate_report_shape(report: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_TOP_LEVEL_FIELDS if field not in report]
    if missing:
        raise ValueError(f"PangolinBench report missing top-level fields: {missing}")
    json_dumps_report(report)
