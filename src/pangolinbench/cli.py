"""Command-line interface for PangolinBench."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .report import write_json_report, write_markdown_summary
from .runner import run_pangolinbench
from .tokenizer_adapter import ByteRoundtripTestAdapter, ReferenceSegmentationAdapter, load_tokenizer_adapter


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PangolinBench tokenizer evaluation.")
    parser.add_argument("--tokenizer", help="Path to tokenizer.json or a directory containing tokenizer.json.")
    parser.add_argument("--baseline-tokenizer", help="Optional baseline tokenizer path for PangolinBench comparison.")
    parser.add_argument("--config", default="configs/pangolinbench.yaml", help="PangolinBench config path.")
    parser.add_argument("--output", default="reports/pangolinbench_report.json", help="JSON report output path.")
    parser.add_argument("--markdown-output", help="Optional Markdown summary output path.")
    parser.add_argument(
        "--byte-roundtrip-test-adapter",
        action="store_true",
        help="Use the PangolinBench byte roundtrip test adapter instead of a production tokenizer.",
    )
    parser.add_argument(
        "--reference-segmentation-adapter",
        action="store_true",
        help="Use the PangolinBench deterministic reference segmentation adapter.",
    )
    parser.add_argument(
        "--no-fail-on-quality-gate",
        action="store_true",
        help="Return exit code 0 even when PangolinBench quality gates fail.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.byte_roundtrip_test_adapter and args.reference_segmentation_adapter:
        parser.error("choose only one of --byte-roundtrip-test-adapter or --reference-segmentation-adapter")
    if args.reference_segmentation_adapter:
        adapter = ReferenceSegmentationAdapter()
    elif args.byte_roundtrip_test_adapter:
        adapter = ByteRoundtripTestAdapter()
    elif args.tokenizer:
        adapter = load_tokenizer_adapter(args.tokenizer)
    else:
        parser.error("--tokenizer is required unless a built-in PangolinBench adapter flag is set")

    baseline_adapter = load_tokenizer_adapter(args.baseline_tokenizer) if args.baseline_tokenizer else None
    report = run_pangolinbench(adapter, config_path=args.config, baseline_adapter=baseline_adapter)
    write_json_report(report, args.output)
    if args.markdown_output:
        write_markdown_summary(report, args.markdown_output)
    status = "passed" if report["ok"] else "failed"
    print(f"PangolinBench {status}: {Path(args.output)}")
    if not report["ok"]:
        for failure in report["quality_gate_failures"]:
            print(f"- {failure}", file=sys.stderr)
    if not report["ok"] and not args.no_fail_on_quality_gate:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
