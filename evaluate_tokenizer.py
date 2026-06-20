"""Compatibility entry point for tokenizer evaluation.

This empty-bootstrap repository has no pre-existing evaluator. The script keeps
the requested PangolinBench flags available without inventing unrelated behavior.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from pangolinbench.cli import main as pangolinbench_main


def main() -> int:
    parser = argparse.ArgumentParser(description="Tokenizer evaluation entry point.")
    parser.add_argument("--tokenizer")
    parser.add_argument("--baseline-tokenizer")
    parser.add_argument("--pangolinbench", action="store_true", help="Run PangolinBench.")
    parser.add_argument("--pangolinbench-config", default="configs/pangolinbench.yaml")
    parser.add_argument("--output", default="reports/evaluation_report.json")
    parser.add_argument("--byte-roundtrip-test-adapter", action="store_true")
    parser.add_argument("--no-fail-on-quality-gate", action="store_true")
    args = parser.parse_args()
    if not args.pangolinbench:
        parser.error("Only PangolinBench evaluation is available in this bootstrap repository")
    forwarded = [
        "--config",
        args.pangolinbench_config,
        "--output",
        args.output,
    ]
    if args.tokenizer:
        forwarded.extend(["--tokenizer", args.tokenizer])
    if args.baseline_tokenizer:
        forwarded.extend(["--baseline-tokenizer", args.baseline_tokenizer])
    if args.byte_roundtrip_test_adapter:
        forwarded.append("--byte-roundtrip-test-adapter")
    if args.no_fail_on_quality_gate:
        forwarded.append("--no-fail-on-quality-gate")
    return pangolinbench_main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
