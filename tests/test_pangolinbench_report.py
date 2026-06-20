import json

from pangolinbench.report import json_dumps_report, markdown_summary, validate_report_shape
from pangolinbench.runner import run_pangolinbench
from pangolinbench.tokenizer_adapter import ByteRoundtripTestAdapter


def test_pangolinbench_json_report_serializes_without_nan_or_infinity():
    report = run_pangolinbench(ByteRoundtripTestAdapter(), config_path="configs/pangolinbench.yaml")
    payload = json_dumps_report(report)
    assert "NaN" not in payload
    assert "Infinity" not in payload
    parsed = json.loads(payload)
    assert parsed["benchmark"] == "PangolinBench"
    assert parsed["summary"]["vocab_size"] >= 256


def test_pangolinbench_report_contains_required_top_level_fields():
    report = run_pangolinbench(ByteRoundtripTestAdapter(), config_path="configs/pangolinbench.yaml")
    validate_report_shape(report)
    for field in [
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
    ]:
        assert field in report


def test_pangolinbench_markdown_summary_contains_name():
    report = run_pangolinbench(ByteRoundtripTestAdapter(), config_path="configs/pangolinbench.yaml")
    assert "PangolinBench" in markdown_summary(report)
