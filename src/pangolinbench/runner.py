"""Runner for PangolinBench."""

from __future__ import annotations

from collections import defaultdict
import copy
from pathlib import Path
from typing import Any

from .fixtures import (
    SUBSETS,
    get_builtin_examples,
    get_builtin_lexicon,
    load_external_examples_from_manifest,
    load_external_lexicon_from_manifest,
)
from .metrics import (
    aggregate_example_metrics,
    aggregate_subset_metrics,
    compute_all_example_metrics,
    compute_lexicon_metrics,
    compute_performance_metrics,
    compute_rich_transcription_metrics,
    compute_script_metrics,
    compute_utf8_fragmentation_diagnostics,
    contains_nonfinite_number,
    safe_div,
)
from .report import validate_report_shape
from .schema import BenchmarkExample, LexiconItem, load_manifest, load_mapping_file
from .tokenizer_adapter import TokenizerAdapter


BENCHMARK_NAME = "PangolinBench"
BENCHMARK_VERSION = "0.1.0"

DEFAULT_CONFIG: dict[str, Any] = {
    "name": BENCHMARK_NAME,
    "version": BENCHMARK_VERSION,
    "use_builtin_fixtures": True,
    "external_manifest": None,
    "subsets": {subset: {"enabled": True, "weight": 1.0} for subset in SUBSETS},
    "rich_transcription": {
        "timestamp_precision_digits": 2,
        "require_structural_tokens": True,
        "no_dense_timestamp_tokens": True,
    },
    "quality_gates": {
        "require_roundtrip_ok": True,
        "require_no_dense_timestamp_tokens": True,
        "require_rich_transcription_tokens": True,
        "require_rich_transcription_json_parse": True,
        "require_timestamp_precision": True,
        "require_required_json_fields": True,
        "max_roundtrip_failure_rate": 0.0,
        "max_dense_timestamp_tokens": 0,
        "fail_on_nan_metrics": True,
    },
    "comparative_quality_gates": {
        "enabled": False,
        "max_traditional_zh_token_regression": 0.02,
        "max_taiwan_lexicon_cost_regression": 0.02,
        "max_rich_transcription_token_regression": 0.02,
    },
    "performance": {
        "enabled": False,
        "warmup_rounds": 2,
        "measured_rounds": 5,
    },
    "report": {
        "include_example_text": False,
        "max_examples_per_subset": 20,
    },
}


def load_pangolinbench_config(path: str | Path | None = None) -> tuple[dict[str, Any], Path | None]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    base_dir: Path | None = None
    if path is not None:
        path = Path(path)
        base_dir = path.parent
        loaded = load_mapping_file(path)
        loaded_inner = loaded.get("pangolinbench", loaded)
        if not isinstance(loaded_inner, dict):
            raise ValueError("PangolinBench config must be a mapping")
        _deep_merge(config, loaded_inner)
    return config, base_dir


def run_pangolinbench(
    adapter: TokenizerAdapter,
    *,
    config_path: str | Path | None = None,
    config: dict[str, Any] | None = None,
    baseline_adapter: TokenizerAdapter | None = None,
) -> dict[str, Any]:
    if config is None:
        normalized_config, config_base_dir = load_pangolinbench_config(config_path)
    else:
        normalized_config = copy.deepcopy(DEFAULT_CONFIG)
        incoming = config.get("pangolinbench", config)
        _deep_merge(normalized_config, incoming)
        config_base_dir = Path(config_path).parent if config_path is not None else None

    examples, lexicon = _load_inputs(normalized_config, config_base_dir=config_base_dir)
    candidate = _evaluate_adapter(adapter, examples, lexicon, normalized_config)
    baseline = _evaluate_adapter(baseline_adapter, examples, lexicon, normalized_config) if baseline_adapter else None

    baseline_comparison = compute_baseline_comparison(candidate, baseline) if baseline else None
    pangolin_score = compute_pangolin_score(candidate, baseline, baseline_comparison) if baseline_comparison else None
    candidate["summary"]["pangolin_score"] = pangolin_score
    if baseline_comparison and pangolin_score is not None:
        baseline_comparison["pangolin_score"] = pangolin_score

    report = {
        "benchmark": BENCHMARK_NAME,
        "benchmark_version": normalized_config.get("version", BENCHMARK_VERSION),
        "tokenizer_name": adapter.name,
        "baseline_tokenizer_name": baseline_adapter.name if baseline_adapter else None,
        "ok": True,
        "quality_gate_failures": [],
        "summary": candidate["summary"],
        "subset_metrics": candidate["subset_metrics"],
        "lexicon_metrics": candidate["lexicon_metrics"],
        "script_metrics": candidate["script_metrics"],
        "rich_transcription_metrics": candidate["rich_transcription_metrics"],
        "utf8_fragmentation_diagnostics": candidate["utf8_fragmentation_diagnostics"],
        "performance_metrics": candidate.get("performance_metrics"),
        "baseline_comparison": baseline_comparison,
        "examples": _compact_examples(
            examples,
            candidate["example_metrics"],
            include_text=bool(normalized_config.get("report", {}).get("include_example_text", False)),
            max_examples_per_subset=int(normalized_config.get("report", {}).get("max_examples_per_subset", 20)),
        ),
        "config": {"pangolinbench": normalized_config},
    }
    failures = apply_quality_gates(report, normalized_config)
    report["quality_gate_failures"] = failures
    report["ok"] = not failures
    validate_report_shape(report)
    return report


def _evaluate_adapter(
    adapter: TokenizerAdapter | None,
    examples: list[BenchmarkExample],
    lexicon: list[LexiconItem],
    config: dict[str, Any],
) -> dict[str, Any]:
    if adapter is None:
        raise ValueError("PangolinBench requires a tokenizer adapter")
    example_metrics = compute_all_example_metrics(examples, adapter)
    summary = aggregate_example_metrics(example_metrics)
    summary["pangolin_score"] = None
    summary["vocab_size"] = _adapter_vocab_size(adapter)
    subset_metrics = aggregate_subset_metrics(example_metrics)
    lexicon_metrics = compute_lexicon_metrics(lexicon, adapter)
    script_metrics = compute_script_metrics(example_metrics)
    rich_config = config.get("rich_transcription", {})
    rich_transcription_metrics = compute_rich_transcription_metrics(
        examples,
        adapter,
        timestamp_precision_digits=int(rich_config.get("timestamp_precision_digits", 2)),
    )
    utf8_fragmentation_diagnostics = compute_utf8_fragmentation_diagnostics(examples, adapter)
    result: dict[str, Any] = {
        "summary": summary,
        "subset_metrics": subset_metrics,
        "lexicon_metrics": lexicon_metrics,
        "script_metrics": script_metrics,
        "rich_transcription_metrics": rich_transcription_metrics,
        "utf8_fragmentation_diagnostics": utf8_fragmentation_diagnostics,
        "example_metrics": example_metrics,
    }
    performance = config.get("performance", {})
    if performance.get("enabled", False):
        result["performance_metrics"] = compute_performance_metrics(
            examples,
            adapter,
            warmup_rounds=int(performance.get("warmup_rounds", 2)),
            measured_rounds=int(performance.get("measured_rounds", 5)),
        )
    else:
        result["performance_metrics"] = None
    return result


def _adapter_vocab_size(adapter: TokenizerAdapter) -> int | None:
    try:
        return len(adapter.get_vocab())
    except Exception:
        return None


def _load_inputs(
    config: dict[str, Any],
    *,
    config_base_dir: Path | None,
) -> tuple[list[BenchmarkExample], list[LexiconItem]]:
    enabled_subsets = {
        subset
        for subset, subset_config in config.get("subsets", {}).items()
        if bool(subset_config.get("enabled", True))
    }
    examples: list[BenchmarkExample] = []
    lexicon: list[LexiconItem] = []
    if config.get("use_builtin_fixtures", True):
        examples.extend(get_builtin_examples(enabled_subsets))
        lexicon.extend(get_builtin_lexicon())
    manifest_path = config.get("external_manifest")
    if manifest_path:
        path = Path(manifest_path)
        if not path.is_absolute() and config_base_dir is not None:
            path = config_base_dir / path
        manifest = load_manifest(path)
        external_examples = load_external_examples_from_manifest(manifest, base_dir=path.parent)
        external_lexicon = load_external_lexicon_from_manifest(manifest, base_dir=path.parent)
        if enabled_subsets:
            external_examples = [example for example in external_examples if example.subset in enabled_subsets]
        examples.extend(external_examples)
        lexicon.extend(external_lexicon)
    if not examples:
        raise ValueError("PangolinBench has no enabled examples")
    return examples, lexicon


def compute_baseline_comparison(candidate: dict[str, Any], baseline: dict[str, Any] | None) -> dict[str, Any] | None:
    if baseline is None:
        return None
    candidate_subset = candidate["subset_metrics"]
    baseline_subset = baseline["subset_metrics"]
    subset_reductions: dict[str, float | None] = {}
    for subset in sorted(set(candidate_subset) | set(baseline_subset)):
        candidate_tokens = candidate_subset.get(subset, {}).get("total_tokens", 0)
        baseline_tokens = baseline_subset.get(subset, {}).get("total_tokens", 0)
        subset_reductions[subset] = _relative_reduction(baseline_tokens, candidate_tokens)
    candidate_lex_cost = candidate["lexicon_metrics"]["summary"].get("weighted_mean_term_token_count")
    baseline_lex_cost = baseline["lexicon_metrics"]["summary"].get("weighted_mean_term_token_count")
    candidate_rich_tokens = _rich_subset_token_count(candidate_subset)
    baseline_rich_tokens = _rich_subset_token_count(baseline_subset)
    better = [subset for subset, value in subset_reductions.items() if value is not None and value > 0]
    worse = [subset for subset, value in subset_reductions.items() if value is not None and value < 0]
    return {
        "relative_token_reduction_by_subset": subset_reductions,
        "relative_tokens_per_char_change": _relative_reduction(
            baseline["summary"].get("tokens_per_char"),
            candidate["summary"].get("tokens_per_char"),
        ),
        "relative_weighted_term_token_cost_change": _relative_reduction(baseline_lex_cost, candidate_lex_cost),
        "relative_asr_transcript_token_reduction": _relative_reduction(baseline_rich_tokens, candidate_rich_tokens),
        "categories_where_candidate_is_worse": worse,
        "categories_where_candidate_is_better": better,
    }


def compute_pangolin_score(
    candidate: dict[str, Any],
    baseline: dict[str, Any] | None,
    comparison: dict[str, Any] | None,
) -> float | None:
    if baseline is None or comparison is None:
        return None
    subset_reductions = comparison.get("relative_token_reduction_by_subset", {})
    components = {
        "traditional_zh_compression": (0.25, subset_reductions.get("traditional_zh_formal")),
        "taiwan_lexicon_weighted_token_cost": (0.20, comparison.get("relative_weighted_term_token_cost_change")),
        "taiwan_mandarin_colloquial_compression": (0.15, subset_reductions.get("taiwan_mandarin_colloquial")),
        "taigi_han_roman_mixed_compression": (0.15, subset_reductions.get("taigi_han_roman_mixed")),
        "hakka_romanized_compression": (0.10, subset_reductions.get("hakka_romanized")),
        "bopomofo_compression": (0.05, subset_reductions.get("bopomofo")),
        "rich_transcription_stability_and_compression": (0.10, comparison.get("relative_asr_transcript_token_reduction")),
    }
    weighted_score = 0.0
    component_scores: dict[str, float] = {}
    for name, (weight, relative_change) in components.items():
        component_score = _relative_change_to_component_score(relative_change)
        component_scores[name] = component_score
        weighted_score += weight * component_score
    score = weighted_score * 100.0
    rich = candidate["rich_transcription_metrics"]
    summary = candidate["summary"]
    penalties = 0.0
    if summary.get("roundtrip_ok_rate") != 1.0:
        penalties += 35.0
    if rich.get("dense_timestamp_tokens_found"):
        penalties += 35.0
    if not rich.get("parsed_json_after_decode_ok"):
        penalties += 35.0
    if not rich.get("timestamp_strings_preserved"):
        penalties += 35.0
    if not rich.get("rich_transcription_tokens_present"):
        penalties += 25.0
    score = max(0.0, min(100.0, score - penalties))
    candidate["summary"]["pangolin_score_components"] = component_scores
    candidate["summary"]["pangolin_score_penalties"] = penalties
    return score


def apply_quality_gates(report: dict[str, Any], config: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    gates = config.get("quality_gates", {})
    summary = report.get("summary", {})
    rich = report.get("rich_transcription_metrics", {})
    roundtrip_ok_rate = summary.get("roundtrip_ok_rate")
    if gates.get("require_roundtrip_ok", True) and roundtrip_ok_rate != 1.0:
        failures.append(f"roundtrip_ok_rate is {roundtrip_ok_rate}, expected 1.0")
    failure_rate = None if roundtrip_ok_rate is None else 1.0 - float(roundtrip_ok_rate)
    max_failure_rate = float(gates.get("max_roundtrip_failure_rate", 0.0))
    if failure_rate is not None and failure_rate > max_failure_rate:
        failures.append(f"roundtrip failure rate {failure_rate:.6f} exceeds {max_failure_rate:.6f}")
    dense_tokens = rich.get("dense_timestamp_tokens_found") or []
    max_dense = int(gates.get("max_dense_timestamp_tokens", 0))
    if gates.get("require_no_dense_timestamp_tokens", True) and len(dense_tokens) > max_dense:
        failures.append(f"dense timestamp tokens found: {dense_tokens}")
    if gates.get("require_rich_transcription_tokens", True) and not rich.get("rich_transcription_tokens_present"):
        failures.append(f"missing rich transcription structural tokens: {rich.get('missing_structural_tokens')}")
    if gates.get("require_rich_transcription_json_parse", True) and not rich.get("parsed_json_after_decode_ok"):
        failures.append("rich transcription JSON could not be parsed after encode/decode")
    if gates.get("require_timestamp_precision", True) and not rich.get("timestamp_strings_preserved"):
        failures.append("required fixed-decimal timestamp strings were not preserved")
    if gates.get("require_required_json_fields", True) and not rich.get("required_fields_present"):
        failures.append("decoded rich transcription JSON is missing Start, End, or Content")
    if gates.get("fail_on_nan_metrics", True) and contains_nonfinite_number(report):
        failures.append("report contains NaN or Infinity metrics")
    comparative = config.get("comparative_quality_gates", {})
    comparison = report.get("baseline_comparison")
    if comparative.get("enabled", False) and comparison:
        subset_reductions = comparison.get("relative_token_reduction_by_subset", {})
        _comparative_gate(
            failures,
            "traditional_zh_formal",
            subset_reductions.get("traditional_zh_formal"),
            float(comparative.get("max_traditional_zh_token_regression", 0.02)),
        )
        _comparative_gate(
            failures,
            "taiwan_lexicon",
            comparison.get("relative_weighted_term_token_cost_change"),
            float(comparative.get("max_taiwan_lexicon_cost_regression", 0.02)),
        )
        _comparative_gate(
            failures,
            "rich_transcription",
            comparison.get("relative_asr_transcript_token_reduction"),
            float(comparative.get("max_rich_transcription_token_regression", 0.02)),
        )
    return failures


def _compact_examples(
    examples: list[BenchmarkExample],
    example_metrics: list[dict[str, Any]],
    *,
    include_text: bool,
    max_examples_per_subset: int,
) -> list[dict[str, Any]]:
    by_id = {example.id: example for example in examples}
    seen: dict[str, int] = defaultdict(int)
    compact: list[dict[str, Any]] = []
    for metric in example_metrics:
        subset = str(metric["subset"])
        if seen[subset] >= max_examples_per_subset:
            continue
        seen[subset] += 1
        item = {
            "id": metric["id"],
            "subset": subset,
            "encode_decode_exact_match": metric["encode_decode_exact_match"],
            "token_count": metric["token_count"],
            "char_count": metric["char_count"],
            "utf8_byte_count": metric["utf8_byte_count"],
            "tokens_per_char": metric["tokens_per_char"],
            "tokens_per_utf8_byte": metric["tokens_per_utf8_byte"],
        }
        if include_text:
            example = by_id[str(metric["id"])]
            item["text"] = example.text
            item["decoded_text"] = metric["decoded_text"]
        compact.append(item)
    return compact


def _rich_subset_token_count(subset_metrics: dict[str, dict[str, Any]]) -> int:
    return sum(
        int(subset_metrics.get(subset, {}).get("total_tokens", 0))
        for subset in ["rich_transcription_json", "rich_transcription_structured_text"]
    )


def _relative_reduction(baseline_value: Any, candidate_value: Any) -> float | None:
    if baseline_value in (None, 0):
        return None
    if candidate_value is None:
        return None
    return safe_div(float(baseline_value) - float(candidate_value), float(baseline_value))


def _relative_change_to_component_score(relative_change: float | None) -> float:
    if relative_change is None:
        return 0.5
    return max(0.0, min(1.0, 0.5 + float(relative_change)))


def _comparative_gate(failures: list[str], name: str, relative_reduction: Any, max_regression: float) -> None:
    if relative_reduction is None:
        return
    if float(relative_reduction) < -max_regression:
        failures.append(f"{name} token regression {-float(relative_reduction):.6f} exceeds {max_regression:.6f}")


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
