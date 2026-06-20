from pangolinbench.runner import run_pangolinbench
from pangolinbench.tokenizer_adapter import ByteRoundtripTestAdapter, ReferenceSegmentationAdapter


class CorruptDecodeAdapter(ByteRoundtripTestAdapter):
    name = "CorruptDecodeAdapter"

    def decode(self, ids, skip_special_tokens=False):
        return super().decode(ids, skip_special_tokens=skip_special_tokens) + "x"


class DenseTimestampVocabAdapter(ByteRoundtripTestAdapter):
    name = "DenseTimestampVocabAdapter"

    def get_vocab(self):
        vocab = super().get_vocab()
        vocab["<|ts_10|>"] = 999_999
        return vocab


class DoubleTokenBaselineAdapter(ByteRoundtripTestAdapter):
    name = "DoubleTokenBaselineAdapter"

    def encode(self, text):
        ids = []
        for value in text.encode("utf-8"):
            ids.extend([value, value])
        return ids

    def decode(self, ids, skip_special_tokens=False):
        return bytes(int(value) for value in ids[::2]).decode("utf-8")


def test_pangolinbench_runner_produces_report():
    report = run_pangolinbench(ByteRoundtripTestAdapter(), config_path="configs/pangolinbench.yaml")
    assert report["benchmark"] == "PangolinBench"
    assert report["ok"] is True
    assert report["summary"]["num_examples"] > 0
    assert report["summary"]["vocab_size"] >= 256


def test_pangolinbench_quality_gates_fail_when_decode_is_corrupt():
    report = run_pangolinbench(CorruptDecodeAdapter(), config_path="configs/pangolinbench.yaml")
    assert report["ok"] is False
    assert report["quality_gate_failures"]


def test_pangolinbench_quality_gates_fail_for_dense_timestamp_vocab():
    report = run_pangolinbench(DenseTimestampVocabAdapter(), config_path="configs/pangolinbench.yaml")
    assert report["ok"] is False
    assert any("dense timestamp" in failure for failure in report["quality_gate_failures"])


def test_pangolinbench_rich_transcription_json_parse_check_passes_after_roundtrip():
    report = run_pangolinbench(ByteRoundtripTestAdapter(), config_path="configs/pangolinbench.yaml")
    assert report["rich_transcription_metrics"]["parsed_json_after_decode_ok"] is True
    assert report["rich_transcription_metrics"]["json_roundtrip_ok"] is True


def test_pangolinbench_baseline_comparison_relative_reduction():
    report = run_pangolinbench(
        ByteRoundtripTestAdapter(),
        config_path="configs/pangolinbench.yaml",
        baseline_adapter=DoubleTokenBaselineAdapter(),
    )
    comparison = report["baseline_comparison"]
    assert comparison is not None
    assert comparison["relative_tokens_per_char_change"] == 0.5
    assert comparison["relative_token_reduction_by_subset"]["traditional_zh_formal"] == 0.5
    assert report["summary"]["pangolin_score"] is not None


def test_pangolinbench_reference_segmentation_adapter_produces_best_line_report():
    report = run_pangolinbench(ReferenceSegmentationAdapter(), config_path="configs/pangolinbench.yaml")
    assert report["benchmark"] == "PangolinBench"
    assert report["summary"]["roundtrip_ok_rate"] == 1.0
    assert report["rich_transcription_metrics"]["rich_transcription_tokens_present"] is True
    assert report["summary"]["total_tokens"] < run_pangolinbench(
        ByteRoundtripTestAdapter(), config_path="configs/pangolinbench.yaml"
    )["summary"]["total_tokens"]
