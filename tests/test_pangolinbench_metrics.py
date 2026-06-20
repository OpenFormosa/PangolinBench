from pangolinbench.fixtures import DENSE_TIMESTAMP_TOKEN_PATTERNS
from pangolinbench.metrics import (
    compute_all_example_metrics,
    compute_lexicon_metrics,
    compute_rich_transcription_metrics,
    count_bopomofo_chars,
    count_cjk_chars,
    find_dense_timestamp_tokens,
    safe_div,
)
from pangolinbench.schema import BenchmarkExample, LexiconItem
from pangolinbench.tokenizer_adapter import ByteRoundtripTestAdapter, ReferenceSegmentationAdapter


def test_pangolinbench_token_count_metrics_with_byte_adapter():
    adapter = ByteRoundtripTestAdapter()
    example = BenchmarkExample(id="x", subset="traditional_zh_formal", text="台灣ABC")
    metrics = compute_all_example_metrics([example], adapter)[0]
    assert metrics["encode_decode_exact_match"] is True
    assert metrics["token_count"] == len("台灣ABC".encode("utf-8"))
    assert metrics["char_count"] == len("台灣ABC")


def test_pangolinbench_safe_division_returns_null_instead_of_nan():
    assert safe_div(1, 0) is None
    assert safe_div(4, 2) == 2.0


def test_pangolinbench_cjk_and_bopomofo_counting_work():
    assert count_cjk_chars("台灣ABCㄅ") == 2
    assert count_bopomofo_chars("台灣ABCㄅㄆㄇ") == 3


def test_pangolinbench_lexicon_oversegmentation_and_weighted_cost():
    adapter = ByteRoundtripTestAdapter()
    metrics = compute_lexicon_metrics(
        [
            LexiconItem(term="健保", category="health_policy", weight=10.0),
            LexiconItem(term="A", category="latin", weight=1.0),
        ],
        adapter,
    )
    terms = {item["term"]: item for item in metrics["terms"]}
    assert terms["健保"]["oversegmented_4"] is True
    assert metrics["summary"]["weighted_mean_term_token_count"] > metrics["summary"]["mean_term_token_count"]


def test_pangolinbench_timestamp_strings_are_detected():
    adapter = ByteRoundtripTestAdapter()
    rich = compute_rich_transcription_metrics(
        [
            BenchmarkExample(
                id="rich",
                subset="rich_transcription_json",
                text='[{"Start":0.00,"End":3.42,"Speaker":0,"Content":"測試 10.25 3575.50 [Silence] [Noise] [Music] [Unintelligible Speech]"}]',
            )
        ],
        adapter,
    )
    assert rich["timestamp_strings_preserved"] is True


def test_pangolinbench_dense_timestamp_regexes_detect_forbidden_tokens():
    assert DENSE_TIMESTAMP_TOKEN_PATTERNS
    assert find_dense_timestamp_tokens(["<|ts_10|>", "<|timestamp_20|>", "<|time_30|>"]) == [
        "<|time_30|>",
        "<|timestamp_20|>",
        "<|ts_10|>",
    ]


def test_pangolinbench_dense_timestamp_regexes_do_not_flag_structural_tokens():
    assert find_dense_timestamp_tokens(["<|start_time|>", "<|end_time|>", "<|duration|>"]) == []


def test_pangolinbench_reference_segmentation_adapter_roundtrips_and_groups_terms():
    adapter = ReferenceSegmentationAdapter()
    text = "健保與台灣 JSON schema ㄅㄆㄇ 0.00 tshit-á。"
    ids = adapter.encode(text)
    tokens = adapter.convert_ids_to_tokens(ids)
    assert adapter.decode(ids) == text
    assert "健保" in tokens
    assert "台灣" in tokens
    assert "JSON" in tokens
    assert "ㄅㄆㄇ" in tokens
    assert "0.00" in tokens
    assert "tshit-á" in tokens
