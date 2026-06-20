"""Metric computation for PangolinBench."""

from __future__ import annotations

from collections import defaultdict
import json
import math
import re
import string
import time
from typing import Any
import unicodedata

from .fixtures import (
    REQUIRED_NON_SPEECH_LABELS,
    REQUIRED_STRUCTURAL_TOKENS,
    REQUIRED_TIMESTAMP_STRINGS,
    find_dense_timestamp_tokens,
)
from .schema import BenchmarkExample, LexiconItem
from .tokenizer_adapter import TokenizerAdapter


DENSE_TIMESTAMP_TOKEN_REGEXES = [
    re.compile(r"<\|ts_[0-9]+\|>"),
    re.compile(r"<\|timestamp_[0-9]+\|>"),
    re.compile(r"<\|time_[0-9]+\|>"),
]

SCRIPT_CATEGORIES = [
    "traditional_han_like_chars",
    "bopomofo_chars",
    "latin_chars",
    "latin_diacritic_chars",
    "digits",
    "ascii_punctuation",
    "cjk_punctuation",
    "emoji_or_symbol_chars",
    "whitespace_chars",
]


def safe_div(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    value = numerator / denominator
    if not math.isfinite(value):
        return None
    return value


def percentile(values: list[int | float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def is_han_like(char: str) -> bool:
    codepoint = ord(char)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x2A6DF
        or 0x2A700 <= codepoint <= 0x2B73F
        or 0x2B740 <= codepoint <= 0x2B81F
        or 0x2B820 <= codepoint <= 0x2CEAF
        or 0x30000 <= codepoint <= 0x3134F
    )


def is_bopomofo(char: str) -> bool:
    codepoint = ord(char)
    return 0x3100 <= codepoint <= 0x312F or 0x31A0 <= codepoint <= 0x31BF


def is_latin(char: str) -> bool:
    if "A" <= char <= "Z" or "a" <= char <= "z":
        return True
    try:
        return unicodedata.name(char).startswith("LATIN") and unicodedata.category(char).startswith("L")
    except ValueError:
        return False


def is_latin_diacritic(char: str) -> bool:
    if not is_latin(char):
        return False
    return ord(char) > 127 or bool(unicodedata.decomposition(char))


def is_cjk_punctuation(char: str) -> bool:
    codepoint = ord(char)
    if 0x3000 <= codepoint <= 0x303F:
        return True
    if 0xFF00 <= codepoint <= 0xFFEF and unicodedata.category(char).startswith("P"):
        return True
    return char in "，。！？、；：「」『』（）《》〈〉【】"


def is_emoji_or_symbol(char: str) -> bool:
    category = unicodedata.category(char)
    return category.startswith("S") and not char.isspace()


def count_character_categories(text: str) -> dict[str, int]:
    counts = {category: 0 for category in SCRIPT_CATEGORIES}
    for char in text:
        if is_han_like(char):
            counts["traditional_han_like_chars"] += 1
        if is_bopomofo(char):
            counts["bopomofo_chars"] += 1
        if is_latin(char):
            counts["latin_chars"] += 1
        if is_latin_diacritic(char):
            counts["latin_diacritic_chars"] += 1
        if char.isdigit():
            counts["digits"] += 1
        if char in string.punctuation:
            counts["ascii_punctuation"] += 1
        if is_cjk_punctuation(char):
            counts["cjk_punctuation"] += 1
        if is_emoji_or_symbol(char):
            counts["emoji_or_symbol_chars"] += 1
        if char.isspace():
            counts["whitespace_chars"] += 1
    return counts


def count_cjk_chars(text: str) -> int:
    return sum(1 for char in text if is_han_like(char))


def count_bopomofo_chars(text: str) -> int:
    return sum(1 for char in text if is_bopomofo(char))


def count_latin_chars(text: str) -> int:
    return sum(1 for char in text if is_latin(char))


def count_digits(text: str) -> int:
    return sum(1 for char in text if char.isdigit())


def count_punctuation(text: str) -> int:
    return sum(1 for char in text if unicodedata.category(char).startswith("P") or char in string.punctuation)


def compute_example_metrics(example: BenchmarkExample, adapter: TokenizerAdapter) -> dict[str, Any]:
    encode_error: str | None = None
    decode_error: str | None = None
    try:
        ids = adapter.encode(example.text)
    except Exception as exc:
        ids = []
        encode_error = str(exc)
    try:
        decoded_text = adapter.decode(ids)
    except Exception as exc:
        decoded_text = ""
        decode_error = str(exc)
    categories = count_character_categories(example.text)
    token_count = len(ids)
    char_count = len(example.text)
    utf8_byte_count = len(example.text.encode("utf-8"))
    result: dict[str, Any] = {
        "id": example.id,
        "subset": example.subset,
        "weight": example.weight,
        "encode_decode_exact_match": decoded_text == example.text and encode_error is None and decode_error is None,
        "decoded_text": decoded_text,
        "token_count": token_count,
        "char_count": char_count,
        "utf8_byte_count": utf8_byte_count,
        "cjk_char_count": categories["traditional_han_like_chars"],
        "bopomofo_char_count": categories["bopomofo_chars"],
        "latin_char_count": categories["latin_chars"],
        "digit_count": categories["digits"],
        "punctuation_count": count_punctuation(example.text),
    }
    result.update(categories)
    result["tokens_per_char"] = safe_div(token_count, char_count)
    result["tokens_per_utf8_byte"] = safe_div(token_count, utf8_byte_count)
    result["tokens_per_cjk_char"] = safe_div(token_count, result["cjk_char_count"])
    result["tokens_per_bopomofo_char"] = safe_div(token_count, result["bopomofo_char_count"])
    result["tokens_per_latin_char"] = safe_div(token_count, result["latin_char_count"])
    if encode_error:
        result["encode_error"] = encode_error
    if decode_error:
        result["decode_error"] = decode_error
    return result


def compute_all_example_metrics(examples: list[BenchmarkExample], adapter: TokenizerAdapter) -> list[dict[str, Any]]:
    return [compute_example_metrics(example, adapter) for example in examples]


def aggregate_example_metrics(example_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    total_tokens = sum(int(item["token_count"]) for item in example_metrics)
    total_chars = sum(int(item["char_count"]) for item in example_metrics)
    total_utf8_bytes = sum(int(item["utf8_byte_count"]) for item in example_metrics)
    total_cjk_chars = sum(int(item["cjk_char_count"]) for item in example_metrics)
    total_bopomofo_chars = sum(int(item["bopomofo_char_count"]) for item in example_metrics)
    total_latin_chars = sum(int(item["latin_char_count"]) for item in example_metrics)
    failures = [item["id"] for item in example_metrics if not item["encode_decode_exact_match"]]
    num_examples = len(example_metrics)
    roundtrip_ok = num_examples - len(failures)
    return {
        "num_examples": num_examples,
        "roundtrip_ok_rate": safe_div(roundtrip_ok, num_examples),
        "roundtrip_failures": failures,
        "total_tokens": total_tokens,
        "total_chars": total_chars,
        "total_utf8_bytes": total_utf8_bytes,
        "total_cjk_chars": total_cjk_chars,
        "total_bopomofo_chars": total_bopomofo_chars,
        "total_latin_chars": total_latin_chars,
        "tokens_per_char": safe_div(total_tokens, total_chars),
        "chars_per_token": safe_div(total_chars, total_tokens),
        "tokens_per_utf8_byte": safe_div(total_tokens, total_utf8_bytes),
        "utf8_bytes_per_token": safe_div(total_utf8_bytes, total_tokens),
        "tokens_per_cjk_char": safe_div(total_tokens, total_cjk_chars),
        "tokens_per_bopomofo_char": safe_div(total_tokens, total_bopomofo_chars),
        "tokens_per_latin_char": safe_div(total_tokens, total_latin_chars),
    }


def aggregate_subset_metrics(example_metrics: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in example_metrics:
        grouped[str(item["subset"])].append(item)
    subset_metrics: dict[str, dict[str, Any]] = {}
    for subset, items in sorted(grouped.items()):
        aggregate = aggregate_example_metrics(items)
        fertilities = [
            safe_div(item["token_count"], item["cjk_char_count"])
            for item in items
            if int(item["cjk_char_count"]) > 0
        ]
        fertility_values = [float(value) for value in fertilities if value is not None]
        weighted_num = sum(float(item["token_count"]) * float(item.get("weight", 1.0)) for item in items if item["cjk_char_count"] > 0)
        weighted_den = sum(float(item["cjk_char_count"]) * float(item.get("weight", 1.0)) for item in items if item["cjk_char_count"] > 0)
        aggregate.update(
            {
                "mean_cjk_char_fertility": safe_div(sum(fertility_values), len(fertility_values)),
                "weighted_cjk_char_fertility": safe_div(weighted_num, weighted_den),
                "p50_cjk_char_fertility": percentile(fertility_values, 0.50),
                "p90_cjk_char_fertility": percentile(fertility_values, 0.90),
            }
        )
        subset_metrics[subset] = aggregate
    return subset_metrics


def compute_lexicon_metrics(lexicon: list[LexiconItem], adapter: TokenizerAdapter) -> dict[str, Any]:
    terms: list[dict[str, Any]] = []
    for item in lexicon:
        ids = adapter.encode(item.term)
        token_count = len(ids)
        char_count = len(item.term)
        utf8_byte_count = len(item.term.encode("utf-8"))
        terms.append(
            {
                "term": item.term,
                "category": item.category,
                "weight": item.weight,
                "token_count": token_count,
                "char_count": char_count,
                "utf8_byte_count": utf8_byte_count,
                "tokens_per_char": safe_div(token_count, char_count),
                "tokens_per_utf8_byte": safe_div(token_count, utf8_byte_count),
                "oversegmented_2": token_count > 2,
                "oversegmented_3": token_count > 3,
                "oversegmented_4": token_count > 4,
            }
        )
    token_counts = [term["token_count"] for term in terms]
    total_weight = sum(float(term["weight"]) for term in terms)
    weighted_sum = sum(float(term["token_count"]) * float(term["weight"]) for term in terms)
    category_breakdown: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for term in terms:
        grouped[str(term.get("category") or "uncategorized")].append(term)
    for category, category_terms in sorted(grouped.items()):
        category_weights = sum(float(term["weight"]) for term in category_terms)
        category_breakdown[category] = {
            "num_terms": len(category_terms),
            "mean_term_token_count": safe_div(sum(int(term["token_count"]) for term in category_terms), len(category_terms)),
            "weighted_mean_term_token_count": safe_div(
                sum(float(term["token_count"]) * float(term["weight"]) for term in category_terms),
                category_weights,
            ),
            "overseg_rate_2": safe_div(sum(1 for term in category_terms if term["oversegmented_2"]), len(category_terms)),
            "overseg_rate_3": safe_div(sum(1 for term in category_terms if term["oversegmented_3"]), len(category_terms)),
            "overseg_rate_4": safe_div(sum(1 for term in category_terms if term["oversegmented_4"]), len(category_terms)),
        }
    return {
        "terms": terms,
        "summary": {
            "num_terms": len(terms),
            "mean_term_token_count": safe_div(sum(token_counts), len(token_counts)),
            "weighted_mean_term_token_count": safe_div(weighted_sum, total_weight),
            "p50_term_token_count": percentile(token_counts, 0.50),
            "p90_term_token_count": percentile(token_counts, 0.90),
            "p95_term_token_count": percentile(token_counts, 0.95),
            "p99_term_token_count": percentile(token_counts, 0.99),
            "overseg_rate_2": safe_div(sum(1 for term in terms if term["oversegmented_2"]), len(terms)),
            "overseg_rate_3": safe_div(sum(1 for term in terms if term["oversegmented_3"]), len(terms)),
            "overseg_rate_4": safe_div(sum(1 for term in terms if term["oversegmented_4"]), len(terms)),
            "category_breakdown": category_breakdown,
        },
    }


def compute_script_metrics(example_metrics: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    script_metrics: dict[str, dict[str, Any]] = {}
    for category in SCRIPT_CATEGORIES:
        char_count = sum(int(item.get(category, 0)) for item in example_metrics)
        containing = [item for item in example_metrics if int(item.get(category, 0)) > 0]
        token_count_when_present = sum(int(item["token_count"]) for item in containing)
        ok_when_present = sum(1 for item in containing if item["encode_decode_exact_match"])
        script_metrics[category] = {
            "char_count": char_count,
            "token_count_for_examples_containing_script": token_count_when_present,
            "tokens_per_char_when_present": safe_div(token_count_when_present, char_count),
            "roundtrip_ok_rate_when_present": safe_div(ok_when_present, len(containing)),
        }
    return script_metrics


def compute_rich_transcription_metrics(
    examples: list[BenchmarkExample],
    adapter: TokenizerAdapter,
    *,
    timestamp_precision_digits: int = 2,
) -> dict[str, Any]:
    rich_examples = [
        example
        for example in examples
        if example.subset in {"rich_transcription_json", "rich_transcription_structured_text"}
    ]
    decoded_by_id: dict[str, str] = {}
    json_roundtrip_ok = True
    parsed_json_after_decode_ok = True
    required_fields_present = True
    for example in rich_examples:
        ids = adapter.encode(example.text)
        decoded = adapter.decode(ids)
        decoded_by_id[example.id] = decoded
        if example.subset == "rich_transcription_json":
            if decoded != example.text:
                json_roundtrip_ok = False
            try:
                parsed = _parse_transcript_segments_json(decoded)
            except Exception:
                parsed_json_after_decode_ok = False
                required_fields_present = False
                continue
            for segment in parsed:
                if not all(field in segment for field in ["Start", "End", "Content"]):
                    required_fields_present = False
    all_decoded_text = "\n".join(decoded_by_id.values())
    timestamp_strings_preserved = all(timestamp in all_decoded_text for timestamp in REQUIRED_TIMESTAMP_STRINGS)
    non_speech_labels_preserved = all(label in all_decoded_text for label in REQUIRED_NON_SPEECH_LABELS)
    try:
        vocab = adapter.get_vocab() if adapter.supports_token_strings() else {}
    except Exception:
        vocab = {}
    dense_timestamp_tokens_found = find_dense_timestamp_tokens(list(vocab.keys()) + [example.text for example in rich_examples])
    rich_transcription_token_ids = {token: adapter.token_to_id(token) for token in REQUIRED_STRUCTURAL_TOKENS}
    missing_structural_tokens = [token for token, token_id in rich_transcription_token_ids.items() if token_id is None]
    return {
        "rich_transcription_token_ids": rich_transcription_token_ids,
        "timestamp_precision_digits": timestamp_precision_digits,
        "json_roundtrip_ok": json_roundtrip_ok,
        "parsed_json_after_decode_ok": parsed_json_after_decode_ok,
        "timestamp_strings_preserved": timestamp_strings_preserved,
        "required_fields_present": required_fields_present,
        "non_speech_labels_preserved": non_speech_labels_preserved,
        "rich_transcription_tokens_present": not missing_structural_tokens,
        "missing_structural_tokens": missing_structural_tokens,
        "dense_timestamp_tokens_found": dense_timestamp_tokens_found,
        "required_timestamp_strings": list(REQUIRED_TIMESTAMP_STRINGS),
        "required_non_speech_labels": list(REQUIRED_NON_SPEECH_LABELS),
        "required_structural_tokens": list(REQUIRED_STRUCTURAL_TOKENS),
        "rich_transcription_total_tokens": sum(len(adapter.encode(example.text)) for example in rich_examples),
        "decoded_examples_checked": sorted(decoded_by_id.keys()),
    }


def _parse_transcript_segments_json(text: str) -> list[dict[str, Any]]:
    try:
        from ubi_tokenizer.transcription.timestamp_format import (  # type: ignore
            parse_transcript_segments_json,
            validate_transcript_segments,
        )
    except Exception:
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise ValueError("rich transcription JSON must decode to a list")
        if not all(isinstance(segment, dict) for segment in parsed):
            raise ValueError("rich transcription JSON segments must be objects")
        return parsed
    parsed = parse_transcript_segments_json(text)
    validate_transcript_segments(parsed)
    return parsed


def compute_utf8_fragmentation_diagnostics(
    examples: list[BenchmarkExample],
    adapter: TokenizerAdapter,
) -> dict[str, Any]:
    span_method = getattr(adapter, "token_byte_spans", None)
    if not callable(span_method):
        return {
            "available": False,
            "reason": "token byte spans are not exposed by this adapter",
        }
    aligned = 0
    total = 0
    incomplete = 0
    cjk_incomplete = 0
    for example in examples:
        ids = adapter.encode(example.text)
        spans = span_method(example.text, ids)
        encoded = example.text.encode("utf-8")
        for start, end in spans:
            total += 1
            token_bytes = encoded[start:end]
            try:
                token_bytes.decode("utf-8")
                aligned += 1
            except UnicodeDecodeError:
                incomplete += 1
                if any(is_han_like(char) for char in example.text):
                    cjk_incomplete += 1
    return {
        "available": True,
        "utf8_boundary_aligned_token_rate": safe_div(aligned, total),
        "incomplete_utf8_token_rate": safe_div(incomplete, total),
        "cjk_incomplete_utf8_token_rate": safe_div(cjk_incomplete, total),
    }


def compute_performance_metrics(
    examples: list[BenchmarkExample],
    adapter: TokenizerAdapter,
    *,
    warmup_rounds: int,
    measured_rounds: int,
) -> dict[str, Any]:
    texts = [example.text for example in examples]
    for _ in range(warmup_rounds):
        for text in texts:
            adapter.encode(text)
    start = time.perf_counter()
    encoded_batches: list[list[int]] = []
    for _ in range(measured_rounds):
        for text in texts:
            encoded_batches.append(adapter.encode(text))
    encode_seconds = time.perf_counter() - start
    decode_start = time.perf_counter()
    for ids in encoded_batches:
        adapter.decode(ids)
    decode_seconds = time.perf_counter() - decode_start
    total_chars = sum(len(text) for text in texts) * measured_rounds
    total_bytes = sum(len(text.encode("utf-8")) for text in texts) * measured_rounds
    total_tokens = sum(len(ids) for ids in encoded_batches)
    return {
        "encode_chars_per_second": safe_div(total_chars, encode_seconds),
        "encode_bytes_per_second": safe_div(total_bytes, encode_seconds),
        "decode_tokens_per_second": safe_div(total_tokens, decode_seconds),
        "benchmark_wall_time_seconds": encode_seconds + decode_seconds,
    }


def contains_nonfinite_number(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(contains_nonfinite_number(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_nonfinite_number(item) for item in value)
    return False
