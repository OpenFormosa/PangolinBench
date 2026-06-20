# PangolinBench

## Motivation

PangolinBench evaluates Taiwan-centric tokenizer behavior across Traditional
Chinese, Taiwan Mandarin, Taiwan-specific terms, Bopomofo, and stress-test
strings for Taigi and Hakka romanization. When connected to licensed heldout
corpora, it can be used to support stronger claims about tokenizer suitability
for Taiwan language modeling.

PangolinBench is not a general NLP benchmark. It is a tokenizer evaluation suite
for diagnosing roundtrip correctness, token efficiency, over-segmentation, script
coverage, and structured rich transcription stability.

## Supported subsets

- `traditional_zh_formal`: synthetic formal Traditional Chinese prose covering
  government-like, education-like, medical-like, finance-like, and technical
  styles.
- `taiwan_mandarin_colloquial`: conversational Taiwan Mandarin strings with
  forms such as `欸`, `啦`, `齁`, `蠻`, `超`, `不會吧`, `有差嗎`, and
  `這樣可以嗎`.
- `taiwan_named_entities`: Taiwan-specific place, institution, public service,
  transport, and industry terms without currently serving political figures.
- `taigi_han_roman_mixed`: Taigi / Taiwanese Han-Roman mixed tokenizer
  stress-test strings.
- `hakka_romanized`: Hakka romanization stress-test strings with mixed Han text.
- `bopomofo`: Bopomofo / Zhuyin and mixed Chinese-Zhuyin strings.
- `taiwan_code_switching`: Traditional Chinese, English, numbers, APIs, and
  technical terms in Taiwan software contexts.
- `rich_transcription_json`: compact ASR rich transcription JSON with fixed
  decimal timestamp strings.
- `rich_transcription_structured_text`: an alternative structural debugging
  format using explicit transcript and segment tokens.
- `unicode_edge_cases`: UTF-8 edge cases including punctuation, URLs, dates,
  phone-like strings, email-like strings, emoji, and Latin diacritics.

The Taigi and Hakka built-in strings are tokenizer stress tests. They are not
authoritative linguistic examples.

## Metrics

PangolinBench reports per-example roundtrip correctness, decoded text,
token counts, character counts, UTF-8 byte counts, CJK character counts,
Bopomofo character counts, Latin counts, digit counts, and punctuation counts.
Report summaries also include `vocab_size` when the tokenizer adapter exposes a
vocabulary. Vocabulary size matters because smaller vocabularies reduce
embedding and output-head cost, while larger vocabularies can reduce token count.
Interpret both together.

Compression metrics are reported overall and by subset:

- `tokens_per_char`
- `chars_per_token`
- `tokens_per_utf8_byte`
- `utf8_bytes_per_token`
- `tokens_per_cjk_char`
- `tokens_per_bopomofo_char`
- `tokens_per_latin_char`

Taiwan lexicon metrics report token cost, weighted token cost, percentiles,
over-segmentation rates, and category breakdowns. CJK fertility is a coarse
tokenizer-level diagnostic: `number_of_tokens_for_text / number_of_cjk_characters`.
It is not the same as linguistic word segmentation.

Script coverage uses simple deterministic Unicode classifiers. It does not claim
to perfectly identify language.

## Quality gates

Default quality gates require:

- exact encode/decode roundtrip for all fixtures
- no dense timestamp tokens such as `<|ts_10|>`
- required rich transcription structural tokens
- rich transcription JSON parseability after encode/decode
- fixed-decimal timestamp preservation
- required JSON fields: `Start`, `End`, and `Content`
- no `NaN` or `Infinity` metrics

The CLI exits nonzero when gates fail unless `--no-fail-on-quality-gate` is set.

## Baseline comparison

Pass `--baseline-tokenizer path/to/baseline` to compare a candidate tokenizer
against a baseline. PangolinBench computes relative token reduction by subset,
relative token-per-character change, relative weighted lexicon token cost change,
relative ASR transcript token reduction, and categories where the candidate is
better or worse.

The formula is:

```text
relative_reduction = (baseline_value - candidate_value) / baseline_value
```

Positive values mean the candidate uses fewer tokens. Negative values mean it
uses more tokens. PangolinBench does not fail by default only because a candidate
is worse than a baseline; comparative quality gates can opt into thresholds.

## Report schema

JSON output includes:

```json
{
  "benchmark": "PangolinBench",
  "benchmark_version": "0.1.0",
  "tokenizer_name": "...",
  "baseline_tokenizer_name": null,
  "ok": true,
  "quality_gate_failures": [],
  "summary": {
    "num_examples": 0,
    "roundtrip_ok_rate": 1.0,
    "total_tokens": 0,
    "tokens_per_char": 0.0,
    "tokens_per_utf8_byte": 0.0,
    "pangolin_score": null
  },
  "subset_metrics": {},
  "lexicon_metrics": {},
  "script_metrics": {},
  "rich_transcription_metrics": {
    "rich_transcription_token_ids": {},
    "timestamp_precision_digits": 2,
    "json_roundtrip_ok": true,
    "dense_timestamp_tokens_found": []
  },
  "baseline_comparison": null,
  "examples": [],
  "config": {}
}
```

Per-example details are compact by default. Configure:

```yaml
report:
  include_example_text: false
  max_examples_per_subset: 20
```

## Adding private heldout data

Do not commit private or licensed corpora to the repository. Add a manifest and
point `external_manifest` at it:

```json
{
  "name": "internal_taiwan_eval",
  "version": "0.1.0",
  "description": "Private heldout tokenizer evaluation data.",
  "examples": [
    {
      "path": "data/private/traditional_zh_formal.jsonl",
      "subset": "traditional_zh_formal",
      "format": "jsonl",
      "text_field": "text",
      "license": "internal"
    }
  ],
  "lexicons": [
    {
      "path": "data/private/taiwan_terms.tsv",
      "format": "tsv",
      "term_field": "term",
      "category_field": "category",
      "weight_field": "weight",
      "license": "internal"
    }
  ]
}
```

The default config uses built-in synthetic fixtures and requires no network
access.

## Interpreting results

Roundtrip correctness means the tokenizer can encode and decode the test string
without changing it. It does not prove the tokenizer is efficient for Taiwan
language modeling.

Compression, lexicon token cost, CJK fertility, Bopomofo behavior, romanized
local-language stress tests, and structured ASR stability should be read
together. The optional `pangolin_score` is a convenience metric when a baseline
is provided, not the sole measure of tokenizer quality. Raw metrics are always
reported.

## Known limitations

Built-in PangolinBench data is synthetic and small by design. It is useful for
CI, smoke tests, and deterministic tokenizer diagnostics, but it is not a
replacement for licensed heldout corpora. PangolinBench does not fully evaluate
all Taiwanese languages unless real data for those languages is provided.
