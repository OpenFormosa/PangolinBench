"""Synthetic, license-clean PangolinBench fixtures."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re
from typing import Any

from .schema import BenchmarkExample, CorpusManifest, LexiconItem


DENSE_TIMESTAMP_TOKEN_PATTERNS = [
    r"<\|ts_[0-9]+\|>",
    r"<\|timestamp_[0-9]+\|>",
    r"<\|time_[0-9]+\|>",
]

REQUIRED_STRUCTURAL_TOKENS = [
    "<|transcript_start|>",
    "<|transcript_end|>",
    "<|segment_start|>",
    "<|segment_end|>",
    "<|speaker|>",
    "<|start_time|>",
    "<|end_time|>",
    "<|duration|>",
    "<|content|>",
    "<|non_speech_event|>",
]

REQUIRED_TIMESTAMP_STRINGS = ["0.00", "3.42", "10.25", "3575.50"]
REQUIRED_NON_SPEECH_LABELS = ["[Silence]", "[Noise]", "[Music]", "[Unintelligible Speech]"]

SUBSETS = [
    "traditional_zh_formal",
    "taiwan_mandarin_colloquial",
    "taiwan_named_entities",
    "taigi_han_roman_mixed",
    "hakka_romanized",
    "bopomofo",
    "taiwan_code_switching",
    "rich_transcription_json",
    "rich_transcription_structured_text",
    "unicode_edge_cases",
]


def _format_transcript_segments_json(segments: list[dict[str, Any]]) -> str:
    try:
        from ubi_tokenizer.transcription.timestamp_format import format_transcript_segments_json  # type: ignore
    except Exception:
        return _internal_format_transcript_segments_json(segments)
    return format_transcript_segments_json(segments)


def _internal_format_transcript_segments_json(segments: list[dict[str, Any]]) -> str:
    """Internal fallback used only when the repository has no canonical timestamp module."""

    items: list[str] = []
    for segment in segments:
        fields: list[str] = [
            f'"Start":{float(segment["Start"]):.2f}',
            f'"End":{float(segment["End"]):.2f}',
        ]
        speaker = segment.get("Speaker")
        speaker_text = "null" if speaker is None else str(int(speaker))
        fields.append(f'"Speaker":{speaker_text}')
        fields.append(f'"Content":{json.dumps(str(segment["Content"]), ensure_ascii=False)}')
        if "Type" in segment:
            fields.append(f'"Type":{json.dumps(str(segment["Type"]), ensure_ascii=False)}')
        items.append("{" + ",".join(fields) + "}")
    return "[" + ",".join(items) + "]"


RICH_TRANSCRIPTION_SEGMENTS = [
    {"Start": 0.00, "End": 3.42, "Speaker": 0, "Content": "大家好，歡迎回來。"},
    {"Start": 3.50, "End": 7.10, "Speaker": 1, "Content": "今天我們要討論台灣本土語音模型。"},
    {"Start": 10.25, "End": 12.00, "Speaker": 0, "Content": "注音 ㄅㄆㄇ 與台語 tshit-á 混合測試。"},
    {"Start": 12.10, "End": 14.00, "Speaker": None, "Content": "[Silence]", "Type": "non_speech"},
    {"Start": 3575.50, "End": 3578.25, "Speaker": 2, "Content": "長會議時間戳測試。"},
]

RICH_TRANSCRIPTION_JSON_TEXT = _format_transcript_segments_json(RICH_TRANSCRIPTION_SEGMENTS)

RICH_TRANSCRIPTION_NON_SPEECH_JSON_TEXT = _format_transcript_segments_json(
    [
        {"Start": 15.00, "End": 15.50, "Speaker": None, "Content": "[Noise]", "Type": "non_speech"},
        {"Start": 16.00, "End": 16.75, "Speaker": None, "Content": "[Music]", "Type": "non_speech"},
        {
            "Start": 17.00,
            "End": 17.50,
            "Speaker": None,
            "Content": "[Unintelligible Speech]",
            "Type": "non_speech",
        },
    ]
)

RICH_TRANSCRIPTION_STRUCTURED_TEXT = """<|transcript_start|>
<|segment_start|><|speaker|>0<|start_time|>0.00<|end_time|>3.42<|content|>大家好，歡迎回來。<|segment_end|>
<|segment_start|><|speaker|>1<|start_time|>3.50<|end_time|>7.10<|content|>今天我們要討論台灣本土語音模型。<|segment_end|>
<|transcript_end|>"""


def _example(
    example_id: str,
    subset: str,
    text: str,
    tags: list[str],
    *,
    weight: float = 1.0,
    notes: str = "Small synthetic fixture for CI.",
) -> BenchmarkExample:
    return BenchmarkExample(
        id=example_id,
        subset=subset,
        text=text,
        weight=weight,
        tags=tags,
        source="synthetic",
        license="repo-fixture",
        notes=notes,
    )


BUILTIN_EXAMPLES: list[BenchmarkExample] = [
    _example("tw_formal_001", "traditional_zh_formal", "本院今日公布新的數位服務規範。", ["traditional_zh", "taiwan", "formal"]),
    _example("tw_formal_002", "traditional_zh_formal", "校務系統更新後，學生可查詢課程與繳費紀錄。", ["traditional_zh", "education"]),
    _example("tw_formal_003", "traditional_zh_formal", "門診流程調整為先報到、再量測、最後完成諮詢。", ["traditional_zh", "medical"]),
    _example("tw_formal_004", "traditional_zh_formal", "金融服務測試資料包含匯款、帳戶與風險控管欄位。", ["traditional_zh", "finance"]),
    _example("tw_formal_005", "traditional_zh_formal", "技術文件描述模型訓練、資料驗證與版本控管。", ["traditional_zh", "technical"]),
    _example("tw_colloquial_001", "taiwan_mandarin_colloquial", "欸，這樣可以嗎？我覺得蠻順的啦。", ["taiwan_mandarin", "colloquial"]),
    _example("tw_colloquial_002", "taiwan_mandarin_colloquial", "不會吧，這個結果超怪，有差嗎？", ["taiwan_mandarin", "colloquial"]),
    _example("tw_colloquial_003", "taiwan_mandarin_colloquial", "你先看一下齁，等一下再改也可以啦。", ["taiwan_mandarin", "colloquial"]),
    _example(
        "tw_entities_001",
        "taiwan_named_entities",
        "地名詞表：台北、臺北、新北、桃園、新竹、台中、臺中、台南、臺南、高雄。",
        ["taiwan_term", "named_entity"],
    ),
    _example(
        "tw_entities_002",
        "taiwan_named_entities",
        "離島與東部欄位：花蓮、台東、臺東、澎湖、金門、馬祖。",
        ["taiwan_term", "named_entity"],
    ),
    _example(
        "tw_entities_003",
        "taiwan_named_entities",
        "制度與生活詞：健保、勞保、勞保年金、戶政事務所、里長、捷運。",
        ["taiwan_term", "institutional"],
    ),
    _example(
        "tw_entities_004",
        "taiwan_named_entities",
        "產業詞測試：科學園區、晶圓代工、半導體。",
        ["taiwan_term", "industry"],
    ),
    _example("taigi_mix_001", "taigi_han_roman_mixed", "台語 Tâi-gí 與漢羅混寫 stress test。", ["taigi", "han_roman"]),
    _example("taigi_mix_002", "taigi_han_roman_mixed", "lí hó，guá 想測試母語 bô tshit-á 的切分。", ["taigi", "romanization"]),
    _example("taigi_mix_003", "taigi_han_roman_mixed", "lán 用台語、母語、漢羅混寫做 tokenizer 壓力測試。", ["taigi", "han_roman"]),
    _example("hakka_roman_001", "hakka_romanized", "Hak-kâ romanization stress test: ngai, ng, 客語。", ["hakka", "romanization"]),
    _example("hakka_roman_002", "hakka_romanized", "客庄詞表：四縣、海陸、大埔、饒平、詔安。", ["hakka", "han"]),
    _example("hakka_roman_003", "hakka_romanized", "Hak-kâ 與客語 mixed text，含 hyphen、apostrophe、tone-like 2 digits。", ["hakka", "mixed"]),
    _example("bopomofo_001", "bopomofo", "ㄅㄆㄇ 基礎注音測試。", ["bopomofo"]),
    _example("bopomofo_002", "bopomofo", "ㄓㄨˋㄧㄣ 與 ㄧㄚˋ 的混合測試。", ["bopomofo", "zhuyin"]),
    _example("bopomofo_003", "bopomofo", "中文與注音 ㄅㄆㄇ 混合測試。", ["bopomofo", "mixed"]),
    _example("tw_code_001", "taiwan_code_switching", "今天 demo API，這個 PR 先不要 merge。", ["code_switching", "technical"]),
    _example("tw_code_002", "taiwan_code_switching", "latency 下降 15%，GPU 訓練成本需要記錄。", ["code_switching", "numbers"]),
    _example("tw_code_003", "taiwan_code_switching", "JSON schema 與 ASR transcript 都要保留格式。", ["code_switching", "structured"]),
    _example("rich_json_001", "rich_transcription_json", RICH_TRANSCRIPTION_JSON_TEXT, ["rich_transcription", "json"]),
    _example("rich_json_002", "rich_transcription_json", RICH_TRANSCRIPTION_NON_SPEECH_JSON_TEXT, ["rich_transcription", "json", "non_speech"]),
    _example("rich_text_001", "rich_transcription_structured_text", RICH_TRANSCRIPTION_STRUCTURED_TEXT, ["rich_transcription", "structured_text"]),
    _example("unicode_edge_001", "unicode_edge_cases", "emoji ✅，全形標點：ＡＢＣ，中文、頓號與句號。", ["unicode", "punctuation"]),
    _example("unicode_edge_002", "unicode_edge_cases", "URL https://example.test/台灣?q=Tâi-gí，日期 2026-06-09。", ["unicode", "url", "latin_diacritic"]),
    _example("unicode_edge_003", "unicode_edge_cases", "電話 02-2345-6789，email test+tw@example.test，括號[]{}「」/colon: ok。", ["unicode", "structured"]),
]

BUILTIN_LEXICON: list[LexiconItem] = [
    LexiconItem(term="健保", category="health_policy", weight=10.0, tags=["taiwan_term", "traditional_zh"], variants=["全民健康保險"]),
    LexiconItem(term="勞保", category="labor_policy", weight=8.0, tags=["taiwan_term", "traditional_zh"]),
    LexiconItem(term="勞保年金", category="labor_policy", weight=8.0, tags=["taiwan_term", "traditional_zh"]),
    LexiconItem(term="戶政事務所", category="institution", weight=7.0, tags=["taiwan_term", "traditional_zh"]),
    LexiconItem(term="里長", category="local_governance", weight=5.0, tags=["taiwan_term", "traditional_zh"]),
    LexiconItem(term="捷運", category="transport", weight=6.0, tags=["taiwan_term", "traditional_zh"]),
    LexiconItem(term="科學園區", category="industry", weight=6.0, tags=["taiwan_term", "traditional_zh"]),
    LexiconItem(term="晶圓代工", category="industry", weight=6.0, tags=["taiwan_term", "traditional_zh"]),
    LexiconItem(term="半導體", category="industry", weight=6.0, tags=["taiwan_term", "traditional_zh"]),
    LexiconItem(term="台北", category="place", weight=4.0, tags=["taiwan_term", "place"], variants=["臺北"]),
    LexiconItem(term="臺南", category="place", weight=4.0, tags=["taiwan_term", "place"], variants=["台南"]),
    LexiconItem(term="Tâi-gí", category="taigi_romanization", weight=4.0, tags=["taigi", "romanization"]),
    LexiconItem(term="tshit-á", category="taigi_romanization", weight=4.0, tags=["taigi", "romanization"]),
    LexiconItem(term="Hak-kâ", category="hakka_romanization", weight=4.0, tags=["hakka", "romanization"]),
    LexiconItem(term="ㄅㄆㄇ", category="bopomofo", weight=4.0, tags=["bopomofo"]),
]


def get_builtin_examples(enabled_subsets: set[str] | None = None) -> list[BenchmarkExample]:
    examples = [example for example in BUILTIN_EXAMPLES if enabled_subsets is None or example.subset in enabled_subsets]
    validate_builtin_fixtures(examples)
    return list(examples)


def get_builtin_lexicon() -> list[LexiconItem]:
    return list(BUILTIN_LEXICON)


def dense_timestamp_regexes() -> list[re.Pattern[str]]:
    return [re.compile(pattern) for pattern in DENSE_TIMESTAMP_TOKEN_PATTERNS]


def find_dense_timestamp_tokens(strings: list[str] | tuple[str, ...]) -> list[str]:
    found: set[str] = set()
    regexes = dense_timestamp_regexes()
    for value in strings:
        for regex in regexes:
            found.update(regex.findall(value))
    return sorted(found)


def validate_builtin_fixtures(examples: list[BenchmarkExample] | None = None) -> None:
    examples = BUILTIN_EXAMPLES if examples is None else examples
    ids: set[str] = set()
    for example in examples:
        if example.id in ids:
            raise ValueError(f"Duplicate PangolinBench fixture id: {example.id}")
        ids.add(example.id)
        if example.source is None or example.license is None:
            raise ValueError(f"PangolinBench fixture {example.id} must include source and license")
        dense = find_dense_timestamp_tokens([example.text])
        if dense:
            raise ValueError(f"PangolinBench fixture {example.id} contains dense timestamp tokens: {dense}")


def load_external_examples_from_manifest(manifest: CorpusManifest, *, base_dir: str | Path) -> list[BenchmarkExample]:
    base = Path(base_dir)
    examples: list[BenchmarkExample] = []
    for spec in manifest.examples:
        path = _resolve_manifest_path(base, spec.path)
        fmt = spec.format.lower()
        if fmt == "jsonl":
            rows = _read_jsonl(path)
        elif fmt == "json":
            loaded = json.loads(path.read_text(encoding="utf-8"))
            rows = loaded if isinstance(loaded, list) else loaded.get("examples", [])
        elif fmt in {"txt", "text"}:
            rows = [{"text": line.rstrip("\n")} for line in path.read_text(encoding="utf-8").splitlines() if line.rstrip("\n")]
        else:
            raise ValueError(f"Unsupported PangolinBench example manifest format: {spec.format}")
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"External PangolinBench example row must be an object: {path}:{index + 1}")
            text = row.get(spec.text_field)
            data = {
                "id": row.get("id", f"{spec.subset}_{path.stem}_{index + 1:05d}"),
                "subset": row.get("subset", spec.subset),
                "text": text,
                "weight": row.get("weight", 1.0),
                "tags": row.get("tags", []),
                "source": row.get("source", f"manifest:{manifest.name}"),
                "license": row.get("license", spec.license),
                "notes": row.get("notes"),
            }
            examples.append(BenchmarkExample.from_dict(data))
    return examples


def load_external_lexicon_from_manifest(manifest: CorpusManifest, *, base_dir: str | Path) -> list[LexiconItem]:
    base = Path(base_dir)
    items: list[LexiconItem] = []
    for spec in manifest.lexicons:
        path = _resolve_manifest_path(base, spec.path)
        fmt = spec.format.lower()
        if fmt == "jsonl":
            rows = _read_jsonl(path)
        elif fmt == "json":
            loaded = json.loads(path.read_text(encoding="utf-8"))
            rows = loaded if isinstance(loaded, list) else loaded.get("lexicons", [])
        elif fmt == "tsv":
            rows = _read_delimited(path, "\t")
        elif fmt == "csv":
            rows = _read_delimited(path, ",")
        else:
            raise ValueError(f"Unsupported PangolinBench lexicon manifest format: {spec.format}")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"External PangolinBench lexicon row must be an object: {path}")
            weight_value = row.get(spec.weight_field) if spec.weight_field else None
            data = {
                "term": row.get(spec.term_field),
                "category": row.get(spec.category_field) if spec.category_field else None,
                "weight": float(weight_value) if weight_value not in (None, "") else 1.0,
                "tags": row.get("tags", []),
                "variants": row.get("variants", []),
                "notes": row.get("notes"),
            }
            if isinstance(data["tags"], str):
                data["tags"] = [tag.strip() for tag in data["tags"].split(",") if tag.strip()]
            if isinstance(data["variants"], str):
                data["variants"] = [variant.strip() for variant in data["variants"].split(",") if variant.strip()]
            items.append(LexiconItem.from_dict(data))
    return items


def _resolve_manifest_path(base_dir: Path, path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
        rows.append(row)
    return rows


def _read_delimited(path: Path, delimiter: str) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))
