import pytest

from pangolinbench.schema import (
    BenchmarkExample,
    CorpusManifest,
    LexiconItem,
    SchemaValidationError,
)


def test_pangolinbench_example_schema_validates():
    example = BenchmarkExample.from_dict(
        {
            "id": "tw_formal_001",
            "subset": "traditional_zh_formal",
            "text": "本院今日公布新的數位服務規範。",
            "weight": 1.0,
            "tags": ["traditional_zh", "taiwan", "formal"],
            "source": "synthetic",
            "license": "repo-fixture",
            "notes": "Small synthetic fixture for CI.",
        }
    )
    assert example.id == "tw_formal_001"
    assert example.subset == "traditional_zh_formal"
    assert example.text


@pytest.mark.parametrize(
    "data",
    [
        {"subset": "traditional_zh_formal", "text": "本院今日公布新的數位服務規範。"},
        {"id": "x", "text": "本院今日公布新的數位服務規範。"},
        {"id": "x", "subset": "traditional_zh_formal"},
    ],
)
def test_pangolinbench_example_schema_rejects_missing_required_fields(data):
    with pytest.raises(SchemaValidationError):
        BenchmarkExample.from_dict(data)


def test_pangolinbench_example_schema_rejects_empty_text_unless_allowed():
    data = {"id": "empty", "subset": "traditional_zh_formal", "text": ""}
    with pytest.raises(SchemaValidationError):
        BenchmarkExample.from_dict(data)
    assert BenchmarkExample.from_dict(data, allow_empty_text=True).text == ""


def test_pangolinbench_lexicon_item_schema_validates():
    item = LexiconItem.from_dict(
        {
            "term": "健保",
            "category": "health_policy",
            "weight": 10.0,
            "tags": ["taiwan_term", "traditional_zh"],
            "variants": ["全民健康保險"],
        }
    )
    assert item.term == "健保"
    assert item.weight == 10.0


def test_pangolinbench_manifest_schema_validates():
    manifest = CorpusManifest.from_dict(
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
                    "license": "internal",
                }
            ],
            "lexicons": [
                {
                    "path": "data/private/taiwan_terms.tsv",
                    "format": "tsv",
                    "term_field": "term",
                    "category_field": "category",
                    "weight_field": "weight",
                    "license": "internal",
                }
            ],
        }
    )
    assert manifest.name == "internal_taiwan_eval"
    assert manifest.examples[0].text_field == "text"
    assert manifest.lexicons[0].term_field == "term"
