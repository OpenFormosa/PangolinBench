from pangolinbench.fixtures import (
    REQUIRED_TIMESTAMP_STRINGS,
    SUBSETS,
    find_dense_timestamp_tokens,
    get_builtin_examples,
)
from pangolinbench.metrics import count_bopomofo_chars, count_cjk_chars


def test_pangolinbench_builtin_fixtures_load_all_subsets():
    examples = get_builtin_examples()
    subsets = {example.subset for example in examples}
    assert set(SUBSETS).issubset(subsets)


def test_pangolinbench_fixture_ids_are_unique_and_licensed():
    examples = get_builtin_examples()
    ids = [example.id for example in examples]
    assert len(ids) == len(set(ids))
    assert all(example.source for example in examples)
    assert all(example.license for example in examples)


def test_pangolinbench_fixtures_have_no_dense_timestamp_tokens():
    examples = get_builtin_examples()
    assert find_dense_timestamp_tokens([example.text for example in examples]) == []


def test_pangolinbench_rich_fixture_includes_fixed_decimal_timestamps():
    rich_text = "\n".join(example.text for example in get_builtin_examples() if example.subset == "rich_transcription_json")
    for timestamp in REQUIRED_TIMESTAMP_STRINGS:
        assert timestamp in rich_text


def test_pangolinbench_traditional_fixture_contains_traditional_chinese_text():
    formal_text = "\n".join(example.text for example in get_builtin_examples() if example.subset == "traditional_zh_formal")
    assert "數位" in formal_text
    assert count_cjk_chars(formal_text) > 0


def test_pangolinbench_bopomofo_fixture_contains_bopomofo_characters():
    bopomofo_text = "\n".join(example.text for example in get_builtin_examples() if example.subset == "bopomofo")
    assert "ㄅㄆㄇ" in bopomofo_text
    assert count_bopomofo_chars(bopomofo_text) > 0
