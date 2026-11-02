"""商品指代与话题承接使用当前目录，缺少依据时澄清，不跨商品传递事实。"""

import json
from types import SimpleNamespace

import pytest

from backend.conversation import resolve_conversation, load_context, conversation_key


def first_context():
    return resolve_conversation(
        "请记住备注 NOTE-PRO。X500 Pro 支持无线充电吗，最高功率是多少？",
        "vivo_x500_pro",
        {},
    ).context


@pytest.mark.parametrize(
    "question,expected",
    [
        ("那普通版呢？", "vivo_x500"),
        ("标准版呢", "vivo_x500"),
        ("Pro Max 呢？", "vivo_x500_pro_max"),
        ("Pro 呢？", "vivo_x500_pro"),
    ],
)
def test_family_followup_keeps_topic_without_copying_notes(question, expected):
    result = resolve_conversation(question, None, first_context())
    assert result.product_id == expected and result.clarification is None
    assert "无线充电" in result.question and "NOTE-PRO" not in result.question
    assert "NOTE-PRO" not in json.dumps(result.context)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("X500 Pro Max", "vivo_x500_pro_max"),
        ("vivo X500 Pro", "vivo_x500_pro"),
        ("X500", "vivo_x500"),
        ("K1", "flame_kettle_k1"),
    ],
)
def test_explicit_names_choose_longest_match(name, expected):
    assert resolve_conversation(name + " 多少钱？", None, {}).product_id == expected


def test_price_variant_is_carried_without_inventing_an_available_sku():
    first = resolve_conversation("X500 Pro 的16+512多少钱？", None, {})
    second = resolve_conversation("那普通版呢？", None, first.context)
    assert second.product_id == "vivo_x500"
    assert "16+512" in second.question and "12+256" not in second.question


def test_personal_note_does_not_erase_the_prior_price_configuration():
    first = resolve_conversation("X500Pro16+512多少钱？", None, {})
    assert first.product_id == "vivo_x500_pro"
    note = resolve_conversation("请记住备注 NOTE-PRO。", "vivo_x500_pro", first.context)
    followup = resolve_conversation("那普通版呢？", None, note.context)
    assert "16+512" in followup.question and "NOTE-PRO" not in followup.question


@pytest.mark.parametrize(
    "question,selected",
    [
        ("那普通版呢？", None),
        ("那普通版呢？", "vivo_x500_pro"),
        ("它呢？", None),
        ("那普通版呢？", "flame_kettle_k1"),
        ("X600 多少钱？", "vivo_x500_pro"),
    ],
)
def test_missing_history_or_unknown_family_does_not_guess(question, selected):
    assert resolve_conversation(question, selected, {}).clarification


def test_remembered_budget_is_not_rewritten_as_a_product_fact():
    result = resolve_conversation(
        "回到 X500 Pro：我给这款商品的备注代号是什么？",
        "vivo_x500_pro_max",
        first_context(),
    )
    assert result.product_id == "vivo_x500_pro" and "备注" in result.question
    assert "无线充电" not in result.question


def test_user_changed_product_selection_takes_precedence_over_remembered_product():
    result = resolve_conversation("这款多少钱？", "flame_kettle_k1", first_context())
    assert result.product_id == "flame_kettle_k1"


def test_comparison_of_explicit_products_keeps_global_search():
    result = resolve_conversation("X500 Pro 和 X500 Pro Max 有什么区别？", None, {})
    assert result.product_id is None and result.source == "comparison"


@pytest.mark.parametrize("value", ["not-json", "[]", '{"topics":null}'])
def test_invalid_context_is_safe(value):
    result = load_context(SimpleNamespace(get=lambda key: value), "session")
    assert not result or result["topics"] == []
    assert conversation_key("one") != conversation_key("two")
