"""明确参数来自实际工具字段或原文，不硬编码演示答案或跨商品引用。"""

import json

from langchain_core.messages import HumanMessage, ToolMessage
import pytest

from backend.citation import Citation
from backend.grounding import grounded_parameter_answer


def structured_messages(value, product_id="vivo_x500_pro", number=6):
    record = {"product_id": product_id, "specifications": {"无线充电": value}}
    citation = Citation(
        number,
        "商品信息接口",
        "演示数据\n" + json.dumps(record),
        "product_business.json",
    )
    return [
        HumanMessage(content="这款支持无线充电吗？"),
        ToolMessage(
            content="真实工具结果",
            name="product_info_tool",
            tool_call_id="executed",
            artifact=[citation],
        ),
    ]


@pytest.mark.parametrize("value", ["最高 37W，需兼容底座", "最高 73W，需兼容底座"])
def test_supported_parameter_uses_actual_value_and_citation_number(value):
    result = grounded_parameter_answer(
        "这款支持无线充电吗，最高功率是多少？",
        structured_messages(value),
        "vivo_x500_pro",
    )
    assert value in result[0] and "[6]" in result[0]
    assert "50W" not in result[0] and result[1][0].source == "product_business.json"


def test_negative_value_is_known_not_missing_and_has_no_invented_power():
    result = grounded_parameter_answer(
        "这款支持无线充电吗？",
        structured_messages("不支持", "vivo_x500", 9),
        "vivo_x500",
    )
    assert (
        "不支持无线充电" in result[0] and "不适用" in result[0] and "[9]" in result[0]
    )
    assert "暂未" not in result[0] and "50W" not in result[0]


@pytest.mark.parametrize(
    "question",
    [
        "这款支持无线充电吗，多少钱？",
        "这款的无线充电底座如何放置？",
        "我给这款的备注是什么？",
        "这款有哪些卖点？",
    ],
)
def test_mixed_usage_and_memory_questions_are_not_replaced_with_parameter_only_answer(
    question,
):
    assert (
        grounded_parameter_answer(
            question, structured_messages("最高50W"), "vivo_x500_pro"
        )
        is None
    )


def test_wrong_product_missing_or_prior_turn_evidence_is_not_used():
    assert (
        grounded_parameter_answer(
            "无线充电功率是多少？",
            structured_messages("最高50W", "vivo_x500_pro_max"),
            "vivo_x500_pro",
        )
        is None
    )
    assert (
        grounded_parameter_answer(
            "无线充电功率是多少？", structured_messages("未提供"), "vivo_x500_pro"
        )
        is None
    )
    old = structured_messages("最高50W") + [
        HumanMessage(content="这款支持无线充电吗？")
    ]
    assert (
        grounded_parameter_answer("这款支持无线充电吗？", old, "vivo_x500_pro") is None
    )


@pytest.mark.parametrize(
    "text",
    [
        "| 无线充电 | 最高 37W |",
        "### Q4：vivo X500 Pro 支持无线充电吗？\n\n支持最高 37W 无线充电。",
    ],
)
def test_rag_table_or_explicit_faq_is_cited_from_the_executed_document(text):
    citation = Citation(4, "vivo X500 Pro, 核心参数", text, "x500_pro.md")
    messages = [
        HumanMessage(content="无线充电多少瓦？"),
        ToolMessage(
            content=text,
            name="product_rag_tool",
            tool_call_id="rag",
            artifact=[citation],
        ),
    ]
    result = grounded_parameter_answer("无线充电多少瓦？", messages, "vivo_x500_pro")
    assert "37W" in result[0] and result[1] == [citation]


def test_comparison_paragraph_does_not_become_a_fact_for_the_current_product():
    citation = Citation(
        2,
        "vivo X500 Pro, 商品简介",
        "X500 不支持无线充电，Pro 更适合需要无线充电的用户。",
        "x500_pro.md",
    )
    messages = [
        HumanMessage(content="支持无线充电吗？"),
        ToolMessage(
            content=citation.snippet,
            name="product_rag_tool",
            tool_call_id="rag",
            artifact=[citation],
        ),
    ]
    assert (
        grounded_parameter_answer("支持无线充电吗？", messages, "vivo_x500_pro") is None
    )
