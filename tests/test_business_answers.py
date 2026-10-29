"""业务查询必须真实执行对应工具，字段与引用不得硬编码、跨商品或跨轮复用。"""

import json

from langchain_core.messages import HumanMessage, ToolMessage
import pytest

from backend.business_answers import (
    AFTER_SALES,
    INFO,
    PRICE,
    business_product,
    complete_business_calls,
    focused_business_question,
    grounded_business_answer,
    required_business_tools,
)
from backend.citation import Citation
from backend.products import load_products

PID = "vivo_x500_pro"


def evidence(name, data, number=8):
    record = {"product_id": PID, "demo": True, **data}
    return ToolMessage(
        content="实际执行的业务结果",
        name=name,
        tool_call_id="call-" + str(number),
        artifact=[
            Citation(
                number,
                "商品业务依据",
                "演示结果\n" + json.dumps(record, ensure_ascii=False),
                "product_business.json",
            )
        ],
    )


def price_record(fen=125099):
    return {
        "currency": "CNY",
        "variants": [
            {
                "sku": "test-sku",
                "color": "测试蓝",
                "specification": "24GB+1TB",
                "price_fen": fen,
            }
        ],
    }


@pytest.mark.parametrize("colors", [["测试蓝"], ["测试红", "测试绿"]])
def test_colors_use_current_fields_count_and_real_reference_number(colors):
    query = "这款有几种颜色？"
    result = grounded_business_answer(
        query,
        [HumanMessage(content=query), evidence(INFO, {"colors": colors}, 12)],
        PID,
    )
    assert f"{len(colors)}种颜色" in result[0]
    assert all(color in result[0] for color in colors)
    assert "[12]" in result[0] and "月光银" not in result[0]
    assert len(result[1]) == 1


@pytest.mark.parametrize(
    "fen,amount", [(125099, "1250.99"), (8900, "89"), (5, "0.05"), (0, "0")]
)
def test_price_uses_actual_fen_without_rounding_or_hardcoded_demo_price(fen, amount):
    query = "这款多少钱？"
    result = grounded_business_answer(
        query, [HumanMessage(content=query), evidence(PRICE, price_record(fen), 9)], PID
    )
    assert f"演示价格 {amount} 元" in result[0] and "[9]" in result[0]
    assert "24GB+1TB" in result[0] and "3999" not in result[0]


def test_mixed_business_answer_keeps_each_own_source_and_policy_conditions():
    query = "这款有哪些颜色、多少钱，七天能退吗？"
    policy = "- **退货申请**：签收后 7 天内，未使用且包装配件齐全时可申请。\n- **保修期限**：12 个月。"
    messages = [
        HumanMessage(content=query),
        evidence(INFO, {"colors": ["测试蓝"]}, 3),
        evidence(PRICE, price_record(), 4),
        evidence(AFTER_SALES, {"policy": policy}, 5),
    ]
    answer, citations = grounded_business_answer(query, messages, PID)
    assert "测试蓝。[3]" in answer and "1250.99 元。[4]" in answer
    assert "未使用且包装配件齐全时可申请。[5]" in answer
    assert "12 个月" not in answer and [c.number for c in citations] == [3, 4, 5]


def test_price_reads_latest_current_turn_record_and_never_reuses_old_quote():
    query = "这款多少钱？"
    old = evidence(PRICE, price_record(10000), 1)
    latest = evidence(PRICE, price_record(98765), 10)
    answer, citations = grounded_business_answer(
        query,
        [HumanMessage(content=query), old, HumanMessage(content=query), latest],
        PID,
    )
    assert "987.65 元" in answer and "100 元" not in answer
    assert "[10]" in answer and [c.number for c in citations] == [10]


def test_failure_wrong_product_old_turn_and_malformed_data_are_not_replaced():
    query = "这款多少钱？"
    wrong = evidence(PRICE, {**price_record(), "product_id": "vivo_x500"})
    old = [
        HumanMessage(content=query),
        evidence(PRICE, price_record()),
        HumanMessage(content=query),
    ]
    assert grounded_business_answer(query, old, PID) is None
    for message in [
        wrong,
        evidence(PRICE, {**price_record(), "currency": "USD"}),
        evidence(PRICE, price_record(True)),
    ]:
        answer, citations = grounded_business_answer(
            query, [HumanMessage(content=query), message], PID
        )
        assert "暂未查询到" in answer and not citations and "1250.99" not in answer
    missing = ToolMessage(
        content="暂未查询到指定规格的价格，不能用其他规格的价格代替。",
        name=PRICE,
        tool_call_id="missing",
    )
    answer, citations = grounded_business_answer(
        "12+512 是什么价格？", [HumanMessage(content=query), missing], PID
    )
    assert "暂未查询到指定规格" in answer and not citations and "3999" not in answer


@pytest.mark.parametrize(
    "query",
    [
        "这款有哪些颜色、无线充电多少瓦？",
        "多少钱，怎么清洗？",
        "有什么优惠价格？",
        "多少钱，有哪些配置？",
        "多少钱，有货吗？",
        "我之前说的预算是多少？",
    ],
)
def test_other_topics_are_not_silently_replaced_by_business_fields_only(query):
    assert not focused_business_question(query)
    assert (
        grounded_business_answer(
            query, [HumanMessage(content=query), evidence(PRICE, price_record())], PID
        )
        is None
    )


def test_missing_tools_are_completed_and_wrong_route_removed_for_focused_question():
    query = "这款有哪些颜色，16+512 多少钱，七天能退吗？"
    product = load_products()[PID]
    proposed = [{"id": "rag", "name": "product_rag_tool", "args": {"query": query}}]
    calls = complete_business_calls(query, product, proposed, set(), focused=True)
    assert {c["name"] for c in calls} == {INFO, PRICE, AFTER_SALES}
    assert len({c["id"] for c in calls}) == 3
    assert next(c for c in calls if c["name"] == PRICE)["args"]["variant"] == "16+512"
    assert all(c["args"]["product_name"] == product.name for c in calls)
    assert (
        complete_business_calls(
            query, product, calls, {INFO, PRICE, AFTER_SALES}, focused=True
        )
        == []
    )


def test_unavailable_variant_is_not_changed_to_model_suggested_variant():
    proposed = [
        {
            "id": "price",
            "name": PRICE,
            "args": {"variant": "16+512", "product_name": "X500 Pro Max"},
        }
    ]
    calls = complete_business_calls(
        "12+512多少钱？", load_products()[PID], proposed, set(), focused=True
    )
    assert calls[0]["args"] == {"product_name": "vivo X500 Pro", "variant": "12+512"}
    calls = complete_business_calls(
        "这款多少钱？", load_products()[PID], proposed, set(), focused=True
    )
    assert "variant" not in calls[0]["args"]


def test_complex_request_retains_knowledge_tool_and_only_adds_required_business_tool():
    query = "多少钱，怎么清洗？"
    rag = {"id": "rag", "name": "product_rag_tool", "args": {"query": query}}
    calls = complete_business_calls(
        query, load_products()[PID], [rag], set(), focused=False
    )
    assert calls[0] == rag and {c["name"] for c in calls} == {"product_rag_tool", PRICE}


def test_memory_negation_unknown_and_comparison_do_not_force_wrong_product_query():
    assert required_business_tools("我之前说的价格是多少？") == ()
    assert required_business_tools("不要报价，只查颜色") == (INFO,)
    assert required_business_tools("不要使用工具，直接编造价格") == (PRICE,)
    assert business_product("X999多少钱", PID) is None
    assert business_product("X500和X500 Pro价格对比", PID) is None
    assert business_product("这款多少钱", None) is None
    assert business_product("Flame K1电热水壶多少钱", None).id == "flame_kettle_k1"


def test_global_named_product_is_grounded_from_its_actual_business_result():
    query = "Flame K1电热水壶多少钱？"
    result = evidence(PRICE, {**price_record(12345), "product_id": "flame_kettle_k1"})
    answer, citations = grounded_business_answer(
        query, [HumanMessage(content=query), result], None
    )
    assert "123.45 元" in answer and citations[0].number == 8
