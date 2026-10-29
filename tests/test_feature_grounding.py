"""卖点引用兜底只能使用本轮当前商品的明确章节，不掩盖混合问题或编造来源。"""

from langchain_core.messages import HumanMessage, ToolMessage
import pytest

from backend.citation import Citation
from backend.grounding import grounded_feature_excerpt


def feature_messages(question="这款有什么卖点？"):
    text = (
        "## 核心卖点\n- 测试特性原文，不是预置手机参数。\n\n## 售后说明\n不得带入此段。"
    )
    citation = Citation(17, "vivo X500 Pro, 核心卖点", text, "x500_pro.md")
    return [
        HumanMessage(content=question),
        ToolMessage(
            content=text,
            name="product_rag_tool",
            tool_call_id="features",
            artifact=[citation],
        ),
    ]


def test_features_use_own_section_verbatim_with_the_real_source_number():
    messages = feature_messages()
    answer, citations = grounded_feature_excerpt(
        "这款有什么卖点？", messages, "vivo_x500_pro"
    )
    assert "测试特性原文，不是预置手机参数。" in answer and "[17]" in answer
    assert "售后说明" not in answer and "不得带入" not in answer
    assert citations == messages[1].artifact


def test_wrong_product_prior_turn_and_non_feature_section_are_not_used():
    query = "这款有什么卖点？"
    messages = feature_messages()
    assert grounded_feature_excerpt(query, messages, "vivo_x500") is None
    assert (
        grounded_feature_excerpt(
            query, messages + [HumanMessage(content=query)], "vivo_x500_pro"
        )
        is None
    )
    wrong = Citation(1, "vivo X500 Pro, FAQ", "FAQ 提到了卖点", "x500_pro.md")
    messages[1] = messages[1].model_copy(update={"artifact": [wrong]})
    assert grounded_feature_excerpt(query, messages, "vivo_x500_pro") is None


@pytest.mark.parametrize(
    "query",
    [
        "有什么卖点，多少钱？",
        "有哪些特点，支持无线充电吗？",
        "有什么优势，怎么用？",
        "我刚才说的备注是什么？",
    ],
)
def test_fallback_does_not_drop_mixed_questions_or_create_memory_citations(query):
    assert (
        grounded_feature_excerpt(query, feature_messages(query), "vivo_x500_pro")
        is None
    )
