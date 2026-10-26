"""第六周真实模型路由与商品问答验收；默认跳过付费 API 调用。"""

import os
import re
import uuid

import pytest

from backend.sessions import make_thread_config, make_thread_id


@pytest.mark.skipif(
    os.getenv("RUN_AGENT_INTEGRATION") != "1",
    reason="设置 RUN_AGENT_INTEGRATION=1 后验证真实四工具路由。",
)
@pytest.mark.parametrize(
    "question, product_id, expected, fact",
    [
        ("这款有什么卖点？", "vivo_x500_pro", {"product_rag_tool"}, None),
        (
            "这款有几个颜色，分别是什么？",
            "vivo_x500_pro",
            {"product_info_tool"},
            "月光银",
        ),
        ("16+512 是什么价格？", "vivo_x500_pro", {"price_tool"}, "3999"),
        ("七天能退吗，有什么条件？", "vivo_x500_pro", {"after_sales_tool"}, "7"),
        ("12+512 是什么价格？", "vivo_x500_pro", {"price_tool"}, "暂未查询到"),
        ("Flame K1 电热水壶多少钱？", None, {"price_tool"}, "129"),
        (
            "这款有哪些颜色，16+512 多少钱，七天能退吗？",
            "vivo_x500_pro",
            {"product_info_tool", "price_tool", "after_sales_tool"},
            "3999",
        ),
    ],
)
def test_real_agent_chooses_business_tools(question, product_id, expected, fact):
    from langchain_core.messages import ToolMessage
    from backend.config import cache, checkpointer
    from backend.retrieval import agent, agent_invoke

    session = "week6-" + uuid.uuid4().hex
    query = (
        question + f"（验收编号 {session}，无需重复编号；请依据工具结果回答并附引用。）"
    )
    try:
        answer, citations, returned = agent_invoke(query, session, product_id)
        state = agent.get_state(make_thread_config(session, product_id)).values
        results = [m for m in state["messages"] if isinstance(m, ToolMessage)]
        names = {m.name for m in results}
        assert returned == session and expected == names, names
        if fact:
            assert fact in answer, answer
        if "12+512" in question:
            assert not citations and "3999" not in answer, answer
        else:
            assert citations and {int(n) for n in re.findall(r"\[(\d+)\]", answer)} == {
                c.number for c in citations
            }
            assert all(
                c.source
                == (
                    "x500_pro.md"
                    if expected == {"product_rag_tool"}
                    else "product_business.json"
                )
                for c in citations
            )
        if expected != {"product_rag_tool"}:
            assert cache.get(cache.make_cache_key(query, product_id)) is None
        print(f"四工具验收通过：{question} -> {sorted(names)}")
    finally:
        cache.redis_client.delete(cache.make_cache_key(query, product_id))
        cache.get_chat_history(session).clear()
        checkpointer.delete_thread(make_thread_id(session, product_id))
