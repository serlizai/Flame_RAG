"""运行真实 Streamlit 页面；仅用替身隔离模型与 Redis，验证验收面板行为。"""

from pathlib import Path
import os
import sys
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture
def demo(monkeypatch):
    state = {
        "names": ["price_tool"],
        "cache_hit": False,
        "calls": [],
        "resolve": False,
        "contexts": {},
        "histories": {},
    }

    def invoke(
        query,
        session_id=None,
        product_id=None,
        *,
        trace=None,
        use_cache=True,
        follow_conversation=False,
    ):
        state["calls"].append((query, session_id, product_id, use_cache))
        original = query
        if state["resolve"] and follow_conversation:
            from backend.conversation import resolve_conversation

            result = resolve_conversation(
                query, product_id, state["contexts"].get(session_id, {})
            )
            query, product_id = result.question, result.product_id
            state["contexts"][session_id] = result.context
            trace.update(
                original_question=original,
                resolved_question=query,
                context_source=result.source,
                session_id=session_id,
            )
        count = state["histories"].get((session_id, product_id), 0)
        trace.update(
            history_turns=count,
            steps=["llm_call", "tool_node", "llm_call", "final_answer"],
        )
        state["histories"][(session_id, product_id)] = count + 1
        trace.update(
            cache_hit=state["cache_hit"],
            product_id=product_id,
            tools=[
                {
                    "name": name,
                    "arguments": {"product_id": product_id},
                    "result": "演示业务结果",
                    "sources": ["product_business.json"],
                    "status": "success",
                }
                for name in state["names"]
            ],
        )
        answer = (
            "WEEK7-PRO"
            if state["resolve"] and "备注代号是什么" in query and count > 0
            else "演示回答"
        )
        return answer, [], session_id

    monkeypatch.setitem(
        sys.modules, "backend.retrieval", SimpleNamespace(agent_invoke=invoke)
    )
    monkeypatch.setitem(
        sys.modules,
        "backend.config",
        SimpleNamespace(
            cache=SimpleNamespace(
                get_chat_history=lambda session_id: SimpleNamespace(messages=[])
            )
        ),
    )
    app = AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=15
    ).run()
    assert not app.exception
    return app, state


def select_price_scenario(app):
    app.selectbox(key="validation_scenario").select("3. 商品报价 → 价格工具").run()
    app.button(key="run_validation").click().run()
    assert not app.exception


def test_validation_displays_actual_execution_and_disables_cache(demo):
    app, state = demo
    select_price_scenario(app)
    assert state["calls"][0][2:] == ("vivo_x500_pro", False)
    assert any("符合预期" in item.value for item in app.success)
    assert any("price_tool" in item.label for item in app.expander)
    assert any(item.value == "演示业务结果" for item in app.text)
    assert app.metric[0].value == "1/4"
    original_session = state["calls"][0][1]
    app.button(key="run_validation").click().run()
    assert state["calls"][-1][1] != original_session
    assert len(app.session_state["messages"]) == 4


def test_wrong_route_is_visible_and_not_counted_as_passed(demo):
    app, state = demo
    state["names"] = ["product_rag_tool"]
    select_price_scenario(app)
    assert not app.success
    assert any("不一致" in item.value for item in app.warning)
    assert app.metric[0].value == "0/4"


def test_cache_hit_is_not_counted_as_tool_execution(demo):
    app, state = demo
    state["cache_hit"] = True
    select_price_scenario(app)
    assert not app.success
    assert any("命中缓存" in item.value for item in app.info)
    assert app.metric[0].value == "0/4"


def test_regular_chat_preserves_trace_across_reruns_and_product_switch(demo):
    app, state = demo
    app.chat_input[0].set_value("这款多少钱？").run()
    assert not app.exception and state["calls"][0][3] is True
    app.selectbox(key="selected_product").select("vivo_x500").run()
    app.chat_input[0].set_value("这款多少钱？").run()
    assert not app.exception and state["calls"][-1][2] == "vivo_x500"
    assert len(app.session_state["messages"]) == 4
    assert all(
        m.get("trace", {}).get("tools")
        for m in app.session_state["messages"]
        if m["role"] == "assistant"
    )


def test_week7_demo_keeps_session_and_updates_selected_product(demo):
    app, state = demo
    state["resolve"] = True
    app.button(key="week7_start").click().run()
    assert not app.exception
    session = app.session_state["thread_id"]
    assert app.session_state["week7_demo_step"] == 1
    app.button(key="week7_next").click().run()
    assert not app.exception
    assert app.selectbox(key="selected_product").value == "vivo_x500"
    message = app.session_state["messages"][-1]
    assert "无线充电" in message["trace"]["resolved_question"]
    assert "WEEK7-PRO" not in message["trace"]["resolved_question"]
    app.button(key="week7_next").click().run()
    assert app.selectbox(key="selected_product").value == "vivo_x500_pro_max"
    app.button(key="week7_next").click().run()
    assert not app.exception
    assert app.selectbox(key="selected_product").value == "vivo_x500_pro"
    assert app.session_state["thread_id"] == session
    assert app.session_state["week7_demo_step"] == 4
    assert app.session_state["messages"][-1]["content"] == "WEEK7-PRO"
    assert any("恢复 Pro" in item.value for item in app.success)
    assert any("tool_node" in item.value for item in app.code)
    assert len({item[1] for item in state["calls"]}) == 1


@pytest.mark.skipif(
    os.getenv("RUN_AGENT_INTEGRATION") != "1",
    reason="设置 RUN_AGENT_INTEGRATION=1 后验证真实页面和模型。",
)
def test_real_validation_panel_shows_three_executed_business_tools():
    """通过真实页面按钮调用模型，核对可见徽标与实际图状态一致。"""
    from backend.sessions import make_thread_id, make_thread_config
    from langchain_core.messages import ToolMessage

    app = AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=180
    )
    session = None
    try:
        app.run()
        assert not app.exception
        app.selectbox(key="validation_scenario").select(
            "5. 混合问题 → 三个业务工具"
        ).run()
        app.button(key="run_validation").click().run(timeout=180)
        session = app.session_state["thread_id"]
        assert not app.exception and not app.error
        assert any("符合预期" in item.value for item in app.success)
        assert app.metric[0].value == "3/4"
        trace = app.session_state["messages"][-1]["trace"]
        expected = {"product_info_tool", "price_tool", "after_sales_tool"}
        assert {call["name"] for call in trace["tools"]} == expected
        assert not trace["cache_hit"]
        assert all(
            call["arguments"]["product_id"] == "vivo_x500_pro" and call["result"]
            for call in trace["tools"]
        )
        from backend.retrieval import agent

        state = agent.get_state(make_thread_config(session, "vivo_x500_pro")).values
        assert {
            m.name for m in state["messages"] if isinstance(m, ToolMessage)
        } == expected
    finally:
        if session:
            from backend.config import cache, checkpointer

            cache.get_chat_history(session).clear()
            checkpointer.delete_thread(make_thread_id(session, "vivo_x500_pro"))
            from backend.conversation import conversation_key

            cache.redis_client.delete(conversation_key(session))
