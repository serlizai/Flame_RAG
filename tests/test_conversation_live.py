"""通过真实 Streamlit 页面检验第七周四轮上下文；默认跳过外部服务。"""

import json
import os
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


@pytest.mark.skipif(
    os.getenv("RUN_AGENT_INTEGRATION") != "1",
    reason="设置 RUN_AGENT_INTEGRATION=1 后验证真实四轮对话演示。",
)
def test_real_week7_panel_inherits_topic_and_restores_independent_memory():
    from backend.sessions import make_thread_id, make_thread_config
    from backend.conversation import conversation_key

    app = AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=180
    )
    session = None
    expected = ["vivo_x500_pro", "vivo_x500", "vivo_x500_pro_max", "vivo_x500_pro"]
    try:
        app.run()
        assert not app.exception
        for i, product_id in enumerate(expected):
            app.button(key="week7_start" if i == 0 else "week7_next").click().run(
                timeout=180
            )
            session = app.session_state["thread_id"] if session is None else session
            assert not app.exception and not app.error
            assert app.session_state["thread_id"] == session
            assert app.selectbox(key="selected_product").value == product_id
            message = app.session_state["messages"][-1]
            trace = message["trace"]
            assert trace["product_id"] == product_id and not trace["cache_hit"]
            assert (
                trace["steps"][0] == "resolve_context"
                and trace["steps"][-1] == "final_answer"
            )
            assert all(
                call["arguments"]["product_id"] == product_id for call in trace["tools"]
            )
            if i < 3:
                assert trace["tools"] and message["citations"], message["content"]
                if i == 1:
                    assert "不支持" in message["content"] and not re.search(
                        r"50\s*(?:W|瓦)", message["content"], re.I
                    )
                    assert (
                        "无线充电" in trace["resolved_question"]
                        and "WEEK7-PRO" not in trace["resolved_question"]
                    )
                else:
                    assert re.search(r"50\s*(?:W|瓦)", message["content"], re.I), (
                        message["content"]
                    )
                for citation in message["citations"]:
                    if citation.source == "product_business.json":
                        assert (
                            json.loads(citation.snippet.split("\n", 1)[1])["product_id"]
                            == product_id
                        )
                    else:
                        assert (
                            citation.source
                            == {
                                "vivo_x500_pro": "x500_pro.md",
                                "vivo_x500": "x500.md",
                                "vivo_x500_pro_max": "x500_pro_max.md",
                            }[product_id]
                        )
            else:
                assert "WEEK7-PRO" in message["content"] and not message["citations"]
                assert trace["history_turns"] == 1
            print(
                f"第七周真实页面第 {i + 1} 轮通过：{product_id}；流程：{' -> '.join(trace['steps'])}"
            )
        from backend.retrieval import agent

        base = agent.get_state(make_thread_config(session, "vivo_x500")).values
        assert all("WEEK7-PRO" not in message.content for message in base["messages"])
        assert app.session_state["week7_demo_step"] == 4
        assert any("恢复 Pro" in item.value for item in app.success)
    finally:
        if session:
            from backend.config import cache, checkpointer

            cache.get_chat_history(session).clear()
            cache.redis_client.delete(conversation_key(session))
            for product_id in set(expected):
                checkpointer.delete_thread(make_thread_id(session, product_id))
