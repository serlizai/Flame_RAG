"""需要真实模型、已建语料库和 Redis 的可选链路冒烟测试。"""

import os

import pytest


@pytest.mark.skipif(
    os.getenv("RUN_AGENT_INTEGRATION") != "1",
    reason="Set RUN_AGENT_INTEGRATION=1 to run the configured backend smoke test.",
)
def test_agent_answers_a_document_question():
    from backend.retrieval import agent_invoke

    answer, citations, session_id = agent_invoke(
        "What information is available in the indexed documents?"
    )

    assert isinstance(answer, str) and answer
    assert isinstance(citations, list)
    assert isinstance(session_id, str) and session_id
