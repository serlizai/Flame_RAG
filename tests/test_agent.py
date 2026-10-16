"""需要真实模型、已建语料库和 Redis 的可选链路冒烟测试。"""

import os
import re
import uuid

import pytest

from backend.sessions import make_thread_config, make_thread_id


@pytest.mark.skipif(
    os.getenv("RUN_AGENT_INTEGRATION") != "1",
    reason="设置 RUN_AGENT_INTEGRATION=1 后运行生活用品真实 HTTP 验收。",
)
def test_api_answers_household_product_questions():
    """经真实 FastAPI 路由调用 Agent，验证跨品类身份、引用及聊天记录。"""
    from fastapi.testclient import TestClient
    from langchain_core.messages import ToolMessage

    from backend.config import cache, checkpointer
    from backend.main import app
    from backend.retrieval import agent

    identifier = uuid.uuid4().hex
    session_id = f"flame-household-test-{identifier}"
    cases = [
        (
            "flame_kettle_k1",
            "kettle_k1.md",
            "这款的容量和额定功率是多少？",
            ("1.7", "1800"),
        ),
        (
            "flame_thermos_t1",
            "thermos_t1.md",
            "这款的容量和内胆材质是什么？",
            ("500", "316L"),
        ),
        (
            "flame_storage_box_s1",
            "storage_box_s1.md",
            "这款的容量和单箱额定载重是多少？",
            ("45", "10"),
        ),
    ]
    queries = []
    try:
        with TestClient(app) as client:
            for product_id, source, question, facts in cases:
                query = f"{question}请查资料并附引用。（验收编号 {identifier}，无需重复编号。）"
                queries.append((query, product_id))
                response = client.post(
                    "/query",
                    json={
                        "session_id": session_id,
                        "product_id": product_id,
                        "message": query,
                    },
                )
                assert response.status_code == 200, response.text
                payload = response.json()
                assert payload["product_id"] == product_id
                assert payload["session_id"] == session_id
                assert all(fact in payload["answer"] for fact in facts)
                assert payload["citations"]
                assert {citation["source"] for citation in payload["citations"]} == {
                    source
                }
                messages = agent.get_state(
                    make_thread_config(session_id, product_id)
                ).values["messages"]
                results = [
                    message for message in messages if isinstance(message, ToolMessage)
                ]
                assert results and all(
                    message.name == "retrieve_product_docs" for message in results
                )
                print(
                    f"真实 HTTP 生活用品验收通过：{product_id}；回答：{payload['answer']}"
                )
        assert len(cache.get_chat_history(session_id).messages) == 2 * len(cases)
    finally:
        for query, product_id in queries:
            cache.redis_client.delete(cache.make_cache_key(query, product_id))
            checkpointer.delete_thread(make_thread_id(session_id, product_id))
        cache.get_chat_history(session_id).clear()


@pytest.mark.skipif(
    os.getenv("RUN_AGENT_INTEGRATION") != "1",
    reason="设置 RUN_AGENT_INTEGRATION=1 后运行真实 Agent 链路测试。",
)
def test_agent_answers_a_document_question():
    """验证真实工具调用、商品回答、引用和 Redis 历史，并清理本次测试数据。"""
    from langchain_core.messages import ToolMessage

    from backend.config import cache, checkpointer, product_vector_store
    from backend.retrieval import agent, agent_invoke

    # 唯一问题防止命中旧答案缓存；独立会话防止混入其他用户的聊天历史。
    identifier = uuid.uuid4().hex
    session_id = f"flame-agent-test-{identifier}"
    query = (
        "请根据商品知识库说明 vivo X500 Pro 是否支持无线充电、最高功率是多少，"
        f"并附上来源引用。（本次验收编号 {identifier}，无需在回答中重复。）"
    )
    config = make_thread_config(session_id)

    try:
        answer, citations, returned_session_id = agent_invoke(query, session_id)

        assert isinstance(answer, str) and answer
        assert returned_session_id == session_id
        assert "50" in answer
        assert citations, "商品回答必须返回资料引用。"
        assert any(c.source == "x500_pro.md" for c in citations)

        # 从真实图状态确认工具已经执行，不能仅凭非空回答认定链路通过。
        state = agent.get_state(config).values
        assert state["product_id"] is None
        tool_messages = [
            message for message in state["messages"] if isinstance(message, ToolMessage)
        ]
        assert any(
            message.name == "retrieve_product_docs" and message.artifact
            for message in tool_messages
        ), "真实 Agent 必须执行商品检索工具。"

        # 引用编号须与回答一致，原文须来自当前真实商品库。
        numbers = {int(number) for number in re.findall(r"\[(\d+)\]", answer)}
        assert numbers == {citation.number for citation in citations}
        documents = set(product_vector_store.get(include=["documents"])["documents"])
        assert all(citation.snippet in documents for citation in citations)
        assert len(cache.get_chat_history(session_id).messages) == 2
        assert cache.get(cache.make_cache_key(query)) is not None
        print(
            f"真实 Agent 验收通过：工具调用 {len(tool_messages)} 次，"
            f"返回 {len(citations)} 条引用；回答：{answer}"
        )
    finally:
        # 只删除本次随机问题的缓存和随机会话的历史。
        cache.redis_client.delete(cache.make_cache_key(query))
        cache.get_chat_history(session_id).clear()
        checkpointer.delete_thread(make_thread_id(session_id))


@pytest.mark.skipif(
    os.getenv("RUN_AGENT_INTEGRATION") != "1",
    reason="设置 RUN_AGENT_INTEGRATION=1 后运行真实商品状态测试。",
)
def test_agent_keeps_product_state_on_cache_and_switch():
    """验证真实问答、切换商品和缓存分支都把当前商品保存到检查点。"""
    from backend.config import cache, checkpointer
    from backend.retrieval import agent, agent_invoke

    identifier = uuid.uuid4().hex
    session_id = f"flame-state-test-{identifier}"
    cached_session_id = f"flame-state-cached-{identifier}"
    pro_query = (
        "请根据商品资料说明 vivo X500 Pro 是否支持无线充电、最高功率是多少，并附引用。"
        f"（验收编号 {identifier}，无需重复编号。）"
    )
    base_query = (
        "请根据商品资料说明 vivo X500 标准版是否支持无线充电，并附引用。"
        f"（验收编号 {identifier}，无需重复编号。）"
    )

    def state_for(session_id, product_id="vivo_x500_pro"):
        return agent.get_state(make_thread_config(session_id, product_id)).values

    try:
        answer, citations, _ = agent_invoke(pro_query, session_id, "vivo_x500_pro")
        assert "50" in answer and citations
        assert state_for(session_id)["product_id"] == "vivo_x500_pro"

        # 同商品答案跨会话命中时，新会话也必须获得自己的状态和消息。
        cached_answer, cached_citations, _ = agent_invoke(
            pro_query, cached_session_id, "vivo_x500_pro"
        )
        cached_state = state_for(cached_session_id)
        assert cached_answer == answer and cached_citations == citations
        assert cached_state["product_id"] == "vivo_x500_pro"
        assert len(cached_state["messages"]) == 2
        assert cached_state["llm_calls"] == 0
        original_pro = state_for(session_id)

        agent_invoke(base_query, session_id, "vivo_x500")
        assert state_for(session_id, "vivo_x500")["product_id"] == "vivo_x500"
        assert state_for(session_id)["messages"] == original_pro["messages"]

        # 返回已有 Pro 对话时跳过共享缓存，保留原上下文并继续执行模型。
        agent_invoke(pro_query, session_id, "vivo_x500_pro")
        assert state_for(session_id)["product_id"] == "vivo_x500_pro"
        assert state_for(session_id)["llm_calls"] > original_pro["llm_calls"]
        assert (
            state_for(session_id)["messages"][: len(original_pro["messages"])]
            == original_pro["messages"]
        )
        assert len(cache.get_chat_history(session_id).messages) == 6
        assert len(cache.get_chat_history(cached_session_id).messages) == 2
        print(
            "真实 State 验收通过：Pro → 普通版 → 继续 Pro 原上下文；"
            "新会话缓存命中保存 2 条消息，模型调用计数为 0。"
        )
    finally:
        # 只清理本次测试的两个问题缓存和两个随机会话。
        cache.redis_client.delete(cache.make_cache_key(pro_query, "vivo_x500_pro"))
        cache.redis_client.delete(cache.make_cache_key(base_query, "vivo_x500"))
        for thread_id in (session_id, cached_session_id):
            cache.get_chat_history(thread_id).clear()
            for product_id in ("vivo_x500_pro", "vivo_x500"):
                checkpointer.delete_thread(make_thread_id(thread_id, product_id))


@pytest.mark.skipif(
    os.getenv("RUN_AGENT_INTEGRATION") != "1",
    reason="设置 RUN_AGENT_INTEGRATION=1 后运行真实商品过滤验收。",
)
def test_agent_filters_current_product_for_implicit_question():
    """不在问题中写型号，验证真实检索、切换商品、引用和缓存范围。"""
    from langchain_core.messages import ToolMessage

    from backend.config import cache, checkpointer, product_vector_store
    from backend.retrieval import agent, agent_invoke

    identifier = uuid.uuid4().hex
    session_id = f"flame-filter-test-{identifier}"
    query = (
        "这款支持无线充电吗？支持的话最高功率是多少？请查询商品资料并附引用。"
        f"（验收编号 {identifier}，无需重复编号。）"
    )
    products = (
        ("vivo_x500_pro", "x500_pro.md"),
        ("vivo_x500", "x500.md"),
        ("vivo_x500_pro_max", "x500_pro_max.md"),
    )
    first_state = None

    try:
        for product_id, source in products:
            answer, citations, returned_session = agent_invoke(
                query, session_id, product_id
            )
            assert returned_session == session_id
            assert citations and {c.source for c in citations} == {source}
            if product_id == "vivo_x500":
                assert "不支持" in answer
                assert "50W" not in answer and "50 W" not in answer
            else:
                assert "50" in answer

            # 查看实际工具结果，而不仅检查最终回答中的商品名称。
            state = agent.get_state(make_thread_config(session_id, product_id)).values
            active_messages = state["messages"]
            tool_messages = [m for m in active_messages if isinstance(m, ToolMessage)]
            assert tool_messages and all(
                m.name == "retrieve_product_docs" for m in tool_messages
            )
            for message in tool_messages:
                assert message.artifact
                sources = {
                    c["source"] if isinstance(c, dict) else c.source
                    for c in message.artifact
                }
                assert sources == {source}
            documents = set(
                product_vector_store.get(
                    where={"product_id": product_id}, include=["documents"]
                )["documents"]
            )
            assert all(c.snippet in documents for c in citations)
            if first_state is None:
                first_state = state
            print(f"真实过滤通过：{product_id}，仅召回 {source}；回答：{answer}")

        # 切回 Pro 读取原来的独立上下文，并重新处理有历史的请求。
        returned_answer, returned_citations, _ = agent_invoke(
            query, session_id, "vivo_x500_pro"
        )
        after = agent.get_state(make_thread_config(session_id, "vivo_x500_pro")).values
        assert "50" in returned_answer
        assert returned_citations and {c.source for c in returned_citations} == {
            "x500_pro.md"
        }
        assert after["product_id"] == "vivo_x500_pro"
        assert (
            after["messages"][: len(first_state["messages"])] == first_state["messages"]
        )
        assert after["llm_calls"] > first_state["llm_calls"]
        assert len(cache.get_chat_history(session_id).messages) == 8
        print("真实商品续聊通过：同会话 Pro → 普通版 → Max → 恢复 Pro 原上下文。")
    finally:
        for product_id, _ in products:
            cache.redis_client.delete(cache.make_cache_key(query, product_id))
            checkpointer.delete_thread(make_thread_id(session_id, product_id))
        cache.get_chat_history(session_id).clear()


@pytest.mark.skipif(
    os.getenv("RUN_AGENT_INTEGRATION") != "1",
    reason="设置 RUN_AGENT_INTEGRATION=1 后运行真实商品提示词验收。",
)
def test_agent_product_prompt_handles_identity_facts_unknown_and_memory():
    """验收角色、商品事实、资料缺失和聊天记忆，避免只检查提示词文字。"""
    from langchain_core.messages import ToolMessage

    from backend.config import cache, checkpointer
    from backend.retrieval import agent, agent_invoke

    identifier = uuid.uuid4().hex
    identity_session = f"flame-prompt-identity-{identifier}"
    product_session = f"flame-prompt-product-{identifier}"
    marker = f"NOTE-{identifier[:12]}"
    identity_query = f"你是谁？（验收 {identifier}，无需重复编号。）"
    known_query = (
        f"我的备注代号是 {marker}，请记住。这款支持无线充电吗，最高功率是多少？"
        "请依据商品知识库回答并附引用。"
    )
    unknown_query = "这款手机处理器的 L2 缓存容量具体是多少？请查询商品资料。"
    memory_query = "我刚才给这款商品设定的备注代号是什么？只回答完整代号。"
    config = make_thread_config(product_session, "vivo_x500_pro")

    try:
        identity, identity_citations, _ = agent_invoke(identity_query, identity_session)
        assert "Flame" in identity and not identity_citations
        identity_state = agent.get_state(make_thread_config(identity_session)).values
        assert not any(isinstance(m, ToolMessage) for m in identity_state["messages"])

        known, known_citations, _ = agent_invoke(
            known_query, product_session, "vivo_x500_pro"
        )
        assert "50" in known and known_citations
        assert {c.source for c in known_citations} == {"x500_pro.md"}
        assert any(
            isinstance(m, ToolMessage) and m.name == "retrieve_product_docs"
            for m in agent.get_state(config).values["messages"]
        )

        unknown, unknown_citations, _ = agent_invoke(
            unknown_query, product_session, "vivo_x500_pro"
        )
        assert "暂未查询到相关信息" in unknown
        assert not unknown_citations, (
            f"未提供 L2 容量的商品原文不能作为该结论的引用。实际回答：{unknown}"
        )

        memory, memory_citations, _ = agent_invoke(
            memory_query, product_session, "vivo_x500_pro"
        )
        assert marker in memory and not memory_citations
        assert len(cache.get_chat_history(product_session).messages) == 6
        print(
            f"真实提示词验收通过：身份={identity}；已知参数={known}；资料缺失={unknown}；备注={memory}"
        )
    finally:
        for query, product_id in (
            (identity_query, None),
            (known_query, "vivo_x500_pro"),
        ):
            cache.redis_client.delete(cache.make_cache_key(query, product_id))
        for session_id, product_id in (
            (identity_session, None),
            (product_session, "vivo_x500_pro"),
        ):
            cache.get_chat_history(session_id).clear()
            checkpointer.delete_thread(make_thread_id(session_id, product_id))
