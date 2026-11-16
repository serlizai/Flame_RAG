"""提供带会话历史、缓存和引用的 Agent 问答入口。"""

from dataclasses import asdict

from langchain.messages import HumanMessage, AIMessage, AIMessageChunk, ToolMessage
from backend.state import MessagesState
from backend.graph import agent_builder_graph
from backend.config import cache, CACHE_TTL, checkpointer
from backend.sessions import make_thread_config, session_lock
from backend.streaming import answer_stream_enabled
from backend.tracing import collect_tool_trace
from backend.conversation import load_context, resolve_conversation, save_context
from backend.scope import assess_scope
from backend.guardrails import guard_user_question, PromptSafeStream
from backend.products import load_products
from backend.prompts import QA_PROMPT, build_system_prompt
from backend.citation import (
    citations_to_cache_payload,
    cache_payload_to_result,
    Citation,
)
from dotenv import load_dotenv
import uuid


load_dotenv()

# 导入模块时只构建并编译一次 Agent。
# backend.config 中的 SQLite 检查点为单例，重启后仍可恢复已保存状态。
# 每次调用通过会话与商品对应的 thread_id 选择独立上下文。
agent = agent_builder_graph(state=MessagesState).compile(checkpointer=checkpointer)


def agent_invoke(
    query: str,
    session_id: str | None = None,
    product_id: str | None = None,
    *,
    trace: dict | None = None,
    use_cache: bool = True,
    follow_conversation: bool = False,
    emit=None,
):
    session_id = session_id or str(uuid.uuid4())
    with session_lock(session_id):
        token = answer_stream_enabled.set(emit is not None)
        try:
            return _agent_invoke(
                query,
                session_id,
                product_id,
                trace=trace,
                use_cache=use_cache,
                follow_conversation=follow_conversation,
                emit=emit,
            )
        finally:
            answer_stream_enabled.reset(token)


def _agent_invoke(
    query,
    session_id,
    product_id,
    *,
    trace,
    use_cache,
    follow_conversation,
    emit,
):
    """恢复当前商品的独立状态，仅在该对话没有历史时使用答案缓存。"""
    session_id = session_id or str(uuid.uuid4())
    original_query = query
    previous_context = load_context(cache, session_id) if follow_conversation else {}
    anchor = product_id or (
        previous_context.get("product_id") if follow_conversation else None
    )
    guard = guard_user_question(query, product_id=anchor)
    if guard.status == "refused":
        history = cache.get_chat_history(session_id)
        history.add_user_message(original_query)
        history.add_ai_message(guard.message)
        if trace is not None:
            trace.clear()
            trace.update(
                cache_hit=False,
                product_id=anchor,
                tools=[],
                steps=["guardrail_refuse"],
                original_question=original_query,
                resolved_question="",
                session_id=session_id,
                guardrail_status=guard.status,
            )
        return guard.message, [], session_id
    query = guard.question
    initial_config = make_thread_config(session_id, anchor)
    initial_state = agent.get_state(initial_config).values
    scope = assess_scope(
        query,
        product_id=anchor,
        has_product_context=bool(previous_context.get("topics"))
        or any(
            (isinstance(m, ToolMessage) and m.status == "success")
            or (
                isinstance(m, HumanMessage)
                and m.additional_kwargs.get("product_question")
            )
            for m in initial_state.get("messages", [])
        ),
    )
    blocked = scope.status in {"out_of_scope", "clarify"}
    if blocked:
        # 不解析越界问题中的商品名和话题，避免改变上一轮商品上下文。
        product_id = anchor
    else:
        query = scope.question
    resolution = (
        resolve_conversation(query, product_id, previous_context)
        if follow_conversation and not blocked
        else None
    )
    if resolution:
        query, product_id = resolution.question, resolution.product_id
    if emit:
        emit("meta", {"session_id": session_id, "product_id": product_id})
    cache_key = cache.make_cache_key(query, product_id=product_id)
    config = make_thread_config(session_id, product_id)
    previous_state = (
        initial_state if config == initial_config else agent.get_state(config).values
    )
    first_turn = not previous_state.get("messages")
    if trace is not None:
        trace.clear()
        trace.update(
            cache_hit=False,
            product_id=product_id,
            tools=[],
            steps=[],
            original_question=original_query,
            resolved_question=query,
            session_id=session_id,
            history_turns=sum(
                isinstance(message, HumanMessage)
                for message in previous_state.get("messages", [])
            ),
            scope_status=scope.status,
            scope_question=scope.question,
            guardrail_status=guard.status,
        )
        if resolution:
            trace.update(
                context_source=resolution.source,
                previous_product_id=previous_context.get("product_id"),
            )
            trace["steps"].append("resolve_context")

    history = cache.get_chat_history(session_id)
    if blocked:
        answer = scope.message
        # 展示历史保留原问题；商品图和话题索引不保存越界内容或含糊请求。
        history.add_user_message(original_query)
        history.add_ai_message(answer)
        if trace is not None:
            trace["steps"].append(
                "scope_reject" if scope.status == "out_of_scope" else "scope_clarify"
            )
        return answer, [], session_id

    metadata = {"product_question": scope.product_question}
    if resolution and query != scope.question:
        metadata.update(
            original_question=scope.question,
            resolved_topics=resolution.context["topics"],
        )
    user_message = HumanMessage(
        content=query,
        additional_kwargs=metadata,
    )

    def present_answer(answer):
        return f"{answer}\n\n{scope.message}" if scope.status == "mixed" else answer

    def remember_context():
        if resolution:
            save_context(cache, session_id, resolution.context, CACHE_TTL)

    if resolution and resolution.clarification:
        answer = present_answer(resolution.clarification)
        agent.update_state(
            config,
            {
                "messages": [
                    HumanMessage(content=scope.question),
                    AIMessage(content=answer),
                ],
                "product_id": product_id,
                "citations": [],
                "llm_calls": previous_state.get("llm_calls", 0),
            },
            as_node="final_answer",
        )
        history.add_user_message(original_query)
        history.add_ai_message(answer)
        remember_context()
        if trace is not None:
            trace.update(clarification=True)
            trace["steps"].append("clarify")
        return answer, [], session_id

    # 有历史的追问依赖当前商品对话，不读取或覆盖跨会话共享的首轮答案。
    cached_payload = (
        cache.get(cache_key)
        if first_turn
        and use_cache
        and scope.status == "allow"
        and guard.status == "allow"
        else None
    )
    if cached_payload is not None:
        if trace is not None:
            trace["cache_hit"] = True
            trace["steps"].append("cache")
        cached_answer, cached_citations = cache_payload_to_result(cached_payload)
        # 复用答案不执行模型，但仍将本轮对话和当前商品写入检查点。
        # 明确标记最终回答节点已完成，下次输入从新一轮流程开始。
        agent.update_state(
            config,
            {
                "messages": [
                    user_message,
                    AIMessage(
                        content=cached_answer,
                        additional_kwargs={
                            "citations": [asdict(c) for c in cached_citations]
                        },
                    ),
                ],
                "product_id": product_id,
                "citations": cached_citations,
                "llm_calls": previous_state.get("llm_calls", 0),
            },
            as_node="final_answer",
        )
        history.add_user_message(original_query)
        history.add_ai_message(cached_answer)
        remember_context()
        return cached_answer, cached_citations, session_id

    # 每个商品和全库模式分别保存状态，切换回来即可继续原来的对话。
    inputs = {"messages": [user_message], "product_id": product_id}
    if trace is None and emit is None:
        result = agent.invoke(inputs, config=config)
    else:
        result = None
        if emit:
            product = load_products().get(product_id)
            system_prompt = (
                build_system_prompt(product.name, product.id)
                if product
                else build_system_prompt()
            )
            stream_guard = PromptSafeStream(system_prompt, QA_PROMPT)
        for mode, event in agent.stream(
            inputs,
            config=config,
            stream_mode=["updates", "values", "messages"]
            if emit
            else ["updates", "values"],
        ):
            if mode == "messages" and emit:
                chunk, metadata = event
                if (
                    isinstance(chunk, AIMessageChunk)
                    and metadata.get("langgraph_node") == "final_answer"
                    and "answer_draft" in metadata.get("tags", [])
                ):
                    content = chunk.content
                    if isinstance(content, list):
                        content = "".join(
                            block.get("text", "")
                            for block in content
                            if isinstance(block, dict) and block.get("type") == "text"
                        )
                    safe_content = stream_guard.push(content)
                    if safe_content:
                        emit("delta", {"text": safe_content})
            elif mode == "updates":
                if emit:
                    emit(
                        "status",
                        {
                            "message": "正在整理回答…"
                            if "tool_node" in event
                            else "正在查阅商品资料…"
                        },
                    )
            if mode == "updates" and trace is not None:
                trace["steps"].extend(
                    name
                    for name in event
                    if name in {"llm_call", "tool_node", "final_answer"}
                )
            if mode == "values":
                result = event
        if result is None:
            raise RuntimeError("Agent did not produce a final state")
    if trace is not None:
        new_messages = result["messages"][len(previous_state.get("messages", [])) :]
        trace["tools"] = collect_tool_trace(new_messages, product_id)

    answer = result["messages"][-1]
    citations = [Citation(**c) for c in answer.additional_kwargs.get("citations", [])]
    output_blocked = answer.additional_kwargs.get("guardrail_blocked", False)
    grounding_empty = answer.additional_kwargs.get("grounding_empty", False)
    if emit and not output_blocked:
        tail = stream_guard.finish()
        if tail:
            emit("delta", {"text": tail})
    if trace is not None and output_blocked:
        trace["guardrail_output_blocked"] = True
    if trace is not None and grounding_empty:
        trace["no_evidence"] = True

    history.add_user_message(original_query)
    history.add_ai_message(present_answer(answer.content))
    remember_context()
    # 业务数据可随时更新，包含价格/售后/商品接口调用的回答不进入共享缓存。
    has_business_call = any(
        isinstance(message, ToolMessage)
        and message.name in {"product_info_tool", "price_tool", "after_sales_tool"}
        for message in result["messages"]
    )
    has_tool_error = any(
        isinstance(message, ToolMessage) and message.status == "error"
        for message in result["messages"]
    )
    if (
        use_cache
        and first_turn
        and scope.status == "allow"
        and guard.status == "allow"
        and not output_blocked
        and not grounding_empty
        and not has_business_call
        and not has_tool_error
    ):
        cache.set(
            cache_key,
            citations_to_cache_payload(answer.content, citations),
            ttl=CACHE_TTL,
        )

    return present_answer(answer.content), citations, session_id


# 命令行调用示例：if __name__ == "__main__":
# 读取用户问题：query = input("请输入问题：")
# 调用问答入口：agent_invoke(query=query)
