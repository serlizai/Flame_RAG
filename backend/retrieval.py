"""提供带会话历史、缓存和引用的 Agent 问答入口。"""

from dataclasses import asdict

from langchain.messages import HumanMessage, AIMessage
from backend.state import MessagesState
from backend.graph import agent_builder_graph
from backend.config import cache, CACHE_TTL, checkpointer
from backend.sessions import make_thread_config
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
    query: str, session_id: str | None = None, product_id: str | None = None
):
    """恢复当前商品的独立状态，仅在该对话没有历史时使用答案缓存。"""
    session_id = session_id or str(uuid.uuid4())
    cache_key = cache.make_cache_key(query, product_id=product_id)
    config = make_thread_config(session_id, product_id)
    previous_state = agent.get_state(config).values
    first_turn = not previous_state.get("messages")

    history = cache.get_chat_history(session_id)

    # 有历史的追问依赖当前商品对话，不读取或覆盖跨会话共享的首轮答案。
    cached_payload = cache.get(cache_key) if first_turn else None
    if cached_payload is not None:
        cached_answer, cached_citations = cache_payload_to_result(cached_payload)
        # 复用答案不执行模型，但仍将本轮对话和当前商品写入检查点。
        # 明确标记最终回答节点已完成，下次输入从新一轮流程开始。
        agent.update_state(
            config,
            {
                "messages": [
                    HumanMessage(content=query),
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
        history.add_user_message(query)
        history.add_ai_message(cached_answer)
        return cached_answer, cached_citations, session_id

    # 每个商品和全库模式分别保存状态，切换回来即可继续原来的对话。
    result = agent.invoke(
        {"messages": [HumanMessage(content=query)], "product_id": product_id},
        config=config,
    )

    answer = result["messages"][-1]
    citations = [Citation(**c) for c in answer.additional_kwargs.get("citations", [])]

    history.add_user_message(query)
    history.add_ai_message(answer.content)
    if first_turn:
        cache.set(
            cache_key,
            citations_to_cache_payload(answer.content, citations),
            ttl=CACHE_TTL,
        )

    return answer.content, citations, session_id


# 命令行调用示例：if __name__ == "__main__":
# 读取用户问题：query = input("请输入问题：")
# 调用问答入口：agent_invoke(query=query)
