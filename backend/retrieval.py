"""提供带会话历史、缓存和引用的 Agent 问答入口。"""

from langchain.messages import HumanMessage
from backend.state import MessagesState
from backend.graph import agent_builder_graph
from backend.config import cache, CACHE_TTL, memory
from backend.citation import (
    citations_to_cache_payload,
    cache_payload_to_result,
    Citation,
)
from dotenv import load_dotenv
import uuid


load_dotenv()

# 导入模块时只构建并编译一次 Agent。
# backend.config 中的检查点对象为单例，同一张图可服务多个会话。
# 每次调用通过 thread_id 选择并恢复对应会话的状态。
agent = agent_builder_graph(state=MessagesState).compile(checkpointer=memory)


def agent_invoke(query: str, session_id: str | None = None):
    session_id = session_id or str(uuid.uuid4())
    cache_key = cache.make_cache_key(query)

    history = cache.get_chat_history(session_id)

    cached_payload = cache.get(cache_key)
    if cached_payload is not None:
        cached_answer, cached_citations = cache_payload_to_result(cached_payload)
        history.add_user_message(query)
        history.add_ai_message(cached_answer)
        return cached_answer, cached_citations, session_id

    config = {"configurable": {"thread_id": session_id}}

    result = agent.invoke({"messages": [HumanMessage(content=query)]}, config=config)

    answer = result["messages"][-1]
    citations = [Citation(**c) for c in answer.additional_kwargs.get("citations", [])]

    history.add_user_message(query)
    history.add_ai_message(answer.content)
    cache.set(
        cache_key,
        citations_to_cache_payload(answer.content, citations),
        ttl=CACHE_TTL,
    )

    return answer.content, citations, session_id


# 命令行调用示例：if __name__ == "__main__":
# 读取用户问题：query = input("请输入问题：")
# 调用问答入口：agent_invoke(query=query)
