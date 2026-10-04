"""提供经典 RAG 对话接口及会话历史管理。"""

import threading
import logging
from cache import RedisCache
from citations import (
    annotate_documents_for_citation,
    cache_payload_to_result,
    citations_to_cache_payload,
    resolve_answer_citations,
)
from chains import get_rag_chain
from prompts import SYSTEM_PROMPT, QA_PROMPT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)


class Flame:
    """通过检索增强生成链路回答问题的 Flame 对话接口。

    结合向量检索和语言模型生成回答，使用 Redis 缓存结果，
    并在进程内保存按会话访问的聊天历史对象。

    属性：
        llm：生成回答的语言模型。
        embeddings：文本向量模型。
        vector_store：用于语义检索的向量库。
        cache：保存回答与聊天历史的 RedisCache 实例。

    参数：
        llm：语言模型实例。
        embeddings：向量模型实例。
        vector_store：检索相关文档的向量库实例。
        redis_url：Redis 地址，默认为 redis://localhost:6379/0。

    方法：
        get_session_history(session_id)：获取或初始化指定会话的历史。
        conversational(query, session_id)：读取缓存或执行 RAG，更新历史并返回回答、消息和引用。
    """

    store = {}
    store_lock = threading.Lock()

    def __init__(
        self, llm, embeddings, vector_store, redis_url="redis://localhost:6379/0"
    ):
        self.llm = llm
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.cache = RedisCache(redis_url)

    def get_session_history(self, session_id):
        with Flame.store_lock:
            if session_id not in Flame.store:
                Flame.store[session_id] = self.cache.get_chat_history(session_id)
                logging.info(f"Created new chat history for session_id: {session_id}")
            else:
                logging.debug(
                    f"Using existing chat history for session_id: {session_id}"
                )
        return Flame.store[session_id]

    def conversational(self, query, session_id):
        """处理指定会话中的用户问题。

        命中缓存时直接返回已有回答；否则执行 RAG 链路并更新历史。

        返回：
            三元组，依次为模型回答、更新后的消息列表和引用列表。
        """
        cache_key = self.cache.make_cache_key(query, session_id)
        cached_payload = self.cache.get(cache_key)
        if cached_payload:
            logging.info(f"Cache hit for key: {cache_key}")
            answer, citations = cache_payload_to_result(cached_payload)
            chat_history = self.get_session_history(session_id).messages
            return answer, chat_history, citations

        logging.info(f"Cache miss for key: {cache_key}. Generating new answer.")

        rag_chain = get_rag_chain(self.llm, self.vector_store, SYSTEM_PROMPT, QA_PROMPT)

        chat_history_obj = self.get_session_history(session_id)
        messages = chat_history_obj.messages

        response = rag_chain.invoke(
            {"input": query, "chat_history": messages},
            config={"configurable": {"session_id": session_id}},
        )

        _, citations = annotate_documents_for_citation(response["context"])
        answer, citations = resolve_answer_citations(response["answer"], citations)

        # 更新聊天历史
        chat_history_obj.add_user_message(query)
        chat_history_obj.add_ai_message(answer)

        # 缓存回答及引用
        self.cache.set(cache_key, citations_to_cache_payload(answer, citations))

        return answer, chat_history_obj.messages, citations
