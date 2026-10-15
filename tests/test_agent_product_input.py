"""验证真实 LangGraph 检查点中的商品身份和缓存状态，不调用外部服务。"""

import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
import pytest

from backend.cache import CACHE_KEY_PREFIX, RedisCache
from backend.citation import Citation, citations_to_cache_payload
from backend.sessions import make_thread_config
from backend.state import MessagesState


class LocalCache(RedisCache):
    """复用实际缓存键算法，用内存替代外部存储。"""

    def __init__(self):
        self.values = {}
        self.histories = {}

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value, ttl=None):
        self.values[key] = value

    def get_chat_history(self, session_id):
        return self.histories.setdefault(session_id, InMemoryChatMessageHistory())


@pytest.fixture
def retrieval_entry(monkeypatch):
    """使用真实状态图和内存检查点，仅替代模型节点和外部 Redis。"""
    cache = LocalCache()
    calls = []

    def final_answer(state):
        return {
            "messages": [AIMessage(content="示例回答")],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    graph = StateGraph(MessagesState)
    graph.add_node("final_answer", final_answer)
    graph.add_edge(START, "final_answer")
    graph.add_edge("final_answer", END)
    compiled = graph.compile(checkpointer=MemorySaver())

    def invoke(inputs, config):
        calls.append((inputs, config))
        return compiled.invoke(inputs, config=config)

    agent = SimpleNamespace(
        invoke=invoke, get_state=compiled.get_state, update_state=compiled.update_state
    )
    builder = SimpleNamespace(compile=lambda checkpointer: agent)
    monkeypatch.setitem(
        sys.modules,
        "backend.graph",
        SimpleNamespace(agent_builder_graph=lambda state: builder),
    )
    monkeypatch.setitem(
        sys.modules,
        "backend.config",
        SimpleNamespace(cache=cache, CACHE_TTL=60, checkpointer=object()),
    )
    path = Path(__file__).resolve().parents[1] / "backend" / "retrieval.py"
    spec = importlib.util.spec_from_file_location("flame_product_input_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, cache, calls


def test_product_id_is_saved_in_graph_state(retrieval_entry):
    entry, cache, calls = retrieval_entry
    answer, citations, session_id = entry.agent_invoke(
        "这款支持什么？", "existing-session", "vivo_x500_pro"
    )
    assert answer == "示例回答" and citations == []
    assert session_id == "existing-session"
    assert calls[0][0]["product_id"] == "vivo_x500_pro"
    assert calls[0][1] == make_thread_config("existing-session", "vivo_x500_pro")
    state = entry.agent.get_state(calls[0][1]).values
    assert state["product_id"] == "vivo_x500_pro"
    assert len(cache.get_chat_history(session_id).messages) == 2


def test_same_question_has_separate_product_and_whole_library_caches(retrieval_entry):
    entry, cache, calls = retrieval_entry
    query = "这款支持什么？"
    for product_id in (None, "vivo_x500", "vivo_x500_pro"):
        entry.agent_invoke(query, "session", product_id)
    assert len(calls) == 3
    assert len(cache.values) == 3

    entry.agent_invoke(query, "new-session", "vivo_x500_pro")
    assert len(calls) == 3
    assert cache.make_cache_key(query) == (
        CACHE_KEY_PREFIX + hashlib.sha256(query.encode()).hexdigest()
    )


def test_switching_product_keeps_independent_conversations(retrieval_entry):
    entry, _, _ = retrieval_entry
    for query, product_id in (
        ("第一个问题", "vivo_x500_pro"),
        ("连续追问", "vivo_x500_pro"),
        ("切换商品后的问题", "vivo_x500"),
        ("全库问题", None),
    ):
        entry.agent_invoke(query, "session", product_id)
        config = make_thread_config("session", product_id)
        assert entry.agent.get_state(config).values["product_id"] == product_id
    for product_id, count in (("vivo_x500_pro", 4), ("vivo_x500", 2), (None, 2)):
        state = entry.agent.get_state(make_thread_config("session", product_id)).values
        assert len(state["messages"]) == count


def test_different_sessions_keep_independent_product_state(retrieval_entry):
    entry, _, _ = retrieval_entry
    entry.agent_invoke("问题一", "session-one", "vivo_x500_pro")
    entry.agent_invoke("问题二", "session-two", "vivo_x500")
    for session_id, product_id in (
        ("session-one", "vivo_x500_pro"),
        ("session-two", "vivo_x500"),
    ):
        state = entry.agent.get_state(make_thread_config(session_id, product_id)).values
        assert state["product_id"] == product_id
        assert len(state["messages"]) == 2


def test_cache_hit_in_new_session_saves_state_without_running_model(retrieval_entry):
    entry, cache, calls = retrieval_entry
    query = "同一个商品问题"
    entry.agent_invoke(query, "original-session", "vivo_x500_pro")
    entry.agent_invoke(query, "new-session", "vivo_x500_pro")

    snapshot = entry.agent.get_state(make_thread_config("new-session", "vivo_x500_pro"))
    assert len(calls) == 1
    assert snapshot.values["product_id"] == "vivo_x500_pro"
    assert len(snapshot.values["messages"]) == 2
    assert snapshot.values["messages"][0].content == query
    assert snapshot.values["llm_calls"] == 0
    assert snapshot.next == ()
    assert len(cache.get_chat_history("new-session").messages) == 2


@pytest.mark.parametrize("cached_product_id", ["vivo_x500_pro", None])
def test_returning_to_product_runs_with_its_own_history(
    retrieval_entry, cached_product_id
):
    entry, _, calls = retrieval_entry
    query = "商品问题"
    entry.agent_invoke(query, "session", cached_product_id)
    entry.agent_invoke("另一个问题", "session", "vivo_x500")
    entry.agent_invoke(query, "session", cached_product_id)

    state = entry.agent.get_state(
        make_thread_config("session", cached_product_id)
    ).values
    assert len(calls) == 3
    assert state["product_id"] == cached_product_id
    assert state["llm_calls"] == 2
    assert len(state["messages"]) == 4


def test_cache_hit_saves_citation_objects_and_answer_metadata(retrieval_entry):
    entry, cache, calls = retrieval_entry
    query = "已缓存的商品问题"
    citation = Citation(1, "商品 FAQ", "完整引用原文", "sample.md")
    cache.set(
        cache.make_cache_key(query, "vivo_x500_pro"),
        citations_to_cache_payload("已缓存回答。[1]", [citation]),
    )

    entry.agent_invoke(query, "cached-session", "vivo_x500_pro")
    state = entry.agent.get_state(
        make_thread_config("cached-session", "vivo_x500_pro")
    ).values
    assert calls == []
    assert state["citations"] == [citation]
    assert state["messages"][-1].additional_kwargs["citations"][0]["snippet"] == (
        "完整引用原文"
    )


def test_followup_ignores_cache_and_does_not_overwrite_first_turn_answer(
    retrieval_entry,
):
    """同样的问题在有历史时也重新执行图，且不能污染其他会话的首轮缓存。"""
    entry, cache, calls = retrieval_entry
    query = "刚才说的是什么？"
    entry.agent_invoke("先前的对话", "session", "vivo_x500_pro")
    key = cache.make_cache_key(query, "vivo_x500_pro")
    cache.values[key] = "特意设置为不能解析的缓存，确保追问完全跳过缓存"
    entry.agent_invoke(query, "session", "vivo_x500_pro")
    assert len(calls) == 2
    assert cache.values[key] == "特意设置为不能解析的缓存，确保追问完全跳过缓存"


def test_whole_library_and_product_threads_both_resume_their_history(retrieval_entry):
    entry, cache, calls = retrieval_entry
    for query, product_id in (
        ("全库第一轮", None),
        ("Pro 第一轮", "vivo_x500_pro"),
        ("全库第二轮", None),
        ("Pro 第二轮", "vivo_x500_pro"),
    ):
        entry.agent_invoke(query, "session", product_id)
    assert len(calls) == 4
    assert len(cache.histories["session"].messages) == 8
    for product_id, prefix in ((None, "全库"), ("vivo_x500_pro", "Pro")):
        state = entry.agent.get_state(make_thread_config("session", product_id)).values
        assert [m.content for m in state["messages"][::2]] == [
            f"{prefix}第一轮" if prefix == "全库" else "Pro 第一轮",
            f"{prefix}第二轮" if prefix == "全库" else "Pro 第二轮",
        ]
