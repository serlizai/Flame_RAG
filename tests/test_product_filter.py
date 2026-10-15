"""用真实 Chroma 和 LangGraph 验证正式商品过滤，替代聊天模型和 Redis。"""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import uuid

from langchain_chroma import Chroma
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
import pytest

from backend.cache import PRODUCT_CACHE_KEY_PREFIX, RedisCache
from backend.citation import Citation
from backend.sessions import make_thread_config


class OfflineEmbeddings(Embeddings):
    """只验证数据库实际执行商品过滤，不评估语义排序。"""

    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


class LocalCache(RedisCache):
    """使用正式缓存键算法和内存存储，避免连接 Redis。"""

    def __init__(self):
        self.values = {}
        self.histories = {}

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value, ttl=None):
        self.values[key] = value

    def get_chat_history(self, session_id):
        return self.histories.setdefault(session_id, InMemoryChatMessageHistory())


class RecordingModel:
    """按正式工具协议发起检索，记录模型实际收到的上下文。"""

    def __init__(self):
        self.tool_inputs = []
        self.answer_inputs = []
        self.answer = "根据当前商品资料回答。[1]"

    def bind_tools(self, tools):
        def invoke(messages):
            self.tool_inputs.append(messages)
            if isinstance(messages[-1], HumanMessage):
                return AIMessage(
                    content="",
                    tool_calls=[tool_call(messages[-1].content)],
                )
            return AIMessage(content="资料已检索。")

        return SimpleNamespace(invoke=invoke)

    def invoke(self, messages):
        self.answer_inputs.append(messages)
        return AIMessage(content=self.answer)


def tool_call(query="这款支持无线充电吗？", **kwargs):
    """创建与真实模型输出一致的工具调用结构。"""
    return {
        "type": "tool_call",
        "name": "retrieve_docs",
        "id": uuid.uuid4().hex,
        "args": {"query": query, **kwargs},
    }


@pytest.fixture
def product_agent(tmp_path, monkeypatch):
    """加载正式工具、节点、工作流和入口，使用临时商品集合。"""
    store = Chroma(
        collection_name="products",
        persist_directory=str(tmp_path / "products"),
        embedding_function=OfflineEmbeddings(),
    )
    documents = [
        Document(
            page_content=f"{name} 的独立资料：{description}",
            metadata={
                "product_id": product_id,
                "product_name": name,
                "source": source,
                "section": "FAQ",
            },
        )
        for product_id, name, source, description in (
            ("vivo_x500", "vivo X500", "x500.md", "不支持无线充电。"),
            ("vivo_x500_pro", "vivo X500 Pro", "x500_pro.md", "支持 50W 无线充电。"),
            (
                "vivo_x500_pro_max",
                "vivo X500 Pro Max",
                "x500_pro_max.md",
                "支持 50W 无线充电，大屏版本。",
            ),
        )
    ]
    store.add_documents(
        documents, ids=[doc.metadata["product_id"] for doc in documents]
    )
    model = RecordingModel()
    cache = LocalCache()
    monkeypatch.setitem(
        sys.modules,
        "backend.config",
        SimpleNamespace(
            model=model,
            product_vector_store=store,
            cache=cache,
            CACHE_TTL=60,
            checkpointer=MemorySaver(),
        ),
    )
    modules = {}
    for name in ("tools", "nodes", "graph", "retrieval"):
        path = Path(__file__).resolve().parents[1] / "backend" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"backend.{name}", path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, f"backend.{name}", module)
        spec.loader.exec_module(module)
        modules[name] = module
    return SimpleNamespace(**modules, model=model, cache=cache, store=store)


@pytest.mark.parametrize(
    "product_id, sources",
    [
        ("vivo_x500", {"x500.md"}),
        ("vivo_x500_pro", {"x500_pro.md"}),
        ("vivo_x500_pro_max", {"x500_pro_max.md"}),
        (None, {"x500.md", "x500_pro.md", "x500_pro_max.md"}),
        ("missing-product", set()),
    ],
)
def test_real_chroma_limits_results_to_selected_product(
    product_agent, product_id, sources
):
    result = product_agent.tools.retrieve_docs.invoke(tool_call(product_id=product_id))
    assert {citation.source for citation in result.artifact} == sources
    if not sources:
        assert "暂未查询到" in result.content
    else:
        assert all(citation.snippet in result.content for citation in result.artifact)


def test_model_can_only_supply_search_query(product_agent):
    schema = product_agent.tools.retrieve_docs.tool_call_schema.model_json_schema()
    assert set(schema["properties"]) == {"query"}


def test_tool_node_overrides_model_supplied_product_and_number(product_agent):
    call = tool_call(product_id="vivo_x500_pro_max", citation_start=999)
    state = {
        "messages": [
            HumanMessage(content="这款呢？"),
            AIMessage(content="", tool_calls=[call]),
        ],
        "product_id": "vivo_x500_pro",
    }
    result = product_agent.nodes.tool_node(state)
    assert {citation.source for citation in result["citations"]} == {"x500_pro.md"}
    assert [citation.number for citation in result["citations"]] == [1]
    assert call["args"]["product_id"] == "vivo_x500_pro_max"


def test_multiple_tool_calls_use_distinct_citation_numbers(product_agent):
    state = {
        "messages": [
            HumanMessage(content="功能和使用方法？"),
            AIMessage(
                content="", tool_calls=[tool_call("功能"), tool_call("使用方法")]
            ),
        ],
        "product_id": "vivo_x500_pro",
    }
    result = product_agent.nodes.tool_node(state)
    assert [citation.number for citation in result["citations"]] == [1, 2]
    assert result["messages"][0].content.startswith("[1]")
    assert result["messages"][1].content.startswith("[2]")


def test_graph_switches_product_without_exposing_previous_product_context(
    product_agent,
):
    entry = product_agent.retrieval
    pro_config = make_thread_config("product-switch", "vivo_x500_pro")
    config = make_thread_config("product-switch", "vivo_x500")
    query = "这款支持无线充电吗？"
    pro_answer, pro_citations, _ = entry.agent_invoke(
        query, "product-switch", "vivo_x500_pro"
    )
    pro_messages = entry.agent.get_state(pro_config).values["messages"]

    _, base_citations, _ = entry.agent_invoke(query, "product-switch", "vivo_x500")
    state = entry.agent.get_state(config).values
    assert {c.source for c in pro_citations} == {"x500_pro.md"}
    assert {c.source for c in base_citations} == {"x500.md"}
    assert len(state["messages"]) == len(pro_messages)
    assert pro_messages == entry.agent.get_state(pro_config).values["messages"]
    assert (
        "vivo X500（product_id=vivo_x500）"
        in product_agent.model.answer_inputs[-1][0].content
    )
    assert all(
        "vivo X500 Pro" not in str(m.content)
        for m in product_agent.model.answer_inputs[-1]
    )
    assert all(
        "vivo X500 Pro" not in str(m.content)
        for m in product_agent.model.tool_inputs[-1]
    )

    # 连续使用同一商品时保留上下文；下一次引用从现有最大编号后继续。
    product_agent.model.answer = "再次查询资料。[2]"
    _, followup_citations, _ = entry.agent_invoke(
        "它如何充电？", "product-switch", "vivo_x500"
    )
    assert [c.number for c in followup_citations] == [2]
    assert any(m.content == query for m in product_agent.model.answer_inputs[-1])
    assert len(entry.agent.get_state(config).values["messages"]) == 2 * len(
        pro_messages
    )

    # 切回 Pro 继续原线程，既能读取 Pro 历史，也不会看到普通版的资料。
    product_agent.model.answer = "再次查询 Pro。[2]"
    entry.agent_invoke("它的无线充功率呢？", "product-switch", "vivo_x500_pro")
    assert (
        entry.agent.get_state(pro_config).values["messages"][: len(pro_messages)]
        == pro_messages
    )
    assert any(m.content == query for m in product_agent.model.answer_inputs[-1])
    assert all(
        "vivo X500 的独立资料" not in str(m.content)
        for m in product_agent.model.answer_inputs[-1]
    )


def test_returning_to_whole_library_drops_previous_product_context(product_agent):
    entry = product_agent.retrieval
    entry.agent_invoke("先看当前商品", "whole-library", "vivo_x500_pro")
    entry.agent_invoke("介绍所有商品", "whole-library", None)
    messages = product_agent.model.answer_inputs[-1]
    assert "当前商品：" not in messages[0].content
    assert all(m.content != "先看当前商品" for m in messages)
    tools = [m for m in messages if isinstance(m, ToolMessage)]
    assert {c.source for c in tools[-1].artifact} == {
        "x500.md",
        "x500_pro.md",
        "x500_pro_max.md",
    }


def test_final_answer_ignores_previous_product_citations(product_agent):
    old = Citation(99, "vivo X500 Pro Max", "旧商品资料", "x500_pro_max.md")
    current = Citation(1, "vivo X500", "当前商品资料", "x500.md")
    state = {
        "product_id": "vivo_x500",
        "messages": [
            HumanMessage(content="当前问题"),
            ToolMessage(
                content="当前商品资料 [1]", artifact=[current], tool_call_id="current"
            ),
        ],
        "citations": [old, current],
    }
    product_agent.model.answer = "当前答案。[1] 旧引用。[99]"
    result = product_agent.nodes.final_answer(state)["messages"][0]
    assert "[99]" not in result.content
    assert {c["source"] for c in result.additional_kwargs["citations"]} == {"x500.md"}
    assert all(
        "旧商品资料" not in m.content for m in product_agent.model.answer_inputs[-1]
    )


def test_unfiltered_product_cache_is_not_reused(product_agent):
    cache = product_agent.cache
    query = "这款支持什么？"
    new_key = cache.make_cache_key(query, "vivo_x500_pro")
    assert new_key.startswith(PRODUCT_CACHE_KEY_PREFIX)
    old_key = new_key.replace(PRODUCT_CACHE_KEY_PREFIX, "flame:llm_cache_product_v1:")
    cache.values[old_key] = "旧的未过滤答案"
    product_agent.retrieval.agent_invoke(query, "new-cache-version", "vivo_x500_pro")
    assert product_agent.model.answer_inputs
    assert cache.values[old_key] == "旧的未过滤答案"
