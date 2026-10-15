"""验证商品线程隔离、SQLite 状态恢复、连接释放和跨进程续聊。"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
import pytest

from backend.checkpoints import checkpoint_path, sqlite_checkpointer
from backend.citation import Citation
from backend.sessions import make_thread_config, make_thread_id
from backend.state import MessagesState
import settings


def test_product_threads_are_stable_and_unambiguous():
    identities = [
        ("session", None),
        ("session", "vivo_x500_pro"),
        ("session", "vivo_x500"),
        ("other-session", "vivo_x500_pro"),
        ("A:B", "C"),
        ("A", "B:C"),
        ("session", "null"),
    ]
    assert len({make_thread_id(*identity) for identity in identities}) == len(
        identities
    )
    assert make_thread_id("session", "vivo_x500_pro") == make_thread_id(
        "session", "vivo_x500_pro"
    )
    assert make_thread_config("session")["configurable"]["thread_id"] == make_thread_id(
        "session"
    )


def test_checkpoint_paths_use_project_root(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("CHECKPOINT_SQLITE_PATH", raising=False)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert checkpoint_path() == tmp_path / "data" / "checkpoints.sqlite"
    monkeypatch.setenv("CHECKPOINT_SQLITE_PATH", "custom/state.sqlite")
    assert checkpoint_path() == tmp_path / "custom" / "state.sqlite"
    assert checkpoint_path(tmp_path / "explicit.sqlite") == tmp_path / "explicit.sqlite"


def _graph(checkpointer):
    """建立小型真实状态图，保存工具原文和项目引用对象。"""
    graph = StateGraph(MessagesState)
    citation = Citation(1, "vivo X500 Pro, FAQ", "持久化引用原文", "x500_pro.md")

    def final_answer(state):
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "retrieve_docs", "args": {}, "id": "call"}],
                ),
                ToolMessage(
                    content="持久化引用原文", artifact=[citation], tool_call_id="call"
                ),
                AIMessage(
                    content="已有回答。[1]",
                    additional_kwargs={"citations": [asdict(citation)]},
                ),
            ],
            "citations": [citation],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    graph.add_node("final_answer", final_answer)
    graph.add_edge(START, "final_answer")
    graph.add_edge("final_answer", END)
    return graph.compile(checkpointer=checkpointer)


def test_reopening_database_restores_messages_citations_and_product(tmp_path):
    path = tmp_path / "nested" / "checkpoints.sqlite"
    config = make_thread_config("session", "vivo_x500_pro")
    with sqlite_checkpointer(path) as saver:
        _graph(saver).invoke(
            {
                "messages": [HumanMessage(content="第一轮问题")],
                "product_id": "vivo_x500_pro",
            },
            config=config,
        )
        assert saver.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    with pytest.raises(sqlite3.ProgrammingError):
        saver.conn.execute("SELECT 1")

    with sqlite_checkpointer(path) as reopened:
        graph = _graph(reopened)
        snapshot = graph.get_state(config)
        assert snapshot.next == ()
        assert snapshot.values["product_id"] == "vivo_x500_pro"
        assert snapshot.values["messages"][0].content == "第一轮问题"
        assert snapshot.values["citations"][0] == Citation(
            1, "vivo X500 Pro, FAQ", "持久化引用原文", "x500_pro.md"
        )
        assert isinstance(snapshot.values["messages"][2], ToolMessage)
        graph.invoke({"messages": [HumanMessage(content="第二轮追问")]}, config=config)
        state = graph.get_state(config).values
        assert state["llm_calls"] == 2
        assert [
            m.content for m in state["messages"] if isinstance(m, HumanMessage)
        ] == ["第一轮问题", "第二轮追问"]


def test_sync_checkpointer_works_in_request_threads(tmp_path):
    with sqlite_checkpointer(tmp_path / "threads.sqlite") as saver:
        graph = _graph(saver)

        def invoke(number):
            config = make_thread_config(f"session-{number}", "vivo_x500_pro")
            graph.invoke(
                {"messages": [HumanMessage(content=str(number))]}, config=config
            )
            return graph.get_state(config).values["messages"][0].content

        with ThreadPoolExecutor(max_workers=2) as pool:
            assert list(pool.map(invoke, range(4))) == ["0", "1", "2", "3"]


def test_delete_thread_does_not_remove_another_product(tmp_path):
    with sqlite_checkpointer(tmp_path / "delete.sqlite") as saver:
        graph = _graph(saver)
        for product_id in ("vivo_x500_pro", "vivo_x500"):
            graph.invoke(
                {
                    "messages": [HumanMessage(content=product_id)],
                    "product_id": product_id,
                },
                config=make_thread_config("session", product_id),
            )
        saver.delete_thread(make_thread_id("session", "vivo_x500_pro"))
        assert (
            graph.get_state(make_thread_config("session", "vivo_x500_pro")).values == {}
        )
        assert graph.get_state(make_thread_config("session", "vivo_x500")).values[
            "messages"
        ]


def test_actual_entry_resumes_product_history_in_a_new_process(tmp_path):
    """重启整个 Python 进程，正式入口从 SQLite 读取 Pro 历史并排除普通版。"""
    root = Path(__file__).resolve().parents[1]
    script = r"""
import json
import sys
from types import SimpleNamespace
sys.path.insert(0, sys.argv[1])
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from backend.cache import RedisCache
from backend.checkpoints import sqlite_checkpointer
from backend.sessions import make_thread_config

class LocalCache(RedisCache):
    def __init__(self):
        self.histories = {}
    def get(self, key):
        return None
    def set(self, key, value, ttl=None):
        pass
    def get_chat_history(self, session_id):
        return self.histories.setdefault(session_id, InMemoryChatMessageHistory())

def builder(state):
    graph = StateGraph(state)
    def answer(state):
        humans = [m.content for m in state['messages'] if isinstance(m, HumanMessage)]
        return {'messages': [AIMessage(content=humans[0])], 'llm_calls': state.get('llm_calls', 0) + 1}
    graph.add_node('final_answer', answer)
    graph.add_edge(START, 'final_answer')
    graph.add_edge('final_answer', END)
    return graph

with sqlite_checkpointer(sys.argv[2]) as saver:
    sys.modules['backend.config'] = SimpleNamespace(cache=LocalCache(), CACHE_TTL=60, checkpointer=saver)
    sys.modules['backend.graph'] = SimpleNamespace(agent_builder_graph=builder)
    from backend.retrieval import agent, agent_invoke
    if sys.argv[3] == 'seed':
        agent_invoke('Pro 的预算是 4321 元', 'session', 'vivo_x500_pro')
        agent_invoke('普通版的预算是 9876 元', 'session', 'vivo_x500')
    else:
        answer, _, session_id = agent_invoke('刚才的预算多少？', 'session', 'vivo_x500_pro')
        state = agent.get_state(make_thread_config('session', 'vivo_x500_pro')).values
        print(json.dumps({'answer': answer, 'session_id': session_id, 'calls': state['llm_calls'],
            'questions': [m.content for m in state['messages'] if isinstance(m, HumanMessage)]}))
"""
    path = tmp_path / "restart.sqlite"
    for mode in ("seed", "resume"):
        result = subprocess.run(
            [sys.executable, "-c", script, str(root), str(path), mode],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    data = json.loads(result.stdout)
    assert data == {
        "answer": "Pro 的预算是 4321 元",
        "session_id": "session",
        "calls": 2,
        "questions": ["Pro 的预算是 4321 元", "刚才的预算多少？"],
    }
