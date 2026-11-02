"""验收工具自身必须识别错误路由、缓存、引用和商品记忆污染。"""

import json

import pytest

from backend.conversation import resolve_conversation
from scripts import check_agent_acceptance as checks


@pytest.fixture
def simulator():
    contexts, counts, calls, cleaned = {}, {}, [], []

    def invoke(query, session, product_id, *, trace, use_cache, follow_conversation):
        calls.append((query, session, product_id, use_cache, follow_conversation))
        if follow_conversation:
            resolved = resolve_conversation(
                query, product_id, contexts.get(session, {})
            )
            product_id, query = resolved.product_id, resolved.question
            contexts[session] = resolved.context
        history = counts.get((session, product_id), 0)
        counts[(session, product_id)] = history + 1
        missing = "12+512" in query
        memory = "备注代号是什么" in query
        if follow_conversation:
            tools = [] if memory else ["product_info_tool"]
            answer = (
                "WEEK7-PRO"
                if memory
                else (
                    "不支持无线充电，功率不适用。[1]"
                    if product_id == "vivo_x500"
                    else "支持最高 50W 无线充电。[1]"
                )
            )
        else:
            case = next(case for case in checks.WEEK6_CASES if case[1] == query)
            tools = case[2]
            answer = (
                "暂未查询到该规格"
                if missing
                else "月光银，3999 元，7 天内未激活、未使用、包装配件齐全并有订单凭证可申请退货。[1]"
            )
        source = (
            "x500_pro.md" if tools == ["product_rag_tool"] else "product_business.json"
        )
        citations = (
            []
            if memory or missing
            else [
                {
                    "number": 1,
                    "label": "商品依据",
                    "source": source,
                    "snippet": "演示数据\n" + json.dumps({"product_id": product_id}),
                }
            ]
        )
        trace.update(
            product_id=product_id,
            cache_hit=False,
            history_turns=history,
            resolved_question=query,
            steps=(["resolve_context"] if follow_conversation else [])
            + ["llm_call", "tool_node", "llm_call", "final_answer"],
            tools=[
                {
                    "name": name,
                    "arguments": {"product_id": product_id},
                    "status": "success",
                }
                for name in tools
            ],
        )
        return answer, citations, session

    return invoke, calls, cleaned


def test_scenarios_use_actual_response_for_next_product_and_clean_own_sessions(
    simulator,
):
    invoke, calls, cleaned = simulator
    report = checks.run_acceptance(
        invoke, inspect=lambda *_: [], cleanup=cleaned.append
    )
    assert report["passed"], report
    assert len(report["results"]) == 10
    assert all(call[3] is False for call in calls)
    assert len(set(call[1] for call in calls[:6])) == 6
    assert len(set(call[1] for call in calls[6:])) == 1
    assert [call[2] for call in calls[6:]] == [
        "vivo_x500_pro",
        "vivo_x500_pro",
        "vivo_x500",
        "vivo_x500_pro_max",
    ]
    assert len(cleaned) == 7 and set(cleaned) == {call[1] for call in calls}


@pytest.mark.parametrize(
    "fault", ["cache", "route", "scope", "citation", "session", "answer"]
)
def test_false_success_is_rejected(simulator, fault):
    original, _, _ = simulator

    def invoke(*args, **kwargs):
        answer, citations, session = original(*args, **kwargs)
        trace = kwargs["trace"]
        if fault == "cache":
            trace["cache_hit"] = True
        elif fault == "route":
            trace["tools"] = []
        elif fault == "scope":
            trace["tools"][0]["arguments"]["product_id"] = "vivo_x500"
        elif fault == "citation":
            citations = []
        elif fault == "session":
            session = "another-session"
        else:
            answer = "无依据的回答"
        return answer, citations, session

    report = checks.run_acceptance(invoke, "week6")
    assert not report["passed"]
    assert any(row["errors"] for row in report["results"])


def test_wrong_source_and_independent_memory_pollution_are_rejected(simulator):
    invoke, _, _ = simulator
    citation = {"number": 1, "source": "x500.md", "snippet": "普通版资料"}
    assert checks.check_citations("支持[1]", [citation], "vivo_x500_pro")
    report = checks.run_acceptance(invoke, "week7", inspect=lambda *_: ["WEEK7-PRO"])
    assert not report["passed"] and "混入" in "".join(report["results"][-1]["errors"])


def test_execution_and_cleanup_failures_do_not_pass(simulator):
    invoke, _, _ = simulator

    def failed_cleanup(_):
        raise OSError("模拟不可用")

    report = checks.run_acceptance(invoke, "week6", cleanup=failed_cleanup)
    assert not report["passed"]
    assert all("清理失败" in "".join(row["errors"]) for row in report["results"])


def test_list_does_not_start_runtime(monkeypatch, capsys):
    def unavailable():
        raise AssertionError("列出场景不应加载真实服务")

    monkeypatch.setattr(checks, "runtime", unavailable)
    assert checks.main(["--list"]) == 0
    assert len(capsys.readouterr().out.splitlines()) == 10


def test_cli_writes_report_and_returns_failure_code(simulator, monkeypatch, tmp_path):
    original, _, _ = simulator

    def cached(*args, **kwargs):
        result = original(*args, **kwargs)
        kwargs["trace"]["cache_hit"] = True
        return result

    monkeypatch.setattr(checks, "runtime", lambda: (cached, lambda *_: [], None))
    report_path = tmp_path / "report.json"
    assert checks.main(["--suite", "week6", "--report", str(report_path)]) == 1
    assert json.loads(report_path.read_text(encoding="utf-8"))["passed"] is False


def test_vue_four_round_request_contract_uses_formal_api(monkeypatch):
    import sys
    from types import SimpleNamespace

    from fastapi.testclient import TestClient
    from backend.main import app

    contexts, calls = {}, []

    def invoke(query, session, product_id, *, follow_conversation, trace):
        calls.append((query, session, product_id))
        assert follow_conversation
        resolved = resolve_conversation(query, product_id, contexts.get(session, {}))
        contexts[session] = resolved.context
        trace["product_id"] = resolved.product_id
        return "WEEK7-PRO" if "备注代号是什么" in query else "演示回答", [], session

    monkeypatch.setitem(
        sys.modules, "backend.retrieval", SimpleNamespace(agent_invoke=invoke)
    )
    selected, session = "vivo_x500_pro", "vue-four-rounds"
    with TestClient(app) as client:
        for query, expected in checks.WEEK7_ROUNDS:
            response = client.post(
                "/api/chat",
                json={
                    "message": query,
                    "session_id": session,
                    "product_id": selected,
                    "follow_conversation": True,
                },
            )
            assert response.status_code == 200
            payload = response.json()
            assert set(payload) == {"answer", "citations", "session_id", "product_id"}
            assert (
                payload["product_id"] == expected and payload["session_id"] == session
            )
            selected = payload["product_id"]
    assert len(calls) == 4 and all(call[1] == session for call in calls)
