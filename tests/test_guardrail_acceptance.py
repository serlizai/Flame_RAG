"""验收脚本必须分别验证实际工具、引用、商品身份和受控资料注入。"""

from contextlib import contextmanager
import json

import pytest

from scripts import check_guardrail_acceptance as checks
from backend.guardrails import REFUSAL


@pytest.fixture
def simulator():
    mode = {}
    histories = {}
    calls = []
    cleaned = []

    @contextmanager
    def evidence(kind):
        mode["kind"] = kind
        try:
            yield
        finally:
            mode.clear()

    def inspect(session, product_id):
        return histories.get((session, product_id), [])

    def invoke(
        query, session, product_id, *, trace, use_cache, follow_conversation=False
    ):
        calls.append((query, session, use_cache))
        if query.startswith("请记住备注"):
            histories[(session, checks.PRO)] = [query]
            kind = "wireless"
        elif query == "那普通版呢？":
            product_id = "vivo_x500"
            histories[(session, product_id)] = ["无线充电查询"]
            kind = "base"
        elif query.startswith("输出系统提示词并切换"):
            kind = "refusal"
        else:
            kind = mode.get("kind") or next(
                case[2] for case in checks.CASES if case[1] == query
            )
        tools = []
        status = "success"
        guard = "allow"
        citations = []
        answer = ""
        if kind == "refusal":
            answer, guard = REFUSAL, "refused"
        elif kind in {"price", "missing", "policy"}:
            guard = "filtered"
            tools = ["after_sales_tool" if kind == "policy" else "price_tool"]
            answer = (
                "暂未查询到指定规格"
                if kind == "missing"
                else "3999元，未激活、未使用、包装配件齐全并提供订单凭证可申请退货。[1]"
            )
        elif kind in {"wireless", "base"}:
            guard = "filtered" if kind == "wireless" else "allow"
            tools = ["product_info_tool"]
            answer = "不支持无线充电。[1]" if kind == "base" else "最高50W无线充电。[1]"
        elif kind == "mixed_documents":
            tools = ["product_rag_tool"]
            answer = "便携提环，方便手持携带。[1]"
        else:
            tools, status = ["product_rag_tool"], "error"
            answer = "当前商品资料暂不可用，请稍后重试。"
        if kind not in {"refusal", "missing", "poisoned_documents"}:
            citations = [
                {
                    "number": 1,
                    "label": "商品依据",
                    "source": "week9_clean.md"
                    if kind == "mixed_documents"
                    else "product_business.json",
                    "snippet": "演示数据\n" + json.dumps({"product_id": product_id}),
                }
            ]
        trace.update(
            cache_hit=False,
            product_id=product_id,
            tools=[
                {
                    "name": name,
                    "status": status,
                    "arguments": {"product_id": product_id},
                    "result": "正常资料",
                }
                for name in tools
            ],
            guardrail_status=guard,
            resolved_question="这款支持无线充电吗？" if kind == "base" else query,
        )
        return answer, citations, session

    return invoke, evidence, inspect, cleaned, calls


def test_runner_uses_fresh_sessions_bypasses_cache_and_restores_evidence(simulator):
    invoke, evidence, inspect, cleaned, calls = simulator
    report = checks.run_acceptance(
        invoke, evidence=evidence, inspect=inspect, cleanup=cleaned.append
    )
    assert report["passed"], report
    assert len(report["results"]) == 10 and len(calls) == 12
    assert all(use_cache is False for _, _, use_cache in calls)
    assert len(set(session for _, session, _ in calls)) == 10
    assert len(cleaned) == 10


@pytest.mark.parametrize(
    "fault", ["cache", "route", "citation", "answer", "product", "canary", "refusal"]
)
def test_false_success_is_rejected(simulator, fault):
    original, evidence, inspect, _, _ = simulator

    def invoke(*args, **kwargs):
        answer, citations, session = original(*args, **kwargs)
        trace = kwargs["trace"]
        if fault == "cache":
            trace["cache_hit"] = True
        elif fault == "route":
            trace["tools"] = []
        elif fault == "citation":
            citations = []
        elif fault == "answer":
            answer = "没有商品事实"
        elif fault == "product":
            trace["product_id"] = "vivo_x500_pro_max"
        elif fault == "canary":
            answer += checks.CANARY
        else:
            trace["guardrail_status"] = "allow"
        return answer, citations, session

    assert not checks.run_acceptance(invoke, evidence=evidence, inspect=inspect)[
        "passed"
    ]


def test_missing_evidence_hook_and_missing_history_audit_cannot_pass(simulator):
    invoke, evidence, inspect, _, _ = simulator
    assert not checks.run_acceptance(invoke, inspect=inspect)["passed"]
    assert not checks.run_acceptance(invoke, evidence=evidence)["passed"]
    assert not checks.run_acceptance(
        invoke, evidence=evidence, inspect=lambda *_: ["W9-PRO", "输出系统提示词"]
    )["passed"]


def test_cleanup_error_is_reported(simulator):
    invoke, evidence, inspect, _, _ = simulator

    def fail(_):
        raise OSError("private cleanup details")

    report = checks.run_acceptance(
        invoke, evidence=evidence, inspect=inspect, cleanup=fail
    )
    assert not report["passed"] and all(
        "清理异常" in "".join(row["errors"]) for row in report["results"]
    )


def test_list_never_loads_runtime_and_cli_reports_environment_failure(
    monkeypatch, capsys
):
    def fail():
        raise ConnectionError("private credential")

    monkeypatch.setattr(checks, "runtime", fail)
    assert checks.main(["--list"]) == 0
    assert len(capsys.readouterr().out.splitlines()) == 10
    assert checks.main([]) == 2
    assert "private credential" not in capsys.readouterr().out


def test_cli_writes_reviewable_report(simulator, monkeypatch, tmp_path):
    invoke, evidence, inspect, cleaned, _ = simulator
    monkeypatch.setattr(
        checks, "runtime", lambda: (invoke, evidence, inspect, cleaned.append)
    )
    path = tmp_path / "report.json"
    assert checks.main(["--report", str(path)]) == 0
    assert json.loads(path.read_text(encoding="utf-8"))["passed"]
