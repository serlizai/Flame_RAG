"""统一验收不能用覆盖数量、空引用或运行失败冒充通过。"""

from contextlib import nullcontext
import json
from types import SimpleNamespace

import pytest

from scripts import check_week9_acceptance as checks


def test_manifest_covers_seven_categories_and_has_29_groups():
    rows = checks.manifest()
    assert len(rows) == 29 and len({name for _, name in rows}) == 29
    assert checks.REQUIRED_CATEGORIES <= {category for category, _ in rows}


@pytest.mark.parametrize(
    "answer,citations,valid",
    [
        ("暂未查询到相关信息。当前资料未提供L2容量。", [], True),
        ("没有找到商品卖点资料。", [], True),
        ("这款L2容量是8MB。", [], False),
        ("暂未查询到相关信息。[1]", [], False),
        ("暂未查询到相关信息。", [{"number": 1}], False),
        ("商品服务暂不可用，请稍后重试。", [], False),
        ("这是一个不错的商品。", [], False),
    ],
)
def test_no_data_validator_checks_refusal_numbers_and_sources(answer, citations, valid):
    assert (not checks.validate_no_data(answer, citations)) is valid


@pytest.mark.parametrize(
    "fault", [None, "citation", "no_tool", "success_tool", "leak", "no_hint"]
)
def test_failure_validator_requires_observed_error_and_safe_answer(fault):
    answer, citations = "商品查询暂不可用，请稍后重试。", []
    trace = {
        "tools": [{"name": "product_rag_tool", "status": "error", "result": answer}]
    }
    if fault == "citation":
        citations = [{"number": 1}]
    elif fault == "no_tool":
        trace["tools"] = []
    elif fault == "success_tool":
        trace["tools"][0]["status"] = "success"
    elif fault == "leak":
        trace["tools"][0]["result"] = "W9-PRIVATE internal path"
    elif fault == "no_hint":
        answer = "数据返回错误"
    assert (not checks.validate_failure(answer, citations, trace)) is (fault is None)


def test_unavailable_dependencies_run_all_groups_and_cannot_pass():
    cleaned = []

    def fail(*args, **kwargs):
        raise OSError("private credential and filesystem details")

    env = SimpleNamespace(
        invoke=fail,
        request=fail,
        controlled=lambda kind: nullcontext(),
        inspect=lambda *_: [],
        cache_get=lambda *_: None,
        cache_seed=lambda *_: None,
        cleanup=cleaned.append,
    )
    report = checks.run_acceptance(env)
    assert report["coverage_complete"] and not report["passed"]
    assert len(report["results"]) == len(cleaned) == 29
    assert all(not row["passed"] for row in report["results"])
    assert "private credential" not in json.dumps(report)
    assert all(session.startswith("acceptance-") for session in cleaned)


def test_missing_delegated_cases_is_incomplete_even_if_covered(monkeypatch):
    monkeypatch.setattr(
        checks,
        "run_business",
        lambda *_args, **_kwargs: {"passed": True, "results": []},
    )
    monkeypatch.setattr(
        checks,
        "run_injection",
        lambda *_args, **_kwargs: {"passed": True, "results": []},
    )

    def fail(*args, **kwargs):
        raise OSError

    env = SimpleNamespace(
        invoke=fail,
        request=fail,
        controlled=lambda kind: nullcontext(),
        cleanup=lambda *_: None,
        inspect=lambda *_: [],
        cache_seed=lambda *_: None,
        cache_get=lambda *_: None,
    )
    report = checks.run_acceptance(env)
    assert not report["coverage_complete"] and not report["passed"]


def test_list_never_loads_runtime_and_environment_failure_writes_failed_report(
    monkeypatch, capsys, tmp_path
):
    def fail():
        raise ConnectionError("private connection settings")

    monkeypatch.setattr(checks, "runtime", fail)
    assert checks.main(["--list"]) == 0
    assert len(capsys.readouterr().out.splitlines()) == 29
    path = tmp_path / "report.json"
    assert checks.main(["--report", str(path)]) == 2
    report = json.loads(path.read_text())
    assert not report["passed"] and report["status"] == "environment_not_ready"
    assert "private connection settings" not in capsys.readouterr().out


def test_account_failure_is_safe_environment_status_not_a_business_pass(
    monkeypatch, capsys, tmp_path
):
    class ProviderError(Exception):
        body = {"code": "Arrearage", "message": "private account and credentials"}

    def fail(_):
        raise ProviderError

    with pytest.raises(checks.AcceptanceEnvironmentError) as caught:
        checks.check_chat_service(SimpleNamespace(invoke=fail))
    assert caught.value.code == "Arrearage"

    def unavailable():
        raise caught.value

    monkeypatch.setattr(checks, "runtime", unavailable)
    path = tmp_path / "account-report.json"
    assert checks.main(["--report", str(path)]) == 2
    report = json.loads(path.read_text())
    assert not report["passed"] and report["error_code"] == "Arrearage"
    assert "private account" not in path.read_text() + capsys.readouterr().out


def test_preflight_does_not_publish_provider_body():
    class ProviderError(Exception):
        body = "private error body"

    def fail(_):
        raise ProviderError

    with pytest.raises(checks.AcceptanceEnvironmentError, match="chat_unavailable"):
        checks.check_chat_service(SimpleNamespace(invoke=fail))
