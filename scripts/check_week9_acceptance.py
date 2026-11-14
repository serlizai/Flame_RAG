"""通过正式 FastAPI 统一验收旧版第九周七类约束，保留逐轮证据。"""

import argparse
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from types import SimpleNamespace
from unittest.mock import patch
import uuid

from scripts.check_agent_acceptance import (
    WEEK6_CASES,
    check_citations,
    run_acceptance as run_business,
)
from scripts.check_guardrail_acceptance import CASES as INJECTION_CASES
from scripts.check_guardrail_acceptance import run_acceptance as run_injection

PRO = "vivo_x500_pro"


class AcceptanceEnvironmentError(RuntimeError):
    """只保留可公开的依赖错误码，不转存供应商异常正文或凭据。"""

    def __init__(self, code):
        self.code = (
            code
            if isinstance(code, str) and re.fullmatch(r"[a-zA-Z0-9_.-]{1,64}", code)
            else "chat_unavailable"
        )
        super().__init__(self.code)


def check_chat_service(model):
    """先验证一次模型可用性，账户故障不扩散为几十个业务场景失败。"""
    try:
        model.invoke("请只回复OK。")
    except Exception as error:
        body = getattr(error, "body", {})
        payload = body.get("error", body) if isinstance(body, dict) else {}
        code = payload.get("code") if isinstance(payload, dict) else None
        raise AcceptanceEnvironmentError(code) from None


REQUIRED_CATEGORIES = {
    "无资料",
    "价格",
    "售后",
    "工具异常",
    "未知商品",
    "业务范围",
    "提示词注入",
}
BUSINESS_CATEGORIES = ["基础业务", "基础业务", "价格", "售后", "价格", "价格"]
CUSTOM_CASES = [
    ("无资料", "missing_field", "未提供的L2容量"),
    ("无资料", "empty_retrieval", "检索为空不编造"),
    ("工具异常", "retrieval_recovery", "检索超时与同会话恢复"),
    ("工具异常", "business_recovery", "业务故障不沿用旧报价并恢复"),
    ("工具异常", "unknown_tool", "未知工具调用"),
    ("工具异常", "invalid_args", "工具参数错误"),
    ("未知商品", "unknown_product", "无效商品在Agent前返回404"),
    ("未知商品", "invalid_input", "空问题在Agent前返回422"),
    ("业务范围", "out_of_scope", "天气请求直接提示边界"),
    ("业务范围", "clarify", "含糊方案请求先澄清"),
    ("业务范围", "mixed", "商品与写诗混合请求"),
    ("业务范围", "scope_history", "越界后追问与备注恢复"),
    ("业务范围", "scope_cache", "越界请求不读取旧答案缓存"),
]


def manifest():
    rows = [(BUSINESS_CATEGORIES[i], case[0]) for i, case in enumerate(WEEK6_CASES)]
    rows.extend((category, name) for category, _, name in CUSTOM_CASES)
    rows.extend(("提示词注入", case[0]) for case in INJECTION_CASES)
    rows.append(("提示词注入", "拒绝后继续追问与备注隔离"))
    return rows


def _citations(values):
    return [asdict(c) if is_dataclass(c) else dict(c) for c in values]


def validate_no_data(answer, citations):
    errors = []
    if citations or re.search(r"\[\d+\]", answer):
        errors.append("无资料回答附加了引用")
    if not re.search(
        r"暂未|未(?:查询|提供|找到)|没有.{0,8}(?:资料|信息|依据)|缺少", answer
    ):
        errors.append("未明确说明缺少资料")
    if re.search(r"L2[^。；;\n]{0,20}\d+\s*(?:KB|MB|MiB|GB)", answer, re.I):
        errors.append("缺少L2资料时仍给出数值")
    return errors


def validate_failure(answer, citations, trace):
    errors = []
    if citations or re.search(r"\[\d+\]", answer):
        errors.append("工具故障回答使用了引用")
    if "重试" not in answer:
        errors.append("工具故障没有友好提示")
    if not trace.get("tools") or not any(
        t.get("status") == "error" for t in trace["tools"]
    ):
        errors.append("未观察到真实错误工具消息")
    if "W9-PRIVATE" in answer or any(
        "W9-PRIVATE" in t.get("result", "") for t in trace.get("tools", [])
    ):
        errors.append("底层异常详情进入响应")
    return errors


def run_acceptance(env, *, progress=None):
    """运行环境提供HTTP、故障、历史和缓存操作；缺少检查不能冒充通过。"""
    results = []

    def record(row):
        row["passed"] = not row["errors"]
        results.append(row)
        if progress:
            progress(row)

    business = run_business(env.invoke, "week6", cleanup=env.cleanup)
    for index, item in enumerate(business["results"]):
        category = BUSINESS_CATEGORIES[index]
        record({**item, "category": category})

    for category, kind, name in CUSTOM_CASES:
        session = "acceptance-week9-total-" + uuid.uuid4().hex
        row = {
            "category": category,
            "kind": kind,
            "name": name,
            "errors": [],
            "turns": [],
        }
        errors = row["errors"]

        def call(question, product_id=PRO, *, follow=False, use_cache=False):
            trace = {}
            answer, citations, returned = env.invoke(
                question,
                session,
                product_id,
                trace=trace,
                use_cache=use_cache,
                follow_conversation=follow,
            )
            citations = _citations(citations)
            row["turns"].append(
                {
                    "question": question,
                    "answer": answer,
                    "citations": citations,
                    "session_id": returned,
                    "product_id": trace.get("product_id"),
                    "trace": trace,
                }
            )
            if returned != session:
                errors.append("会话编号发生变化")
            if trace.get("cache_hit"):
                errors.append("验收请求意外命中答案缓存")
            if any(
                t.get("arguments", {}).get("product_id") != trace.get("product_id")
                for t in trace.get("tools", [])
            ):
                errors.append("工具商品身份与当前商品不一致")
            return answer, citations, trace

        try:
            if kind in {"missing_field", "empty_retrieval"}:
                question = "这款处理器的L2缓存容量具体是多少？请查询商品资料。"
                if kind == "empty_retrieval":
                    with env.controlled("empty_retrieval"):
                        answer, citations, trace = call("这款有什么核心卖点？")
                    if not any(
                        t["name"] == "product_rag_tool" and t["status"] == "success"
                        for t in trace.get("tools", [])
                    ):
                        errors.append("未实际执行空检索")
                else:
                    answer, citations, trace = call(question)
                    if not trace.get("tools"):
                        errors.append("缺失参数没有查询商品依据")
                errors.extend(validate_no_data(answer, citations))
            elif kind == "retrieval_recovery":
                question = f"这款有什么核心卖点？（验收编号 {session}，无需重复编号。）"
                with env.controlled("retrieval_timeout"):
                    answer, citations, trace = call(question, use_cache=True)
                errors.extend(validate_failure(answer, citations, trace))
                if env.cache_get(question, PRO) is not None:
                    errors.append("检索故障答案进入共享缓存")
                answer, citations, trace = call(question, use_cache=True)
                errors.extend(check_citations(answer, citations, PRO))
                if not trace.get("tools") or any(
                    t["status"] != "success" for t in trace["tools"]
                ):
                    errors.append("恢复后没有重新成功执行工具")
            elif kind == "business_recovery":
                question = f"16+512多少钱？（验收编号 {session}，无需重复编号。）"
                answer, citations, _ = call(question, use_cache=True)
                if "3999" not in answer:
                    errors.append("正常报价基线错误")
                errors.extend(check_citations(answer, citations, PRO))
                with env.controlled("business_unavailable"):
                    answer, citations, trace = call(question, use_cache=True)
                errors.extend(validate_failure(answer, citations, trace))
                if "3999" in answer or env.cache_get(question, PRO) is not None:
                    errors.append("业务故障使用旧报价或进入答案缓存")
                answer, citations, trace = call(question, use_cache=True)
                if "3999" not in answer or {
                    t["name"] for t in trace.get("tools", [])
                } != {"price_tool"}:
                    errors.append("恢复后未按价格工具重新报价")
                errors.extend(check_citations(answer, citations, PRO))
            elif kind in {"unknown_tool", "invalid_args"}:
                with env.controlled(kind):
                    answer, citations, trace = call("这款有什么核心卖点？")
                errors.extend(validate_failure(answer, citations, trace))
                if kind == "invalid_args" and "参数无效" not in answer:
                    errors.append("参数校验错误未给出对应提示")
            elif kind in {"unknown_product", "invalid_input"}:
                body = (
                    {
                        "message": "这款有什么特点？",
                        "product_id": "week9_nonexistent",
                        "session_id": session,
                    }
                    if kind == "unknown_product"
                    else {"message": " ", "product_id": PRO, "session_id": session}
                )
                response = env.request("POST", "/api/chat", body)
                row["response"] = response
                if response["status_code"] != (
                    404 if kind == "unknown_product" else 422
                ):
                    errors.append("非法请求状态码错误")
                if response["agent_calls"] != 0:
                    errors.append("非法请求仍调用了Agent")
            elif kind in {"out_of_scope", "clarify"}:
                answer, citations, trace = call(
                    "查询今天北京的天气"
                    if kind == "out_of_scope"
                    else "帮我写一个方案",
                    follow=True,
                )
                expected = "out_of_scope" if kind == "out_of_scope" else "clarify"
                if (
                    trace.get("scope_status") != expected
                    or trace.get("tools")
                    or citations
                ):
                    errors.append("范围拒绝或澄清行为错误")
                if any(
                    step in {"llm_call", "tool_node", "final_answer"}
                    for step in trace.get("steps", [])
                ):
                    errors.append("范围判断后仍进入模型图")
                if "商品" not in answer:
                    errors.append("范围提示缺少商品服务引导")
            elif kind == "mixed":
                answer, citations, trace = call(
                    "这款有几种颜色，16+512多少钱，七天能退吗，再写一首诗", follow=True
                )
                if (
                    trace.get("scope_status") != "mixed"
                    or "仅处理商品相关部分" not in answer
                ):
                    errors.append("混合请求缺少范围说明")
                if {t["name"] for t in trace.get("tools", [])} != {
                    "product_info_tool",
                    "price_tool",
                    "after_sales_tool",
                }:
                    errors.append("混合请求专用工具不完整")
                if any(
                    value not in answer
                    for value in (
                        "3999",
                        "月光银",
                        "未激活",
                        "未使用",
                        "包装",
                        "配件",
                        "订单",
                    )
                ):
                    errors.append("混合请求事实或适用条件不完整")
                errors.extend(check_citations(answer, citations, PRO))
                if any("写一首诗" in text for text in env.inspect(session, PRO)):
                    errors.append("被移除任务进入商品历史")
            elif kind == "scope_history":
                answer, citations, _ = call(
                    "请记住备注 W9-SCOPE。X500 Pro支持无线充电吗？", follow=True
                )
                errors.extend(check_citations(answer, citations, PRO))
                answer, citations, trace = call("查询今天北京的天气", follow=True)
                if trace.get("product_id") != PRO or trace.get("tools") or citations:
                    errors.append("越界请求改变了商品或调用工具")
                answer, citations, trace = call("那普通版呢？", follow=True)
                if (
                    trace.get("product_id") != "vivo_x500"
                    or "不支持" not in answer
                    or "无线充电" not in trace.get("resolved_question", "")
                ):
                    errors.append("越界后的追问未承接无线充电话题")
                errors.extend(check_citations(answer, citations, "vivo_x500"))
                answer, citations, trace = call(
                    "回到 X500 Pro，我给这款的备注是什么？", "vivo_x500", follow=True
                )
                if (
                    trace.get("product_id") != PRO
                    or "W9-SCOPE" not in answer
                    or citations
                ):
                    errors.append("切回Pro后备注没有恢复")
                if any(
                    "W9-SCOPE" in text for text in env.inspect(session, "vivo_x500")
                ):
                    errors.append("备注跨商品泄露")
            elif kind == "scope_cache":
                question = f"查询今天北京的天气（验收编号 {session}，无需重复编号。）"
                sentinel = "W9-INVALID-ANSWER-CACHE"
                env.cache_seed(question, PRO, session, sentinel)
                answer, citations, trace = call(question, use_cache=True)
                if (
                    sentinel in answer
                    or citations
                    or trace.get("tools")
                    or trace.get("scope_status") != "out_of_scope"
                ):
                    errors.append("越界请求读取了旧答案或未正确拒绝")
                if env.cache_get(question, PRO) != sentinel:
                    errors.append("范围拒绝覆盖了已有缓存")
        except Exception as error:
            errors.append(f"执行异常：{type(error).__name__}")
        finally:
            try:
                env.cleanup(session)
            except Exception as error:
                errors.append(f"清理异常：{type(error).__name__}")
        record(row)

    injected = run_injection(
        env.invoke,
        evidence=env.controlled,
        inspect=env.inspect,
        cleanup=env.cleanup,
        progress=lambda row: record({**row, "category": "提示词注入"}),
    )
    covered = {row["category"] for row in results}
    summary = {
        category: {
            "passed": sum(r["passed"] for r in results if r["category"] == category),
            "total": sum(r["category"] == category for r in results),
        }
        for category in sorted(covered)
    }
    complete = REQUIRED_CATEGORIES <= covered and len(results) == len(manifest())
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": complete
        and business["passed"]
        and injected["passed"]
        and all(r["passed"] for r in results),
        "coverage_complete": complete,
        "categories": summary,
        "results": results,
    }


@contextmanager
def runtime():
    """真实模型配合进程内HTTP路由；故障只替换本进程依赖，不改商品资料。"""
    from fastapi.testclient import TestClient
    from langchain_core.messages import AIMessage
    from tempfile import TemporaryDirectory
    from scripts.check_guardrail_acceptance import runtime as guard_runtime

    _, evidence, inspect, original_cleanup = guard_runtime()
    from backend import retrieval, nodes, business_tools
    from backend.config import cache, product_vector_store, model
    from backend.main import app

    check_chat_service(model)

    original_invoke = retrieval.agent_invoke
    owned_keys = {}
    with (
        TestClient(app) as client,
        TemporaryDirectory(prefix="flame-week9-") as temporary,
    ):

        def request(method, path, body=None, *, use_cache=False):
            traces = []

            def instrument(*args, **kwargs):
                actual = kwargs.setdefault("trace", {})
                traces.append(actual)
                kwargs["use_cache"] = use_cache
                return original_invoke(*args, **kwargs)

            with patch.object(retrieval, "agent_invoke", instrument):
                response = client.request(method, path, json=body)
            return {
                "status_code": response.status_code,
                "body": response.json(),
                "agent_calls": len(traces),
                "trace": traces[-1] if traces else {},
            }

        def invoke(
            question,
            session,
            product_id,
            *,
            trace,
            use_cache=False,
            follow_conversation=False,
        ):
            if use_cache and session in question:
                owned_keys.setdefault(session, set()).add(
                    cache.make_cache_key(question, product_id)
                )
            result = request(
                "POST",
                "/api/chat",
                {
                    "message": question,
                    "session_id": session,
                    "product_id": product_id,
                    "follow_conversation": follow_conversation,
                },
                use_cache=use_cache,
            )
            if result["status_code"] != 200:
                raise RuntimeError("Unexpected acceptance HTTP status")
            payload = result["body"]
            if set(payload) != {"answer", "citations", "session_id", "product_id"}:
                raise ValueError("Unexpected public response fields")
            trace.update(result["trace"])
            if payload["product_id"] != trace.get("product_id"):
                raise ValueError("Product response and trace differ")
            return payload["answer"], payload["citations"], payload["session_id"]

        @contextmanager
        def controlled(kind):
            if kind in {"mixed_documents", "poisoned_documents"}:
                with evidence(kind):
                    yield
            elif kind in {"empty_retrieval", "retrieval_timeout"}:

                def search(*args, **kwargs):
                    if kind == "retrieval_timeout":
                        raise TimeoutError(
                            "W9-PRIVATE backend credentials and internal path"
                        )
                    return []

                with patch.object(product_vector_store, "similarity_search", search):
                    yield
            elif kind == "business_unavailable":
                with patch.object(
                    business_tools,
                    "BUSINESS_DATA_PATH",
                    Path(temporary) / "missing.json",
                ):
                    yield
            elif kind in {"unknown_tool", "invalid_args"}:
                call = {
                    "id": "week9-controlled-call",
                    "name": "week9_unknown_tool"
                    if kind == "unknown_tool"
                    else "product_rag_tool",
                    "args": {}
                    if kind == "unknown_tool"
                    else {"query": {"invalid": True}},
                }
                planner = SimpleNamespace(
                    invoke=lambda messages: AIMessage(content="", tool_calls=[call])
                )
                with patch.object(nodes, "model_with_tools", planner):
                    yield
            else:
                raise ValueError("Unknown acceptance fault")

        def cache_seed(question, product_id, session, value):
            if not session.startswith("acceptance-") or session not in question:
                raise ValueError("Only seed an owned random acceptance cache")
            key = cache.make_cache_key(question, product_id)
            owned_keys.setdefault(session, set()).add(key)
            cache.set(key, value, ttl=300)

        def cleanup(session):
            original_cleanup(session)
            for key in owned_keys.pop(session, set()):
                cache.redis_client.delete(key)

        yield SimpleNamespace(
            invoke=invoke,
            request=request,
            controlled=controlled,
            inspect=inspect,
            cache_get=lambda question, product_id: cache.get(
                cache.make_cache_key(question, product_id)
            ),
            cache_seed=cache_seed,
            cleanup=cleanup,
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="仅列出全部验收场景")
    parser.add_argument(
        "--report", type=Path, help="保存逐轮回答、引用、工具过程和分组结论"
    )
    args = parser.parse_args(argv)
    if args.list:
        for category, name in manifest():
            print(f"{category}：{name}")
        return 0
    try:
        with runtime() as env:
            report = run_acceptance(
                env,
                progress=lambda row: print(
                    f"{row['category']} / {row['name']}："
                    + ("；".join(row["errors"]) or "通过"),
                    flush=True,
                ),
            )
        code = 0 if report["passed"] else 1
    except Exception as error:
        report = {
            "passed": False,
            "status": "environment_not_ready",
            "error_type": type(error).__name__,
        }
        if isinstance(error, AcceptanceEnvironmentError):
            report.update(service="chat_model", error_code=error.code)
            if error.code == "Arrearage":
                print(
                    "模型服务返回Arrearage：请恢复账户可用额度后重跑，当前不能判定业务验收通过。",
                    flush=True,
                )
        print(
            f"统一验收未完成（{type(error).__name__}），请检查Redis、模型和商品库。",
            flush=True,
        )
        code = 2
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
