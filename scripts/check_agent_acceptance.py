"""第六七周独立验收工具；不依赖页面，始终绕过答案缓存。"""

import argparse
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import re
import uuid

from backend.tracing import routing_matches

WEEK6_CASES = [
    ("商品卖点", "这款有什么卖点？", ["product_rag_tool"], None),
    ("商品颜色", "这款有几个颜色，分别是什么？", ["product_info_tool"], "月光银"),
    ("商品报价", "16+512 是什么价格？", ["price_tool"], "3999"),
    ("七天退货", "七天能退吗，有什么条件？", ["after_sales_tool"], r"7|七"),
    (
        "混合查询",
        "这款有哪些颜色，16+512 多少钱，七天能退吗？",
        ["product_info_tool", "price_tool", "after_sales_tool"],
        "3999",
    ),
    ("不存在的规格", "12+512 是什么价格？", ["price_tool"], "暂未查询到"),
]
WEEK7_ROUNDS = [
    (
        "请记住这款商品的备注代号是 WEEK7-PRO。X500 Pro 支持无线充电吗，最高功率是多少？",
        "vivo_x500_pro",
    ),
    ("那普通版呢？", "vivo_x500"),
    ("Pro Max 呢？", "vivo_x500_pro_max"),
    ("回到 X500 Pro：我给这款商品的备注代号是什么？", "vivo_x500_pro"),
]
SOURCES = {
    "vivo_x500": "x500.md",
    "vivo_x500_pro": "x500_pro.md",
    "vivo_x500_pro_max": "x500_pro_max.md",
}


def check_citations(answer, citations, product_id):
    errors = []
    numbers = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    if not citations or numbers != {c["number"] for c in citations}:
        errors.append("回答缺少有效引用或编号不一致")
    for citation in citations:
        if citation.get("source") == "product_business.json":
            try:
                source_product = json.loads(citation["snippet"].split("\n", 1)[1])[
                    "product_id"
                ]
            except (ValueError, KeyError, IndexError, TypeError):
                source_product = None
            if source_product != product_id:
                errors.append("业务引用不属于当前商品")
        elif citation.get("source") != SOURCES.get(product_id):
            errors.append("文档引用不属于当前商品")
    return errors


def run_acceptance(invoke, suite="all", *, inspect=None, cleanup=None):
    """问答入口可注入替身；真实运行使用正式 Agent，并清理自己的独立会话。"""
    if suite not in {"all", "week6", "week7"}:
        raise ValueError("未知验收套件")
    results = []

    def call(name, question, session, selected, expected, week, index):
        trace = {}
        row = {"suite": week, "name": name, "errors": [], "trace": trace}
        errors = row["errors"]
        try:
            answer, citations, returned = invoke(
                question,
                session,
                selected,
                trace=trace,
                use_cache=False,
                follow_conversation=week == "week7",
            )
            citations = [asdict(c) if is_dataclass(c) else dict(c) for c in citations]
            row.update(answer=answer, citations=citations, trace=trace)
            if returned != session:
                errors.append("会话编号发生变化")
            if trace.get("product_id") != expected:
                errors.append("实际商品与预期不一致")
            if trace.get("cache_hit"):
                errors.append("验收不允许用缓存冒充实际执行")
            if any(
                t.get("arguments", {}).get("product_id") != expected
                for t in trace.get("tools", [])
            ):
                errors.append("工具查询了错误商品")
            if week == "week6":
                case = WEEK6_CASES[index]
                if not routing_matches(trace, case[2]):
                    errors.append("工具路由不符合预期")
                if case[3] and not re.search(case[3], answer):
                    errors.append("回答缺少预期事实")
                if index == 4 and "月光银" not in answer:
                    errors.append("混合回答遗漏商品颜色")
                if index in {3, 4} and any(
                    condition not in answer
                    for condition in ("未激活", "未使用", "包装", "配件", "订单")
                ):
                    errors.append("售后回答遗漏退货适用条件")
                if index == 5:
                    if citations or re.search(r"3999|4999", answer):
                        errors.append("不存在的配置被替代报价或附加无关引用")
                else:
                    errors.extend(check_citations(answer, citations, expected))
            else:
                steps = trace.get("steps", [])
                if (
                    not steps
                    or steps[0] != "resolve_context"
                    or steps[-1] != "final_answer"
                ):
                    errors.append("未经过上下文解析与回答流程")
                if index < 3:
                    if not trace.get("tools") or any(
                        t.get("status") != "success" for t in trace["tools"]
                    ):
                        errors.append("本轮没有成功执行工具")
                    errors.extend(check_citations(answer, citations, expected))
                    if index == 1:
                        if "不支持" not in answer or re.search(
                            r"50\s*(?:W|瓦)", answer, re.I
                        ):
                            errors.append("普通版的否定结论错误或混入 Pro 参数")
                        resolved = trace.get("resolved_question", "")
                        if "无线充电" not in resolved or "WEEK7-PRO" in resolved:
                            errors.append("追问未承接话题或带入备注")
                    elif not re.search(r"50\s*(?:W|瓦)", answer, re.I):
                        errors.append("回答未反映演示资料的无线充电参数")
                elif (
                    "WEEK7-PRO" not in answer
                    or citations
                    or trace.get("history_turns") != 1
                ):
                    errors.append("未恢复 Pro 独立备注，或个人记忆附加了商品引用")
        except Exception as error:
            errors.append(f"执行异常：{type(error).__name__}")
        results.append(row)

    def clean(session):
        if cleanup:
            try:
                cleanup(session)
            except Exception as error:
                results[-1]["errors"].append(
                    f"验收数据清理失败：{type(error).__name__}"
                )

    if suite in {"all", "week6"}:
        for index, (name, question, _, _) in enumerate(WEEK6_CASES):
            session = "acceptance-week6-" + uuid.uuid4().hex
            try:
                call(
                    name,
                    question,
                    session,
                    "vivo_x500_pro",
                    "vivo_x500_pro",
                    "week6",
                    index,
                )
            finally:
                clean(session)
    if suite in {"all", "week7"}:
        session = "acceptance-week7-" + uuid.uuid4().hex
        selected = "vivo_x500_pro"
        try:
            for index, (question, expected) in enumerate(WEEK7_ROUNDS):
                call(
                    f"多轮第{index + 1}轮",
                    question,
                    session,
                    selected,
                    expected,
                    "week7",
                    index,
                )
                # 与 Vue 一致，按服务端响应切换商品，而不是提前写入预期商品。
                selected = results[-1].get("trace", {}).get("product_id", selected)
            try:
                if inspect is None:
                    results[-1]["errors"].append("缺少商品独立历史隔离检查")
                elif any("WEEK7-PRO" in text for text in inspect(session, "vivo_x500")):
                    results[-1]["errors"].append("普通版的历史混入 Pro 备注")
            except Exception as error:
                results[-1]["errors"].append(f"历史检查失败：{type(error).__name__}")
        finally:
            clean(session)
    for row in results:
        row["passed"] = not row["errors"]
    return {"passed": all(r["passed"] for r in results), "results": results}


def runtime():
    from backend.config import cache, checkpointer
    from backend.conversation import conversation_key
    from backend.retrieval import agent, agent_invoke
    from backend.sessions import make_thread_config, make_thread_id

    def inspect(session, product_id):
        state = agent.get_state(make_thread_config(session, product_id)).values
        return [m.content for m in state.get("messages", [])]

    def cleanup(session):
        if not session.startswith("acceptance-"):
            raise ValueError("只能清理本工具创建的会话")
        cache.get_chat_history(session).clear()
        cache.redis_client.delete(conversation_key(session))
        for product_id in [None, *SOURCES]:
            checkpointer.delete_thread(make_thread_id(session, product_id))

    return agent_invoke, inspect, cleanup


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["all", "week6", "week7"], default="all")
    parser.add_argument("--list", action="store_true", help="仅列出场景，不调用模型")
    parser.add_argument(
        "--report", type=Path, help="保存真实回答、引用及执行过程到 JSON"
    )
    args = parser.parse_args(argv)
    if args.list:
        if args.suite in {"all", "week6"}:
            for name, question, tools, _ in WEEK6_CASES:
                print(f"第六周 {name}：{question} -> {', '.join(tools)}")
        if args.suite in {"all", "week7"}:
            for question, product_id in WEEK7_ROUNDS:
                print(f"第七周 {question} -> {product_id}")
        return 0
    try:
        invoke, inspect, cleanup = runtime()
        report = run_acceptance(invoke, args.suite, inspect=inspect, cleanup=cleanup)
    except Exception as error:
        print(
            f"验收未完成（{type(error).__name__}），请检查模型、本地 BGE、Redis 和运行日志。"
        )
        return 2
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    for row in report["results"]:
        print(
            f"{row['suite']} {row['name']}：" + ("；".join(row["errors"]) or "通过"),
            flush=True,
        )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
