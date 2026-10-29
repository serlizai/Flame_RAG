"""约束明确业务查询的工具覆盖，按本轮真实业务字段呈现答案与引用。"""

import json
import re
import uuid

from langchain_core.messages import HumanMessage, ToolMessage

from backend.citation import Citation
from backend.conversation import is_memory_question, resolve_conversation
from backend.products import load_products

INFO = "product_info_tool"
PRICE = "price_tool"
AFTER_SALES = "after_sales_tool"


def current_turn(messages):
    index = max(
        (i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=-1
    )
    return (messages[index].content, messages[index + 1 :]) if index >= 0 else ("", [])


def required_business_tools(question):
    if is_memory_question(question):
        return ()
    # “不要报价，只查颜色”不产生价格请求；“不要用工具编造价格”仍须查价格。
    text = re.sub(
        r"(?:不需要|不用|不要|别)(?:查询|提供|告诉我|给我|查)?(?:价格|报价|颜色|配色|售后)[^，,。；;！？?\n]*",
        "",
        question,
    )
    return tuple(
        name
        for name, pattern in (
            (INFO, r"颜色|配色|几种色"),
            (PRICE, r"价格|报价|售价|多少钱|多少元|多少块"),
            (AFTER_SALES, r"售后|退货|退款|退换|换货|保修|质保|七天.*退|7\s*天.*退"),
        )
        if re.search(pattern, text)
    )


def business_product(question, product_id):
    resolution = resolve_conversation(question, product_id, {})
    if resolution.source in {"unknown_product", "comparison"}:
        return None
    # 显式商品范围仍由 State 控制；全库模式只接受一个明确命名的商品。
    target = product_id or (
        resolution.product_id if resolution.source == "explicit_name" else None
    )
    return load_products().get(target)


def focused_business_question(question):
    """含知识解释、其他参数或用户记忆时仍由原回答流程综合处理。"""
    return not is_memory_question(question) and not re.search(
        r"卖点|特点|优势|原理|清洗|清洁|使用方法|怎么用|如何使用|无线|有线|容量|功率|尺寸|材质|处理器|缓存|L2|防水|电池|续航|包装|对比|比较|区别|备注|预算|参数|有哪些配置|配置有哪些|可选配置|库存|有货|现货|发货|运费|优惠|折扣|活动|分期|订单",
        question,
        re.I,
    )


def requested_variant(question):
    match = re.search(
        r"\d+\s*(?:GB)?\s*(?:RAM)?\s*\+\s*\d+\s*(?:GB|TB)?(?:\s*ROM)?", question, re.I
    )
    if match:
        return match.group().strip()
    match = re.search(r"SKU\s*[:：]?\s*(\d+)", question, re.I)
    return match.group(1) if match else None


def complete_business_calls(question, product, proposed, completed, *, focused):
    """保留模型提出的有效调用，补上遗漏的业务工具，不依赖价格记录中的颜色字段。"""
    required = required_business_tools(question)
    if not product or not required:
        return proposed
    result, seen = [], set()
    for call in proposed:
        name = call["name"]
        if name not in required:
            if not focused:
                result.append(call)
            continue
        if name in completed or name in seen:
            continue
        args = {"product_name": product.name}
        if name == PRICE:
            variant = requested_variant(question)
            proposed_variant = call.get("args", {}).get("variant")
            if variant:
                args["variant"] = variant
            elif (
                isinstance(proposed_variant, str)
                and proposed_variant.strip()
                and proposed_variant in question
            ):
                args["variant"] = proposed_variant
        result.append({**call, "args": args})
        seen.add(name)
    for name in required:
        if name in completed or name in seen:
            continue
        args = {"product_name": product.name}
        if name == PRICE and (variant := requested_variant(question)):
            args["variant"] = variant
        result.append(
            {
                "id": "business-" + uuid.uuid4().hex,
                "name": name,
                "args": args,
                "type": "tool_call",
            }
        )
    return result


def _business_record(message, product_id):
    if message.status != "success":
        return None
    for item in message.artifact or []:
        citation = Citation(**item) if isinstance(item, dict) else item
        if citation.source != "product_business.json":
            continue
        try:
            data = json.loads(citation.snippet.split("\n", 1)[1])
        except (ValueError, IndexError, TypeError):
            continue
        if (
            isinstance(data, dict)
            and data.get("product_id") == product_id
            and data.get("demo") is True
        ):
            return data, citation
    return None


def _policy_excerpt(question, policy):
    # Markdown 政策按完整规则行保留条件；段落型政策保留原文，避免截断适用条件。
    lines = [line for line in policy.splitlines() if line.startswith("- **")]
    if not lines or re.search(r"售后|流程", question):
        return policy.strip()
    labels = []
    if re.search(r"退货|退款|退换|七天.*退|7\s*天.*退", question):
        labels.append("退货")
    if re.search(r"换货|退换", question):
        labels.append("换货")
    if re.search(r"保修|质保", question):
        labels.extend(["保修", "服务范围"])
    matched = [
        line
        for line in lines
        if any(label in line.split("**", 2)[1] for label in labels)
    ]
    return "\n".join(matched) if matched else policy.strip()


def grounded_business_answer(question, messages, product_id):
    """只处理明确业务字段；数据、金额、条件、引用均取自本轮对应工具。"""
    required = required_business_tools(question)
    product = business_product(question, product_id)
    if not required or not product or not focused_business_question(question):
        return None
    _, turn = current_turn(messages)
    tool_messages = [m for m in turn if isinstance(m, ToolMessage)]
    if any(m.name not in required for m in tool_messages):
        return None
    latest = {m.name: m for m in tool_messages}
    if not set(required) <= latest.keys():
        return None
    parts, citations = [], []
    for name in required:
        message = latest[name]
        record = _business_record(message, product.id)
        if record is None:
            if name == PRICE and str(message.content).startswith("暂未查询到指定规格"):
                parts.append("暂未查询到指定规格的价格，不能用其他规格的价格代替。")
            else:
                label = {INFO: "颜色", PRICE: "价格", AFTER_SALES: "售后"}[name]
                parts.append(f"暂未查询到当前商品可靠的{label}信息，请稍后重试。")
            continue
        data, citation = record
        marker = f"[{citation.number}]"
        if name == INFO:
            colors = data.get("colors")
            if (
                not isinstance(colors, list)
                or not colors
                or any(not isinstance(c, str) or not c.strip() for c in colors)
            ):
                parts.append("暂未查询到当前商品的颜色信息。")
                continue
            parts.append(
                f"{product.name}共有{len(colors)}种颜色：{'、'.join(colors)}。{marker}"
            )
        elif name == PRICE:
            variants = data.get("variants")
            if (
                data.get("currency") != "CNY"
                or not isinstance(variants, list)
                or not variants
                or any(
                    not isinstance(v, dict)
                    or type(v.get("price_fen")) is not int
                    or v["price_fen"] < 0
                    or not all(
                        isinstance(v.get(key), str) and v[key].strip()
                        for key in ("specification", "color", "sku")
                    )
                    for v in variants
                )
            ):
                parts.append("暂未查询到当前商品可靠的价格信息。")
                continue
            values = []
            for variant in variants:
                amount = variant["price_fen"]
                yuan = (
                    str(amount // 100)
                    if amount % 100 == 0
                    else f"{amount // 100}.{amount % 100:02d}"
                )
                values.append(
                    f"{variant['specification']}，{variant['color']}，SKU {variant['sku']}，演示价格 {yuan} 元"
                )
            parts.extend(f"{value}。{marker}" for value in values)
        else:
            policy = data.get("policy")
            if not isinstance(policy, str) or not policy.strip():
                parts.append("暂未查询到当前商品的售后规则。")
                continue
            parts.append(f"演示售后规则：\n{_policy_excerpt(question, policy)}{marker}")
        citations.append(citation)
    return "\n\n".join(parts), citations
