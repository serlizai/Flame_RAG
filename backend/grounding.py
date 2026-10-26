"""对明确的参数查询呈现真实工具字段，避免把否定值当缺失或遗漏来源。"""

import json
import re

from langchain_core.messages import HumanMessage, ToolMessage

from backend.citation import Citation
from backend.conversation import is_memory_question
from backend.products import load_products


def _wireless_value(citation, product):
    if citation.source == "product_business.json":
        try:
            record = json.loads(citation.snippet.split("\n", 1)[1])
        except (ValueError, IndexError):
            return None
        if not isinstance(record, dict) or record.get("product_id") != product.id:
            return None
        specifications = record.get("specifications")
        return (
            specifications.get("无线充电") if isinstance(specifications, dict) else None
        )
    if not citation.label.startswith(product.name + ", "):
        return None
    row = re.search(r"^\|\s*无线充电\s*\|\s*([^|\n]+)\|\s*$", citation.snippet, re.M)
    if row:
        return row.group(1).strip()
    # 只读取明确针对当前型号的无线充 FAQ，不从其他型号的对比段落推断。
    for block in re.finditer(
        r"^(### [^\n]+)\n+(.+?)(?=^### |\Z)", citation.snippet, re.M | re.S
    ):
        heading, body = block.groups()
        if (
            "无线充电" in heading
            and "支持" in heading
            and product.name.casefold() in heading.casefold()
            and not re.search(r"相比|区别|升级|底座|包装", heading)
        ):
            return body.strip()
    return None


def grounded_parameter_answer(question, messages, product_id):
    """仅处理单一无线充参数；事实与编号必须来自本轮已执行工具的同一条引用。"""
    if not re.search(r"无线充", question) or is_memory_question(question):
        return None
    if re.search(
        r"如何|怎么|保护壳|底座|清洗|价格|多少钱|售价|颜色|配色|退货|保修|材质|L2|缓存|处理器|有线|NFC|蓝牙|对比|区别|比较|卖点",
        question,
        re.I,
    ):
        return None
    product = load_products().get(product_id)
    if product is None:
        return None
    last_user = max(
        (i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=-1
    )
    for message in reversed(messages[last_user + 1 :]):
        if not isinstance(message, ToolMessage) or message.name not in {
            "product_info_tool",
            "product_rag_tool",
            "retrieve_product_docs",
            "retrieve_docs",
        }:
            continue
        for item in message.artifact or []:
            citation = Citation(**item) if isinstance(item, dict) else item
            value = _wireless_value(citation, product)
            if not isinstance(value, str) or not value.strip():
                continue
            value = value.strip()
            if value.startswith("不支持"):
                answer = f"根据演示商品资料，{product.name}不支持无线充电，无线充电最高功率不适用。[{citation.number}]"
            elif re.search(r"未提供|不详|未知|暂未", value):
                continue
            elif re.search(
                r"\d+(?:\.\d+)?\s*(?:W|瓦)", value, re.I
            ) or value.startswith("支持"):
                answer = f"根据演示商品资料，{product.name}支持无线充电。资料标注：{value.rstrip('。')}。[{citation.number}]"
            else:
                continue
            return answer, [citation]
    return None
