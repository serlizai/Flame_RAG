"""承接会话话题和商品指代，只传递问题，不跨商品复制答案、备注或工具资料。"""

from dataclasses import dataclass
import hashlib
import json
import re

from backend.products import load_products

TOPICS = {
    "wireless": (
        r"无线充",
        "这款支持无线充电吗？如果支持，最高功率是多少；如果不支持，请说明不支持且功率不适用。请引用商品依据。",
    ),
    "wired": (r"有线充|快充", "这款支持什么有线充电，最高功率是多少？"),
    "price": (r"价格|多少钱|售价|报价", "这款多少钱？"),
    "colors": (r"颜色|配色", "这款有哪些颜色？"),
    "returns": (r"退货|换货|退换|七天|7\s*天", "这款的退换货条件是什么？"),
    "warranty": (r"保修|质保", "这款的保修条件和期限是什么？"),
    "capacity": (r"容量", "这款的容量是多少？"),
    "power": (r"功率", "这款的额定功率是多少？"),
    "material": (r"材质|内胆", "这款是什么材质？"),
    "features": (r"卖点|优势|特点", "这款有哪些核心卖点？"),
    "cleaning": (r"清洗|清洁|洗碗机", "这款应该如何清洁，能放洗碗机吗？"),
    "usage": (r"使用方法|怎么用|如何使用", "这款应该如何使用？"),
    "specifications": (r"参数|配置|内存|存储", "这款有哪些基础参数和可选配置？"),
    "waterproof": (r"防水|防护等级", "这款支持什么防水或防护等级？"),
}


def conversation_key(session_id):
    return (
        "flame:conversation_context:v1:"
        + hashlib.sha256(session_id.encode()).hexdigest()
    )


def load_context(cache, session_id):
    try:
        value = json.loads(cache.get(conversation_key(session_id)) or "{}")
        if not isinstance(value, dict):
            return {}
        return {
            "product_id": value.get("product_id")
            if isinstance(value.get("product_id"), str)
            else None,
            "topics": [topic for topic in value.get("topics", []) if topic in TOPICS]
            if isinstance(value.get("topics"), list)
            else [],
            "variant": value.get("variant")[:80]
            if isinstance(value.get("variant"), str)
            else None,
        }
    except (ValueError, TypeError):
        return {}


def save_context(cache, session_id, context, ttl):
    cache.set(
        conversation_key(session_id), json.dumps(context, ensure_ascii=False), ttl=ttl
    )


def _compact(text):
    return re.sub(r"\s+", "", text).casefold()


def is_memory_question(question):
    return bool(
        re.search(
            r"我之前|我刚才|我问过|问了什么|(?:备注|预算)[^。；;！？?\n]*?(?:什么|多少)",
            question,
        )
    )


def _mentioned_products(question, products):
    candidates = []
    text = _compact(question)
    for product in products.values():
        aliases = {
            product.id,
            product.name,
            product.name.removeprefix("vivo ").removeprefix("Flame "),
        }
        if product.name.startswith("Flame "):
            aliases.add(product.name.split()[1])
        for alias in aliases:
            for match in re.finditer(
                r"(?<![a-z0-9_])" + re.escape(_compact(alias)) + r"(?![a-z_])", text
            ):
                candidates.append((match.start(), match.end(), product.id))
    accepted = []
    for start, end, product_id in sorted(
        candidates, key=lambda item: item[1] - item[0], reverse=True
    ):
        if not any(
            start < other_end and end > other_start
            for other_start, other_end, _ in accepted
        ):
            accepted.append((start, end, product_id))
    return {item[2] for item in accepted}


def _relative_product(question, anchor, products):
    if not anchor:
        return None
    family = anchor.removesuffix("_pro_max").removesuffix("_pro")
    if (
        sum(pid in products for pid in (family, family + "_pro", family + "_pro_max"))
        < 2
    ):
        return None
    text = _compact(question)
    if re.search(r"promax|max版|顶配版|max(?:呢|[？?]|$)", text):
        candidate = family + "_pro_max"
    elif re.search(r"pro版|专业版|pro(?:呢|[？?]|$)", text):
        candidate = family + "_pro"
    elif re.search(r"普通版|标准版|基础版", text):
        candidate = family
    else:
        return None
    return candidate if candidate in products else None


@dataclass(frozen=True)
class Resolution:
    product_id: str | None
    question: str
    context: dict
    source: str
    clarification: str | None = None


def resolve_conversation(question, selected_product_id, previous, products=None):
    products = products if products is not None else load_products()
    previous_id = previous.get("product_id")
    anchor = selected_product_id or (previous_id if previous_id in products else None)
    named_models = re.findall(
        r"(?<![a-z0-9_])(?:vivo)?x\d{3,4}(?:pro(?:max)?)?(?![a-z0-9_])",
        _compact(question),
    )
    known_models = {
        _compact(name)
        for product in products.values()
        for name in (product.name, product.name.removeprefix("vivo "))
    }
    if any(name not in known_models for name in named_models):
        return Resolution(
            None,
            question,
            {"product_id": None, "topics": [], "variant": None},
            "unknown_product",
            "暂未查询到这个型号，请提供商品列表中的完整商品名称。",
        )
    mentioned = _mentioned_products(question, products)
    if len(mentioned) > 1:
        return Resolution(
            None,
            question,
            {"product_id": None, "topics": [], "variant": None},
            "comparison",
        )
    relative = _relative_product(question, anchor, products)
    target = next(iter(mentioned)) if mentioned else relative or anchor
    source = (
        "explicit_name"
        if mentioned
        else "relative_name"
        if relative
        else "selected_product"
        if selected_product_id
        else "session"
    )
    memory_question = is_memory_question(question)
    topics = (
        []
        if memory_question
        else [
            key for key, (pattern, _) in TOPICS.items() if re.search(pattern, question)
        ]
    )
    if "wireless" in topics or "wired" in topics:
        topics = [topic for topic in topics if topic != "power"]
    variant_match = re.search(r"\d+\s*(?:GB)?\s*\+\s*\d+\s*(?:GB|TB)?", question, re.I)
    variant = (
        variant_match.group().strip() if "price" in topics and variant_match else None
    )
    is_followup = (
        not memory_question
        and not topics
        and bool(re.search(r"那|呢|这款|它|换成", question))
    )
    clarification = None
    resolved = question
    if (
        not mentioned
        and re.search(r"普通版|标准版|基础版|pro\s*max\s*呢|pro\s*呢", question, re.I)
        and not relative
    ):
        clarification = "请先明确商品系列或完整型号，暂时无法确定你说的是哪一款。"
    elif is_followup:
        topics = [topic for topic in previous.get("topics", []) if topic in TOPICS]
        if not target or not topics:
            clarification = "请明确要了解哪款商品、哪方面信息，例如先问‘X500 Pro 支持无线充电吗？’。"
        else:
            variant = previous.get("variant") if "price" in topics else None
            resolved = "\n".join(TOPICS[topic][1] for topic in topics)
            if variant:
                resolved += f"\n用户指定的配置为：{variant}。没有该配置时请明确说明，不报价其他配置。"
    elif not target and re.search(r"这款|它|这个商品", question):
        clarification = "请先选择商品或提供完整商品名称，再询问这款商品的信息。"
    retaining_topic = not topics and target == previous_id
    context = {
        "product_id": target,
        "topics": previous.get("topics", []) if retaining_topic else topics,
        "variant": previous.get("variant") if retaining_topic else variant,
    }
    # 只保存受支持的话题标识和用户指定配置，不保存备注、答案或检索原文。
    context["topics"] = [topic for topic in context["topics"] if topic in TOPICS]
    return Resolution(target, resolved, context, source, clarification)
