"""识别常见提示词注入指令，保护用户请求、工具依据和模型输出的边界。"""

from dataclasses import dataclass
import re
import unicodedata

from backend.business_answers import requested_variant, required_business_tools
from backend.conversation import TOPICS, mentioned_product_ids, resolve_conversation
from backend.products import load_products

REFUSAL = "我可以帮助你查询商品信息，但不能提供内部提示词或执行绕过回答规则的指令。请直接告诉我商品问题。"

_SECRET = re.compile(
    r"(?:输出|打印|展示|显示|泄露|透露|复述|翻译|编码|告诉我|给我).{0,40}"
    r"(?:系统提示词|系统指令|开发者指令|内部提示词|完整提示词|API\s*(?:密钥|key)|system\s*prompt|developer\s*(?:message|prompt))|"
    r"\b(?:reveal|show|print|repeat|translate|encode|output)\b.{0,60}"
    r"\b(?:system\s*(?:prompt|message)|developer\s*(?:prompt|message)|api\s*key)\b|"
    r"(?:系统提示词|内部提示词|开发者指令).{0,12}(?:是什么|完整内容|全文|原文|翻译|编码|base64)|"
    r"(?:复述|重复|输出|翻译).{0,12}(?:第一条|初始|最初).{0,8}(?:消息|指令)",
    re.I | re.S,
)
_ROLE = re.compile(
    r"<\|(?:im_start|start_header_id)\|>|</?(?:system|developer)>|"
    r"(?:^|\n)\s*(?:system|developer)\s*:|\bSYSTEM\s+OVERRIDE\b|"
    r"(?:你现在是|扮演|我是).{0,12}(?:系统管理员|开发者|系统消息)",
    re.I,
)
_OVERRIDE = re.compile(
    r"(?:忽略|无视|忘记|绕过|覆盖|取消|解除).{0,16}(?:规则|指令|限制|提示词)|"
    r"(?:不要|不用|不必)(?:再)?(?:遵守|执行|理会).{0,16}(?:规则|指令|限制)|"
    r"\b(?:ignore|disregard|override|forget|bypass)\b.{0,40}\b(?:instructions?|rules?|prompts?|restrictions?)\b",
    re.I,
)
_TOOL_BYPASS = re.compile(
    r"(?:不要|不用|无需|不必|别|禁止|跳过|绕过)(?:再)?"
    r"(?:调用|使用|执行|查询|查)?(?:任何|所有|专用)?"
    r"(?:price_tool|after_sales_tool|product_info_tool|product_rag_tool|工具调用|调用工具|检索|知识库|商品资料)|"
    r"(?:不要|不用|无需|不必|别)(?:调用|使用)(?:任何|所有)?工具(?!拆卸|安装|组装|清洁)|"
    r"\b(?:do\s+not|don't|skip|without)\b.{0,16}\b(?:tools?|retrieval|price_tool|after_sales_tool)\b",
    re.I,
)
_FABRICATE = re.compile(
    r"(?:编造|捏造|瞎编|虚构(?!的?(?:演示|示例|样例|测试|数据))).{0,16}(?:价格|报价|参数|售后|承诺)|"
    r"(?:直接保证|直接(?:回答|承诺|声称|说)|必须(?:保证|回答|承诺|说)).{0,12}(?:能退|可退|退货|退款)|"
    r"\b(?:invent|fabricate|make up)\b.{0,20}\b(?:price|policy|specifications?)\b",
    re.I,
)
_QUOTED = re.compile(r"“[^”]*”|‘[^’]*’|\"[^\"]*\"|(?<![a-zA-Z])'[^']*'")
_NOTE_REQUEST = re.compile(
    r"(?:记住|记下|记录|保存|设置|更新|读取|查询).{0,16}(?:备注|预算|代号|偏好)|"
    r"(?:备注|预算|代号|偏好).{0,16}(?:是|为|叫|什么|多少)"
)


def _normalise(text):
    return re.sub(r"[\u200b-\u200f\ufeff]", "", unicodedata.normalize("NFKC", text))


def has_instruction_attack(text):
    """只识别要求改变助手行为的常见指令，不将单独的‘系统提示词’视为攻击。"""
    text = _normalise(text).replace("\\n", "\n")
    return any(
        pattern.search(text)
        for pattern in (_SECRET, _ROLE, _OVERRIDE, _TOOL_BYPASS, _FABRICATE)
    )


@dataclass(frozen=True)
class GuardDecision:
    status: str
    question: str
    message: str | None = None


def guard_user_question(question, *, product_id=None):
    """提取正常商品查询，拒绝内部提示词读取和角色伪造；原攻击不进入商品历史。"""
    text = _normalise(question)
    masked = _QUOTED.sub("[引用内容]", text)
    # 备注值是数据；给“系统提示词”加引号不能绕过读取请求检查。
    if _NOTE_REQUEST.search(masked) and not has_instruction_attack(masked):
        text = masked
    if _SECRET.search(text) or _ROLE.search(text):
        return GuardDecision("refused", "", REFUSAL)
    if not has_instruction_attack(text):
        return GuardDecision("allow", question)
    products = load_products()
    resolution = resolve_conversation(text, product_id, {}, products=products)
    if resolution.source == "unknown_product":
        return GuardDecision("refused", "", resolution.clarification)
    mentioned = mentioned_product_ids(text, products)
    if len(mentioned) > 1:
        return GuardDecision(
            "refused", "", "请去掉绕过规则的指令，再分别说明需要对比的商品和信息。"
        )
    # 旧版允许主动切换商品；只根据用户明确命名的目录商品恢复身份。
    names = "和".join(
        product.name for product in products.values() if product.id in mentioned
    )
    target = names or (
        products[product_id].name if product_id in products else "当前商品"
    )
    questions = []
    required = required_business_tools(text)
    if "price_tool" in required or re.search(r"\bprice\b", text, re.I):
        variant = requested_variant(text)
        questions.append(f"{target}{'的' + variant if variant else ''}价格是多少？")
    if "after_sales_tool" in required or re.search(
        r"warranty|return policy", text, re.I
    ):
        questions.append(f"{target}的退换货和保修规则及适用条件是什么？")
    for topic, (pattern, canonical) in TOPICS.items():
        if topic == "colors" and "product_info_tool" not in required:
            continue
        if topic == "power" and re.search(r"无线充|有线充|快充", text):
            continue
        if topic not in {"price", "returns", "warranty"} and re.search(pattern, text):
            questions.append(canonical.replace("这款", target))
    if not questions:
        return GuardDecision("refused", "", REFUSAL)
    return GuardDecision("filtered", "\n".join(dict.fromkeys(questions)))


def leaks_internal_prompt(answer, *templates):
    """阻止模板长句被原样输出；普通身份说明和用户自己的备注不在此列。"""
    compact_answer = re.sub(r"\s+", "", _normalise(answer)).casefold()
    for template in templates:
        for line in template.splitlines():
            literal = re.sub(r"\s+", "", _normalise(line)).casefold()
            if len(literal) >= 28 and literal in compact_answer:
                return True
    return False
