"""以明确意图和商品上下文判断服务范围；不确定时澄清，不调用分类模型。"""

from dataclasses import dataclass
import re
from typing import Literal

from backend.conversation import mentioned_product_ids
from backend.products import load_products

BOUNDARY_MESSAGE = (
    "我主要提供商品介绍、选购、使用和售后相关帮助。你可以告诉我想了解的商品或具体问题。"
)
CLARIFICATION_MESSAGE = (
    "你希望了解哪款商品、哪方面信息？如果需要方案，请说明是商品选购方案还是使用方案。"
)
MIXED_NOTICE = "本次仅处理商品相关部分；其他请求超出了商品服务范围。"

# 分隔独立任务，保留报价、型号及引号内作为备注或资料的数据。
_SEPARATOR = re.compile(
    r"[，,、。；;！？?\n]+|然后|另外|顺便|同时|并且|以及|"
    r"并(?=帮|请|写|查|生成|翻译|告诉)|再(?=帮|请|写|查|生成|翻译|告诉)|\band\b|\bthen\b",
    re.I,
)
_PRODUCT_REFERENCE = re.compile(
    r"商品|产品|这款|那款|这台|这件|这个型号|该型号|本机|本品|它|"
    r"(?<![a-z0-9])(?:vivo\s*)?x\d{3,4}\b|\bproduct\b",
    re.I,
)
_PRODUCT_INTENT = re.compile(
    r"价格|报价|售价|多少钱|多少元|多少块|便宜|性价比|售后|退货|退款|退换|换货|保修|质保|能退|可退|退吗|"
    r"七天.*退|7\s*天.*退|颜色|配色|参数|规格|SKU|配置|容量|功率|材质|内胆|"
    r"尺寸|重量|内存|存储|处理器|缓存|防水|充电|续航|电池|NFC|蓝牙|Wi.?Fi|"
    r"清洗|清洁|洗碗机|怎么用|如何使用|使用方法|卖点|选购|保养|故障|错误码|连接电脑|"
    r"配件|包装|库存|有货|现货|发货|运费|优惠|折扣|分期|订单|"
    r"\b(?:price|warranty|specifications?|colou?rs?|battery|charging|returns?)\b",
    re.I,
)
_RELATIVE_PRODUCT = re.compile(
    r"普通版|标准版|基础版|pro\s*(?:max)?\s*(?:版|呢|$)", re.I
)
_FOLLOWUP = re.compile(
    r"^(?:那|它|这个|还有|然后)?(?:呢|为什么|怎么回事|能详细说说吗|详细说说|"
    r"说详细一点|继续|接着说|还有吗|有什么条件|需要什么条件|分别是什么|分别有哪些|可以吗|支持吗|多少)$"
)
_GREETING = re.compile(
    r"^(?:你好|您好|嗨|哈喽|谢谢(?:你)?|多谢|好的|好|明白了|再见|"
    r"你是谁|你叫什么(?:名字)?|你能做什么|你能帮我做什么|"
    r"hello|hi|thanks|thank you|who are you)[呀啊呢吗么吧！!。.？?\s]*$",
    re.I,
)
_MEMORY = re.compile(
    r"(?:记住|记下|记录|保存|设置|更新).*(?:备注|预算|代号|偏好)|"
    r"(?:备注|预算|代号|偏好).*(?:记住|记下|记录|保存|是什么|是多少|多少|什么|是|为|叫)|"
    r"我(?:之前|刚才|上次).*(?:说了什么|问了什么|问过什么|说的是什么)|"
    r"刚才说到哪|刚才说的是什么|我问过哪些(?:商品|问题)"
)
_FORMAT = re.compile(
    r"^(?:请|麻烦|帮我|并)?(?:用中文|简短回答|简单说|说详细一点|详细说明|"
    r"列个表格|列出步骤|列成清单|附上?(?:来源)?引用|给出依据|引用商品依据|"
    r"查资料|查询资料|依据工具结果回答|依据资料回答|"
    r"无需重复编号|无需在回答中重复|不用重复编号|"
    r"只回答(?:完整)?(?:代号|备注|预算|商品名称|结论|数字)).*$"
)
_ANNOTATION_ID = re.compile(
    r"^(?:本次)?(?:验收|测试|会话)(?:编号|标识|ID)?[ ：:_-]*[a-zA-Z0-9_-]+$", re.I
)
_OFF_TOPIC = re.compile(
    r"(?:写|创作|生成|编|续写).{0,16}(?:诗|小说|故事|作文|历史论文|情书|歌词)|"
    r"(?:查(?:询)?|告诉我|播报|预测|预报).{0,16}(?:股价|股市|股票|基金行情)|"
    r"(?:今天|明天|后天|现在).{0,12}(?:天气(?:如何|怎样|怎么样)|会下雨吗|气温多少)|"
    r"(?:推荐|分析|预测|买卖|买什么|投资).{0,12}(?:股票|基金|证券|比特币)|"
    r"(?:解答?|求解|计算|做).{0,10}(?:数学题|方程|微积分|数学作业)|"
    r"(?:讲解|介绍|分析).{0,12}(?:历史事件|朝代历史|国际局势)|"
    r"(?:诊断|开处方|开药|治疗方案)|"
    r"\b(?:write|compose|generate)\b.{0,30}\b(?:poem|novel|essay|story)\b|"
    r"\b(?:weather forecast|stock price|solve the equation)\b",
    re.I,
)
_WEATHER_TASK = re.compile(
    r"(?:查(?:询)?|告诉我|播报|预测|预报).{0,80}(?:天气|气温|降雨)|"
    r"\b(?:weather forecast|what.{0,12}weather)\b",
    re.I,
)
_GENERAL_FACT = re.compile(
    r"(?:介绍|讲解|解释|分析|我想知道|告诉我).{0,20}(?:首都|总统|朝代|国际局势|历史事件|量子纠缠)|"
    r"(?:首都|总统|朝代|国际局势|历史事件|量子纠缠).{0,10}(?:是什么|是谁|在哪里|有哪些|有什么|怎么|如何)"
)
_PRODUCT_FEATURE = re.compile(r"功能|显示|支持|使用|能用|防水|耐热|接口|开启|关闭")
_PROPERTY_REQUEST = re.compile(
    r"^(?:请|帮我)?(?:介绍一下|介绍|说说|有哪些|有什么)?(?:核心)?(?:特点|优势|区别|对比).*$"
)
_ENVIRONMENT_USE = re.compile(
    r"(?:天气|高温|低温|下雨|雨天|室外|户外).{0,12}(?:能用|使用|会坏|安全吗|防水)"
)
_CONTEXT_DETAIL = re.compile(
    r"^(?:如果|假如|已经|尚未|还没|未激活|没(?:有)?(?:使用|拆封|激活)|"
    r"但是|不过|适合|能不能|可不可以|能否|是否|需要什么|需要哪些).*$"
)
_QUOTED = re.compile(r"""“[^”]*”|‘[^’]*’|"[^"]*"|(?<![a-zA-Z])'[^']*'""")
_CODE_TASK = re.compile(
    r"(?:写|生成|实现|调试|修改|开发).{0,16}(?:代码|程序|脚本|网站|Python|JavaScript|SQL)|"
    r"\b(?:write|generate|debug)\b.{0,20}\b(?:code|script|program)\b",
    re.I,
)
_DEVICE_INTERFACE = re.compile(
    r"设备接口|开发接口|连接电脑|蓝牙接口|串口|SDK|API", re.I
)


@dataclass(frozen=True)
class ScopeDecision:
    status: Literal["allow", "out_of_scope", "clarify", "mixed"]
    question: str
    message: str | None = None
    product_question: bool = False


def _clauses(question):
    """仅在引号及括号之外拆分任务，避免把备注内容当成额外指令。"""
    pairs = {"“": "”", "‘": "’", '"': '"', "'": "'", "（": "）", "(": ")"}
    closing = []
    clauses = []
    start = index = 0
    while index < len(question):
        char = question[index]
        if closing and char == closing[-1]:
            closing.pop()
        elif char in pairs and not (
            char == "'"
            and index > 0
            and question[index - 1].isascii()
            and question[index - 1].isalnum()
        ):
            closing.append(pairs[char])
        elif not closing and (match := _SEPARATOR.match(question, index)):
            if part := question[start:index].strip():
                clauses.append(part)
            index = start = match.end()
            continue
        index += 1
    if part := question[start:].strip():
        clauses.append(part)
    return clauses


def _is_annotation(text):
    """标识与输出格式可以修饰已允许的问题，括号不能藏入额外任务。"""
    if not (
        (text.startswith("（") and text.endswith("）"))
        or (text.startswith("(") and text.endswith(")"))
    ):
        return False
    parts = _clauses(text[1:-1])
    return bool(parts) and all(
        _ANNOTATION_ID.fullmatch(part) or _FORMAT.fullmatch(part) for part in parts
    )


def assess_scope(
    question, *, product_id=None, has_product_context=False, products=None
):
    """目录名称帮助识别商品；选中商品本身不能使任意无关任务变为允许。"""
    products = load_products() if products is None else products
    allowed, rejected, uncertain = [], [], []
    product_question = memory_task = False
    for clause in _clauses(question):
        text = _QUOTED.sub("[引用内容]", clause)
        reference = bool(
            _PRODUCT_REFERENCE.search(text) or mentioned_product_ids(text, products)
        )
        if (
            _OFF_TOPIC.search(text)
            or (
                _WEATHER_TASK.search(text)
                and not (reference and _PRODUCT_FEATURE.search(text))
            )
            or (_GENERAL_FACT.search(text) and not reference)
            or (
                _CODE_TASK.search(text)
                and not (reference and _DEVICE_INTERFACE.search(text))
            )
        ):
            rejected.append(clause)
            continue
        # 引号中的备注值是数据，但提及预算或备注不能掩盖额外的无关任务。
        if _MEMORY.search(text):
            allowed.append(clause)
            memory_task = True
            continue
        if _GREETING.fullmatch(clause):
            allowed.append(clause)
            continue
        if allowed and (_FORMAT.fullmatch(text) or _is_annotation(text)):
            allowed.append(clause)
            continue
        if memory_task and re.fullmatch(r"(?:请)?记住", text):
            allowed.append(clause)
            continue
        intent = bool(_PRODUCT_INTENT.search(text) or _PROPERTY_REQUEST.fullmatch(text))
        context = has_product_context or product_question
        if (
            reference
            or intent
            or _RELATIVE_PRODUCT.search(clause)
            or ((product_id or context) and _ENVIRONMENT_USE.search(text))
            or ((product_id or context) and _CONTEXT_DETAIL.fullmatch(text))
            or (context and (_FOLLOWUP.fullmatch(clause) or _FORMAT.fullmatch(clause)))
        ):
            allowed.append(clause)
            product_question = True
        else:
            uncertain.append(clause)
    if uncertain:
        return ScopeDecision("clarify", "", CLARIFICATION_MESSAGE)
    if rejected:
        if allowed and (product_question or memory_task):
            return ScopeDecision(
                "mixed", "；".join(allowed), MIXED_NOTICE, product_question
            )
        return ScopeDecision("out_of_scope", "", BOUNDARY_MESSAGE)
    if not allowed:
        return ScopeDecision("clarify", "", CLARIFICATION_MESSAGE)
    return ScopeDecision("allow", question, product_question=product_question)
