"""统一工具调用的异常边界，不向回答或日志暴露参数和底层异常内容。"""

import logging

from langchain_core.messages import ToolMessage
from pydantic import ValidationError

from backend.citation import Citation
from backend.tracing import TOOL_LABELS
from backend.guardrails import has_instruction_attack

logger = logging.getLogger(__name__)


def _failure(tool_call, *, invalid_arguments=False):
    name = tool_call["name"]
    label = TOOL_LABELS.get(name)
    if label is None:
        content = "暂不支持本次查询方式，请重新描述商品问题后重试。"
    elif invalid_arguments:
        content = f"{label}参数无效，暂时无法完成查询，请重新描述问题后重试。"
    else:
        content = f"{label}暂不可用，无法确认相关信息，请稍后重试。"
    return ToolMessage(
        content=content,
        name=name,
        tool_call_id=tool_call["id"],
        status="error",
        artifact=[],
    )


def invoke_tool_safely(tool_call, tools_by_name, *, product_id, citation_start):
    """每个调用独立处理失败，仍执行同批其他工具；进程退出信号继续上抛。"""
    name = tool_call["name"]
    tool = tools_by_name.get(name)
    if tool is None:
        logger.warning("工具调用失败：未知工具")
        return _failure(tool_call)
    try:
        # 商品和引用编号只信任服务端，不能被模型参数覆盖。
        call = {
            **tool_call,
            "args": {
                **tool_call["args"],
                "product_id": product_id,
                "citation_start": citation_start,
            },
        }
        message = tool.invoke(call)
        if not isinstance(message, ToolMessage):
            raise TypeError("Invalid tool response")
        if message.status == "error":
            logger.warning(
                "工具调用失败：%s，工具返回错误状态", TOOL_LABELS.get(name, "工具")
            )
            return _failure(tool_call)
        # 在异常边界内校验引用，错误输出不能污染后续引用编号或图状态。
        artifacts = message.artifact if message.artifact is not None else []
        if not isinstance(artifacts, list):
            raise TypeError("Invalid tool citations")
        citations = []
        for item in artifacts:
            citation = Citation(**item) if isinstance(item, dict) else item
            if not isinstance(citation, Citation):
                raise TypeError("Invalid tool citation")
            if has_instruction_attack(citation.snippet) or has_instruction_attack(
                citation.label
            ):
                raise ValueError("Unsafe tool evidence")
            citations.append(citation)
        if has_instruction_attack(str(message.content)):
            raise ValueError("Unsafe tool instructions")
        return message.model_copy(
            update={"name": name, "tool_call_id": call["id"], "artifact": citations}
        )
    except ValidationError:
        logger.warning("工具调用失败：%s，参数校验失败", TOOL_LABELS.get(name, "工具"))
        return _failure(tool_call, invalid_arguments=True)
    except Exception as error:
        # 此处是外部工具的统一异常边界；不记录异常文本、堆栈或请求参数。
        logger.warning(
            "工具调用失败：%s，异常类型=%s",
            TOOL_LABELS.get(name, "工具"),
            type(error).__name__,
        )
        return _failure(tool_call)
