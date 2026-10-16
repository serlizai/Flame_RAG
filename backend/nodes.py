"""实现语言模型调用、工具执行和最终回答节点。"""

from dataclasses import asdict
from langchain.messages import SystemMessage, ToolMessage, AIMessage, HumanMessage
from backend.state import MessagesState
from typing import Literal
from backend.tools import retrieve_product_docs
from backend.config import model
from backend.citation import Citation, resolve_answer_citations
from backend.prompts import QA_PROMPT, build_system_prompt
from backend.products import load_products


# 定义模型可调用的工具
# --------------
tools = [retrieve_product_docs]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)

# 旧版会话可能保存了尚未执行的旧名称调用，模型只绑定上面的新工具。
_LEGACY_RETRIEVAL_TOOL_NAME = "retrieve_docs"


def _messages_for_model(messages):
    """复制并统一旧会话中的工具名称，SQLite 原始历史继续保留。"""
    converted = []
    for message in messages:
        if isinstance(message, AIMessage) and (
            any(
                call["name"] == _LEGACY_RETRIEVAL_TOOL_NAME
                for call in message.tool_calls
            )
            or any(
                call.get("function", {}).get("name") == _LEGACY_RETRIEVAL_TOOL_NAME
                for call in message.additional_kwargs.get("tool_calls", [])
            )
        ):
            message = message.model_copy(deep=True)
            for call in message.tool_calls:
                if call["name"] == _LEGACY_RETRIEVAL_TOOL_NAME:
                    call["name"] = retrieve_product_docs.name
            # 同步保留在附加字段中的原始服务商格式，避免混入旧名称。
            for call in message.additional_kwargs.get("tool_calls", []):
                if call.get("function", {}).get("name") == _LEGACY_RETRIEVAL_TOOL_NAME:
                    call["function"]["name"] = retrieve_product_docs.name
        elif (
            isinstance(message, ToolMessage)
            and message.name == _LEGACY_RETRIEVAL_TOOL_NAME
        ):
            message = message.model_copy(update={"name": retrieve_product_docs.name})
        converted.append(message)
    return converted


def _context_citations(messages):
    """从当前商品的工具结果和缓存回答中收集可用引用。"""
    citations = []
    for message in messages:
        if isinstance(message, ToolMessage) and message.artifact:
            # 检查点恢复 ToolMessage 时，artifact 中的数据类可能变成字典。
            citations.extend(
                Citation(**citation) if isinstance(citation, dict) else citation
                for citation in message.artifact
            )
        elif isinstance(message, AIMessage):
            citations.extend(
                Citation(**citation)
                for citation in message.additional_kwargs.get("citations", [])
            )
    return citations


def _system_message(state: dict):
    """补充当前商品身份，使模型能理解“这款”，检索范围仍由后端控制。"""
    product_id = state.get("product_id")
    if product_id is not None:
        product = load_products()[product_id]
        prompt = build_system_prompt(product.name, product.id)
    else:
        prompt = build_system_prompt()
    return SystemMessage(content=prompt)


# 语言模型调用节点
# ---------
def llm_call(state: dict):
    """让语言模型根据当前消息决定是否调用工具。"""

    return {
        "messages": [
            model_with_tools.invoke(
                [_system_message(state)] + _messages_for_model(state["messages"])
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# 工具执行节点
# -----------
def tool_node(state: dict):
    """执行模型发出的工具调用，并收集工具结果与引用。"""

    result = []
    new_citations = []
    citation_start = (
        max((c.number for c in _context_citations(state["messages"])), default=0) + 1
    )
    for tool_call in state["messages"][-1].tool_calls:
        if tool_call["name"] == _LEGACY_RETRIEVAL_TOOL_NAME:
            tool_call = {**tool_call, "name": retrieve_product_docs.name}
        tool = tools_by_name[tool_call["name"]]
        # 覆盖模型可能提交的同名参数，商品范围只信任服务端 State。
        if tool.name == retrieve_product_docs.name:
            tool_call = {
                **tool_call,
                "args": {
                    **tool_call["args"],
                    "product_id": state.get("product_id"),
                    "citation_start": citation_start,
                },
            }
        tool_message = tool.invoke(tool_call)
        result.append(tool_message)
        if tool_message.artifact:
            new_citations.extend(tool_message.artifact)
            citation_start += len(tool_message.artifact)
    return {"messages": result, "citations": new_citations}


# 最终回答节点
# ------------------


def final_answer(state: dict):
    """工具循环结束后生成最终回答，并校验和整理引用编号。"""

    messages = state["messages"]

    context = "\n".join(m.content for m in messages if isinstance(m, ToolMessage))
    question = next(
        m.content for m in reversed(messages) if isinstance(m, HumanMessage)
    )

    qa_prompt = QA_PROMPT.format(context=context, input=question)

    result = model.invoke(
        [_system_message(state)]
        + _messages_for_model(messages)
        + [HumanMessage(content=qa_prompt)]
    )

    resolved_answer, kept_citations = resolve_answer_citations(
        result.content, _context_citations(messages)
    )
    sources = "\n".join(f"[{c.number}] {c.label}" for c in kept_citations)
    content = (
        f"{resolved_answer}\n\nSources:\n{sources}"
        if kept_citations
        else resolved_answer
    )

    return {
        "messages": [
            AIMessage(
                content=content,
                additional_kwargs={"citations": [asdict(c) for c in kept_citations]},
            )
        ]
    }


def should_continue(state: MessagesState) -> Literal["tool_node", "final_answer"]:
    """根据最后一条模型消息是否包含工具调用，选择工具节点或最终回答节点。"""

    messages = state["messages"]
    last_message = messages[-1]

    # 模型发出工具调用时，继续进入工具执行节点
    if last_message.tool_calls:
        return "tool_node"

    return "final_answer"
