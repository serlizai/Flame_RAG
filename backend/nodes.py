"""实现语言模型调用、工具执行和最终回答节点。"""

from dataclasses import asdict
from langchain.messages import SystemMessage, ToolMessage, AIMessage, HumanMessage
from backend.state import MessagesState
from typing import Literal
import re
from backend.tools import product_rag_tool
from backend.business_tools import product_info_tool, price_tool, after_sales_tool
from backend.config import model
from backend.citation import Citation, resolve_answer_citations
from backend.prompts import QA_PROMPT, build_system_prompt
from backend.products import load_products
from backend.conversation import is_memory_question
from backend.grounding import grounded_feature_excerpt, grounded_parameter_answer
from backend.business_answers import (
    business_product,
    complete_business_calls,
    current_turn,
    focused_business_question,
    grounded_business_answer,
    required_business_tools,
)


# 定义模型可调用的工具
# --------------
tools = [product_rag_tool, product_info_tool, price_tool, after_sales_tool]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)

# 旧版会话可能保存了尚未执行的旧名称调用，模型只绑定上面的新工具。
_LEGACY_RETRIEVAL_TOOL_NAMES = {"retrieve_docs", "retrieve_product_docs"}


def _messages_for_model(messages):
    """复制并统一旧会话中的工具名称，SQLite 原始历史继续保留。"""
    converted = []
    for message in messages:
        if isinstance(message, AIMessage) and (
            any(
                call["name"] in _LEGACY_RETRIEVAL_TOOL_NAMES
                for call in message.tool_calls
            )
            or any(
                call.get("function", {}).get("name") in _LEGACY_RETRIEVAL_TOOL_NAMES
                for call in message.additional_kwargs.get("tool_calls", [])
            )
        ):
            message = message.model_copy(deep=True)
            for call in message.tool_calls:
                if call["name"] in _LEGACY_RETRIEVAL_TOOL_NAMES:
                    call["name"] = product_rag_tool.name
            # 同步保留在附加字段中的原始服务商格式，避免混入旧名称。
            for call in message.additional_kwargs.get("tool_calls", []):
                if call.get("function", {}).get("name") in _LEGACY_RETRIEVAL_TOOL_NAMES:
                    call["function"]["name"] = product_rag_tool.name
        elif (
            isinstance(message, ToolMessage)
            and message.name in _LEGACY_RETRIEVAL_TOOL_NAMES
        ):
            message = message.model_copy(update={"name": product_rag_tool.name})
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
    """模型选择工具，程序补齐明确业务请求中遗漏的专用工具。"""
    question, turn = current_turn(state["messages"])
    required = required_business_tools(question)
    product = business_product(question, state.get("product_id")) if required else None
    focused = focused_business_question(question)
    completed = {m.name for m in turn if isinstance(m, ToolMessage)}
    if product and required and focused and set(required) <= completed:
        # 所需业务工具已真实执行，进入字段呈现；不让模型重复调用或改用其他来源。
        return {"messages": [AIMessage(content="")]}
    response = model_with_tools.invoke(
        [_system_message(state)] + _messages_for_model(state["messages"])
    )
    calls = complete_business_calls(
        question, product, response.tool_calls, completed, focused=focused
    )
    if calls != response.tool_calls:
        response = AIMessage(
            content="" if focused else response.content, tool_calls=calls
        )
    return {
        "messages": [response],
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
        if tool_call["name"] in _LEGACY_RETRIEVAL_TOOL_NAMES:
            tool_call = {**tool_call, "name": product_rag_tool.name}
        tool = tools_by_name[tool_call["name"]]
        # 覆盖模型可能提交的同名参数，商品范围只信任服务端 State。
        if tool.name in tools_by_name:
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

    last_user = max(
        i for i, message in enumerate(messages) if isinstance(message, HumanMessage)
    )
    grounded = grounded_business_answer(question, messages, state.get("product_id"))
    if grounded is None:
        grounded = grounded_parameter_answer(
            question, messages, state.get("product_id")
        )

    qa_prompt = QA_PROMPT.format(context=context, input=question)

    result = (
        AIMessage(content=grounded[0])
        if grounded
        else model.invoke(
            [_system_message(state)]
            + _messages_for_model(messages)
            + [HumanMessage(content=qa_prompt)]
        )
    )

    available_citations = _context_citations(messages)
    resolved_answer, kept_citations = resolve_answer_citations(
        result.content, available_citations
    )
    if is_memory_question(question):
        resolved_answer, kept_citations = resolve_answer_citations(result.content, [])
    current_evidence = any(
        isinstance(message, ToolMessage) and message.artifact
        for message in messages[last_user + 1 :]
    )
    if current_evidence and not kept_citations:
        excerpt = grounded_feature_excerpt(question, messages, state.get("product_id"))
        if excerpt:
            resolved_answer, kept_citations = resolve_answer_citations(
                excerpt[0], excerpt[1]
            )
    missing_information = bool(
        re.match(
            r"^[\s*]*(?:暂未查询到|未查询到|没有找到|当前资料未提供)", resolved_answer
        )
    )
    if (
        current_evidence
        and available_citations
        and not kept_citations
        and not missing_information
        and not is_memory_question(question)
    ):
        # 只重试一次，不把所有检索结果自动当成答案引用。
        repair = model.invoke(
            [
                _system_message(state),
                HumanMessage(
                    content=qa_prompt
                    + "\n\n上次回答没有可核验的引用编号。请重新依据上方工具结果回答，只回答用户所问。"
                    "每个已证实的商品事实后必须写实际存在的 [编号]；没有依据、备注或资料缺失说明不引用。"
                    "不要补充未询问的包装或其他参数，不得编造编号。"
                ),
            ]
        )
        resolved_answer, kept_citations = resolve_answer_citations(
            repair.content, available_citations
        )
        if not kept_citations and not re.match(
            r"^[\s*]*(?:暂未查询到|未查询到|没有找到|当前资料未提供)", resolved_answer
        ):
            resolved_answer = (
                "暂未生成带有效引用的回答，请重试或查看本轮工具返回的原文。"
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
