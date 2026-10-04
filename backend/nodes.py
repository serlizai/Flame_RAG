"""实现语言模型调用、工具执行和最终回答节点。"""

from dataclasses import asdict
from langchain.messages import SystemMessage, ToolMessage, AIMessage, HumanMessage
from backend.state import MessagesState
from typing import Literal
from backend.tools import retrieve_docs
from backend.config import model
from backend.citation import resolve_answer_citations
from backend.prompts import SYSTEM_PROMPT, QA_PROMPT


# 定义模型可调用的工具
# --------------
tools = [retrieve_docs]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)


# 语言模型调用节点
# ---------
def llm_call(state: dict):
    """让语言模型根据当前消息决定是否调用工具。"""

    return {
        "messages": [
            model_with_tools.invoke(
                [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
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
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        tool_message = tool.invoke(tool_call)
        result.append(tool_message)
        if tool_message.artifact:
            new_citations.extend(tool_message.artifact)
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

    result = model.invoke(messages + [HumanMessage(content=qa_prompt)])

    resolved_answer, kept_citations = resolve_answer_citations(
        result.content, state.get("citations", [])
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
