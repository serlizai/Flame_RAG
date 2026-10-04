"""定义 LangGraph 使用的消息、模型调用次数和引用状态。"""

from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator

from backend.citation import Citation


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    citations: Annotated[list[Citation], operator.add]
