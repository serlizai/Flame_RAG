"""定义 LangGraph 使用的消息、当前商品、模型调用次数和引用状态。"""

from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator

from backend.citation import Citation


class MessagesState(TypedDict):
    """同一会话的每个商品独立保存消息和引用，商品身份保持一致。"""

    messages: Annotated[list[AnyMessage], operator.add]
    product_id: str | None
    llm_calls: int
    citations: Annotated[list[Citation], operator.add]
