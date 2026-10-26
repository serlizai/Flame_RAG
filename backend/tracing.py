"""将本轮真实工具消息转换为可展示的诊断信息，不包含系统提示词或服务凭据。"""

from langchain_core.messages import AIMessage, ToolMessage

TOOL_LABELS = {
    "product_rag_tool": "商品知识检索",
    "product_info_tool": "商品信息查询",
    "price_tool": "价格查询",
    "after_sales_tool": "售后规则查询",
}


def collect_tool_trace(messages, product_id=None):
    """只接收本轮新消息；工具名称和输出以实际 ToolMessage 为准。"""
    calls = {
        call["id"]: call
        for message in messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    }
    trace = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        proposed = calls.get(message.tool_call_id, {}).get("args", {})
        arguments = {
            key: value
            for key, value in proposed.items()
            if key in {"query", "product_name", "variant"}
        }
        arguments["product_id"] = product_id
        sources = []
        for citation in message.artifact or []:
            source = (
                citation.get("source")
                if isinstance(citation, dict)
                else citation.source
            )
            if source and source not in sources:
                sources.append(source)
        trace.append(
            {
                "name": message.name,
                "arguments": arguments,
                "result": message.content,
                "sources": sources,
                "status": message.status,
            }
        )
    return trace


def routing_matches(trace, expected):
    """缓存不算工具执行；只判断实际路由，不自动判定回答事实是否正确。"""
    return (
        not trace.get("cache_hit", False)
        and {item["name"] for item in trace.get("tools", [])} == set(expected)
        and all(item.get("status") == "success" for item in trace.get("tools", []))
    )
