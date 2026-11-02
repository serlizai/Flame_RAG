"""工具过程必须来自真实消息，不能把模型建议或缓存当作已执行工具。"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from backend.citation import Citation
from backend.tracing import collect_tool_trace, routing_matches


def test_trace_records_executed_tools_with_effective_identity():
    messages = [
        HumanMessage(content="多少钱？"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "price_tool",
                    "id": "price-1",
                    "args": {
                        "variant": "16+512",
                        "product_id": "wrong",
                        "api_key": "do-not-display",
                    },
                }
            ],
        ),
        ToolMessage(
            content="演示报价",
            name="price_tool",
            tool_call_id="price-1",
            artifact=[Citation(3, "商品价格", "原始业务结果", "product_business.json")],
        ),
    ]
    result = collect_tool_trace(messages, "vivo_x500_pro")
    assert result == [
        {
            "name": "price_tool",
            "arguments": {"variant": "16+512", "product_id": "vivo_x500_pro"},
            "result": "演示报价",
            "sources": ["product_business.json"],
            "status": "success",
        }
    ]


def test_suggested_tool_is_not_counted_without_execution():
    assert (
        collect_tool_trace(
            [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "price_tool", "id": "not-run", "args": {}}],
                )
            ]
        )
        == []
    )


def test_cached_or_failed_tool_execution_never_passes_routing_check():
    success = {
        "tools": [{"name": "price_tool", "status": "success"}],
        "cache_hit": False,
    }
    assert routing_matches(success, ["price_tool"])
    assert not routing_matches({**success, "cache_hit": True}, ["price_tool"])
    assert not routing_matches(
        {"tools": [{"name": "price_tool", "status": "error"}]}, ["price_tool"]
    )
    assert not routing_matches(success, ["product_rag_tool"])


def test_restored_citation_dictionaries_and_multiple_tools_are_supported():
    calls = [
        ToolMessage(
            content="ok",
            name=name,
            tool_call_id=str(i),
            artifact=[
                {"source": "product_business.json"},
                {"source": "product_business.json"},
            ],
        )
        for i, name in enumerate(["price_tool", "after_sales_tool"])
    ]
    trace = {"tools": collect_tool_trace(calls), "cache_hit": False}
    assert routing_matches(trace, ["price_tool", "after_sales_tool"])
    assert all(call["sources"] == ["product_business.json"] for call in trace["tools"])
