"""检查 .env 中的聊天服务及一次完整工具调用，不加载 BGE 模型。"""

import os

from settings import PROJECT_ROOT  # noqa: F401 -- loads the project's .env


def main():
    required = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL")
    missing = [name for name in required if not os.getenv(name, "").strip()]
    if missing:
        print("请先在项目 .env 中填写：" + ", ".join(missing))
        return 2

    from langchain.chat_models import init_chat_model
    from langchain_core.messages import HumanMessage, ToolMessage

    probe = {
        "type": "function",
        "function": {
            "name": "runtime_probe",
            "description": "检查聊天服务的工具调用是否可用。",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
        },
    }
    try:
        model = init_chat_model(
            model=os.environ["OPENAI_MODEL"],
            model_provider="openai",
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ["OPENAI_BASE_URL"],
            timeout=30,
            max_retries=0,
        ).bind_tools([probe])
        messages = [
            HumanMessage(
                content="请调用 runtime_probe 工具，value 为 ok；收到工具结果后回复 OK。"
            )
        ]
        response = model.invoke(messages)
        calls = response.tool_calls
        if not calls or any(call["name"] != "runtime_probe" for call in calls):
            print(
                "聊天服务可连接，但本次未返回预期工具调用，请检查模型是否支持工具调用。"
            )
            return 1
        messages.append(response)
        messages.extend(
            ToolMessage(content="ok", tool_call_id=call["id"]) for call in calls
        )
        final = model.invoke(messages)
        if not final.content or final.tool_calls:
            print("工具调用已返回，但模型未生成最终文本，请检查服务的工具消息兼容性。")
            return 1
    except Exception as error:
        # Avoid printing credentials, request headers, or provider response bodies.
        status = getattr(error, "status_code", None)
        print(f"聊天 API 检查失败：{type(error).__name__}，HTTP 状态：{status or '无'}")
        return 1
    print("聊天 API 连接、模型工具调用及工具结果回传：通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
