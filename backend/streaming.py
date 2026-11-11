"""以有界队列连接同步 Agent 与异步 SSE 响应。"""

import asyncio
from contextvars import ContextVar
from dataclasses import asdict, is_dataclass
import json
import logging
import queue
import threading
import time
import uuid

from openai import RateLimitError

answer_stream_enabled = ContextVar("answer_stream_enabled", default=False)
STREAM_TIMEOUT = 300
HEARTBEAT_INTERVAL = 15
logger = logging.getLogger(__name__)


def encode_event(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def invoke_stream_agent(request, emit, trace):
    # 在工作线程加载模型，HTTP 事件循环可以立即发送准备状态与心跳。
    from backend.retrieval import agent_invoke

    return agent_invoke(
        request.message,
        request.session_id,
        request.product_id,
        follow_conversation=request.follow_conversation,
        trace=trace,
        emit=emit,
    )


async def chat_events(request):
    request = request.model_copy(
        update={"session_id": request.session_id or str(uuid.uuid4())}
    )
    events = queue.Queue(maxsize=64)
    disconnected = threading.Event()

    def emit(event, data):
        while not disconnected.is_set():
            try:
                events.put((event, data), timeout=0.1)
                return
            except queue.Full:
                continue

    def run():
        trace = {}
        try:
            emit("status", {"message": "正在准备回答…"})
            answer, citations, session_id = invoke_stream_agent(request, emit, trace)
            emit(
                "done",
                {
                    "answer": answer,
                    "citations": [
                        asdict(c) if is_dataclass(c) else dict(c) for c in citations
                    ],
                    "session_id": session_id,
                    "product_id": trace.get("product_id", request.product_id),
                },
            )
        except RateLimitError:
            logger.warning("Chat stream failed (RateLimitError)")
            emit(
                "error",
                {
                    "code": "model_rate_limited",
                    "message": "聊天模型服务当前限流或额度不足，请稍后重试，或检查聊天 API 的可用额度。",
                },
            )
        except Exception as error:
            logger.warning("Chat stream failed (%s)", type(error).__name__)
            emit("error", {"message": "问答服务暂时无法完成回答，请稍后重试。"})

    worker = threading.Thread(target=run, name="flame-chat-stream", daemon=True)
    worker.start()
    started = last_sent = time.monotonic()
    try:
        while True:
            if time.monotonic() - started >= STREAM_TIMEOUT:
                yield encode_event("error", {"message": "回答超时，请稍后重试。"})
                return
            try:
                event, data = await asyncio.to_thread(events.get, True, 0.25)
            except queue.Empty:
                if time.monotonic() - last_sent >= HEARTBEAT_INTERVAL:
                    yield ": ping\n\n"
                    last_sent = time.monotonic()
                continue
            yield encode_event(event, data)
            last_sent = time.monotonic()
            if event in {"done", "error"}:
                return
    finally:
        # 断线后工作线程完成当前轮并落库，队列不再阻塞或占用增长的内存。
        disconnected.set()
