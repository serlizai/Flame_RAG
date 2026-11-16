"""SSE 的边界校验、最终结果、背压、断线和超时行为。"""

import asyncio
import json
import threading

from fastapi.testclient import TestClient
import httpx
from openai import RateLimitError
import pytest

from backend import main, streaming
from backend.citation import Citation
from backend.sessions import session_lock, _session_locks


def events(text):
    result = []
    for block in text.split("\n\n"):
        if block.startswith("event:"):
            name, payload = block.split("\ndata: ", 1)
            result.append((name[7:], json.loads(payload)))
    return result


def test_http_stream_uses_final_validated_answer_and_real_request(monkeypatch):
    calls = []

    def invoke(request, emit, trace):
        calls.append(request)
        trace["product_id"] = "vivo_x500"
        emit("meta", {"session_id": request.session_id, "product_id": "vivo_x500"})
        emit("delta", {"text": "初稿\n[999]"})
        return (
            "最终答案。[1]",
            [Citation(1, "资料", "原文", "x500.md")],
            request.session_id,
        )

    monkeypatch.setattr(streaming, "invoke_stream_agent", invoke)
    with TestClient(main.app) as client:
        response = client.post(
            "/api/chat/stream",
            json={
                "message": "那普通版呢？",
                "product_id": "vivo_x500_pro",
                "session_id": "stream-session",
                "follow_conversation": True,
            },
        )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-accel-buffering"] == "no"
    output = events(response.text)
    assert [name for name, _ in output] == ["status", "meta", "delta", "done"]
    assert output[2][1]["text"] == "初稿\n[999]"
    assert output[-1][1] == {
        "answer": "最终答案。[1]",
        "session_id": "stream-session",
        "product_id": "vivo_x500",
        "citations": [
            {"number": 1, "label": "资料", "snippet": "原文", "source": "x500.md"}
        ],
    }
    assert calls[0].follow_conversation and calls[0].product_id == "vivo_x500_pro"


@pytest.mark.parametrize(
    "payload,status",
    [
        ({"message": "问题", "product_id": "missing"}, 404),
        ({"message": " "}, 422),
        ({"message": "问题", "follow_conversation": "true"}, 422),
    ],
)
def test_invalid_request_never_starts_agent(monkeypatch, payload, status):
    monkeypatch.setattr(
        streaming, "invoke_stream_agent", lambda *args: pytest.fail("不应调用")
    )
    with TestClient(main.app) as client:
        assert client.post("/api/chat/stream", json=payload).status_code == status


def test_error_event_does_not_leak_exception_or_claim_completion(monkeypatch):
    def fail(*args):
        raise RuntimeError("secret-key-and-internal-address")

    monkeypatch.setattr(streaming, "invoke_stream_agent", fail)
    with TestClient(main.app) as client:
        response = client.post("/api/chat/stream", json={"message": "问题"})
    assert events(response.text)[-1][0] == "error"
    assert "secret-key" not in response.text
    assert "event: done" not in response.text


def test_rate_limit_explains_model_service_error_without_leaking_details(monkeypatch):
    def limited(*args):
        raise RateLimitError(
            "private-key-and-provider-detail",
            response=httpx.Response(
                429, request=httpx.Request("POST", "https://example.test/chat")
            ),
            body={"code": "rate_limit_exceeded", "message": "private-provider-detail"},
        )

    monkeypatch.setattr(streaming, "invoke_stream_agent", limited)
    with TestClient(main.app) as client:
        response = client.post("/api/chat/stream", json={"message": "问题"})
    output = events(response.text)
    assert output[-1][0] == "error"
    assert output[-1][1]["code"] == "model_rate_limited"
    assert "限流" in output[-1][1]["message"]
    assert "额度" in output[-1][1]["message"]
    assert "private" not in response.text
    assert "event: done" not in response.text


def test_disconnect_releases_full_queue_and_allows_worker_to_finish(monkeypatch):
    finished = threading.Event()

    def invoke(request, emit, trace):
        for _ in range(200):
            emit("delta", {"text": "内容"})
        finished.set()
        return "最终答案", [], request.session_id

    monkeypatch.setattr(streaming, "invoke_stream_agent", invoke)

    async def disconnect():
        stream = streaming.chat_events(main.QueryRequest(message="问题"))
        await anext(stream)
        await stream.aclose()

    asyncio.run(disconnect())
    assert finished.wait(3)


def test_timeout_is_terminal_and_worker_can_finish(monkeypatch):
    release = threading.Event()
    finished = threading.Event()

    def invoke(request, emit, trace):
        release.wait(3)
        emit("delta", {"text": "结束"})
        finished.set()
        return "最终答案", [], request.session_id

    monkeypatch.setattr(streaming, "invoke_stream_agent", invoke)
    monkeypatch.setattr(streaming, "STREAM_TIMEOUT", 0.01)

    async def consume():
        stream = streaming.chat_events(main.QueryRequest(message="问题"))
        await anext(stream)
        await asyncio.sleep(0.02)
        return [item async for item in stream]

    try:
        assert "event: error" in "".join(asyncio.run(consume()))
    finally:
        release.set()
    assert finished.wait(3)


def test_session_lock_serializes_threads_and_removes_idle_entries():
    entered = threading.Event()
    release = threading.Event()
    second_entered = threading.Event()

    def first():
        with session_lock("same-stream-session"):
            entered.set()
            release.wait(3)

    def second():
        with session_lock("same-stream-session"):
            second_entered.set()

    one, two = threading.Thread(target=first), threading.Thread(target=second)
    one.start()
    assert entered.wait(2)
    two.start()
    try:
        assert not second_entered.wait(0.03)
        with session_lock("other-stream-session"):
            pass
    finally:
        release.set()
        one.join(3)
        two.join(3)
    assert second_entered.is_set()
    assert "same-stream-session" not in _session_locks
