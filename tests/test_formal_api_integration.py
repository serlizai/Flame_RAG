"""通过真实 Uvicorn HTTP 验收正式接口、会话兼容和首轮缓存。"""

import json
import os
import re
import subprocess
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import uuid

import pytest


@pytest.mark.skipif(
    os.getenv("RUN_AGENT_INTEGRATION") != "1",
    reason="设置 RUN_AGENT_INTEGRATION=1 后运行真实正式接口验收。",
)
def test_formal_api_http_session_and_cache(tmp_path):
    """使用随机端口和独立检查点文件，仅清理本次随机 Redis 数据。"""
    from backend.cache import RedisCache
    from backend.checkpoints import sqlite_checkpointer
    from backend.sessions import make_thread_config
    from settings import PROJECT_ROOT

    cache = RedisCache(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    cache.redis_client.ping()
    identifier = uuid.uuid4().hex
    session_id = f"flame-formal-api-{identifier}"
    cached_session = f"flame-formal-api-cache-{identifier}"
    marker = f"NOTE-{identifier}"
    query = f"如何按使用说明清洁这款商品？请检索使用说明并附引用。（验收编号 {identifier}，无需重复编号。）"
    database = tmp_path / "checkpoints.sqlite"
    log_path = tmp_path / "uvicorn.log"
    environment = os.environ.copy()
    environment["CHECKPOINT_SQLITE_PATH"] = str(database)

    with log_path.open("w", encoding="utf-8") as log:
        server = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "backend.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                "0",
            ],
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    try:
        address = None
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            output = log_path.read_text(encoding="utf-8")
            match = re.search(r"Uvicorn running on (http://127\.0\.0\.1:\d+)", output)
            if match:
                address = match.group(1)
                break
            assert server.poll() is None, output
            time.sleep(0.1)
        assert address is not None, "临时 Uvicorn 启动超时。"

        def request(method, route, payload=None):
            """走真实 TCP 请求；错误状态也解析成可断言的 JSON 响应。"""
            data = json.dumps(payload).encode("utf-8") if payload is not None else None
            message = Request(
                address + route,
                data=data,
                headers={"Content-Type": "application/json"},
                method=method,
            )
            try:
                response = urlopen(message, timeout=120)
            except HTTPError as error:
                response = error
            with response:
                return response.status, json.load(response)

        def chat(route, message, product_id="flame_kettle_k1", session=session_id):
            status, response = request(
                "POST",
                route,
                {"message": message, "product_id": product_id, "session_id": session},
            )
            assert status == 200, response
            assert response["session_id"] == session
            assert response["product_id"] == product_id
            return response

        assert request("GET", "/api/health") == (
            200,
            {"status": "ok", "service": "flame"},
        )
        status, catalog = request("GET", "/api/products")
        assert status == 200 and len(catalog) == 10
        for product in catalog:
            assert request("GET", "/api/products/" + product["id"]) == (200, product)
        assert request(
            "POST", "/api/chat", {"message": "问题", "product_id": "unknown"}
        ) == (404, {"detail": "当前商品不存在。"})
        assert request("POST", "/api/chat", {"message": " "})[0] == 422

        first = chat("/api/chat", query)
        assert "断电" in first["answer"]
        assert first["citations"] and {
            citation["source"] for citation in first["citations"]
        } == {"kettle_k1.md"}
        cached = chat("/api/chat", query, session=cached_session)
        assert (
            cached["answer"] == first["answer"]
            and cached["citations"] == first["citations"]
        )
        with sqlite_checkpointer(database) as saver:
            snapshot = saver.get_tuple(
                make_thread_config(cached_session, "flame_kettle_k1")
            )
            assert snapshot is not None
            state = snapshot.checkpoint["channel_values"]
            assert state["llm_calls"] == 0 and len(state["messages"]) == 2

        # 兼容入口写入的对话，正式入口继续使用同一个商品线程。
        chat("/query", f"我的备注代号是 {marker}，请记住。")
        memory = chat("/api/chat", "我刚才说的备注代号是什么？")
        assert marker in memory["answer"] and not memory["citations"]

        thermos = chat("/api/chat", query, product_id="flame_thermos_t1")
        assert "清洗" in thermos["answer"] or "手洗" in thermos["answer"]
        assert thermos["citations"] and {
            citation["source"] for citation in thermos["citations"]
        } == {"thermos_t1.md"}
        assert len(cache.get_chat_history(session_id).messages) == 8
        assert len(cache.get_chat_history(cached_session).messages) == 2
        print(
            "正式 HTTP 验收通过：四个正式接口、404/422、新会话缓存、旧新入口续聊和商品切换。"
        )
        print(
            f"水壶回答：{first['answer']}；保温杯回答：{thermos['answer']}；备注记忆正确。"
        )
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=5)
        for product_id in ("flame_kettle_k1", "flame_thermos_t1"):
            cache.redis_client.delete(cache.make_cache_key(query, product_id))
        cache.get_chat_history(session_id).clear()
        cache.get_chat_history(cached_session).clear()
