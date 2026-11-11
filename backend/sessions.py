"""将前端会话和商品身份映射到独立、稳定的 LangGraph 线程。"""

import hashlib
import json
from contextlib import contextmanager
import threading

_session_locks = {}
_locks_guard = threading.Lock()


@contextmanager
def session_lock(session_id):
    """同一会话的商品切换和历史写入串行执行，空闲锁及时释放。"""
    with _locks_guard:
        lock, users = _session_locks.get(session_id, (threading.Lock(), 0))
        _session_locks[session_id] = (lock, users + 1)
    try:
        with lock:
            yield
    finally:
        with _locks_guard:
            _, users = _session_locks[session_id]
            if users == 1:
                del _session_locks[session_id]
            else:
                _session_locks[session_id] = (lock, users - 1)


def make_thread_id(session_id: str, product_id: str | None = None) -> str:
    """同一会话的每个商品各有一份状态；None 单独表示全库对话。"""
    # 使用结构化数据计算摘要，避免拼接分隔符造成不同身份共用一个线程。
    identity = json.dumps(
        {"session_id": session_id, "product_id": product_id},
        ensure_ascii=False,
        sort_keys=True,
    )
    return "flame:thread:v1:" + hashlib.sha256(identity.encode()).hexdigest()


def make_thread_config(session_id: str, product_id: str | None = None) -> dict:
    """生成调用 Agent、读取检查点和清理指定线程共用的配置。"""
    return {"configurable": {"thread_id": make_thread_id(session_id, product_id)}}
