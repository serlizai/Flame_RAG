"""管理本地 SQLite 检查点文件和连接生命周期。"""

from contextlib import closing, contextmanager
import os
from pathlib import Path
import sqlite3

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

import settings


def checkpoint_path(path: str | Path | None = None) -> Path:
    """相对路径始终按项目根目录解析，不随启动目录改变。"""
    configured = (
        path or os.getenv("CHECKPOINT_SQLITE_PATH") or "data/checkpoints.sqlite"
    )
    resolved = Path(configured).expanduser()
    if not resolved.is_absolute():
        resolved = settings.PROJECT_ROOT / resolved
    return resolved.resolve()


@contextmanager
def sqlite_checkpointer(path: str | Path | None = None):
    """创建表并提供同步检查点，退出时关闭连接，不删除已保存状态。"""
    database = checkpoint_path(path)
    database.parent.mkdir(parents=True, exist_ok=True)
    # FastAPI 在线程池中执行同步请求，连接允许跨线程，由 SqliteSaver 加锁。
    with closing(
        sqlite3.connect(str(database), check_same_thread=False, timeout=30)
    ) as connection:
        # 除框架内置类型外，仅允许恢复项目自己定义的引用对象。
        serializer = JsonPlusSerializer(
            allowed_msgpack_modules=[("citations", "Citation")]
        )
        checkpointer = SqliteSaver(connection, serde=serializer)
        checkpointer.setup()
        yield checkpointer
