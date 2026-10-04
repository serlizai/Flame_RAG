"""统一应用、建库脚本和评估工具使用的环境配置与数据路径。"""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

_database_path = Path(os.getenv("CHROMA_PERSIST_DIRECTORY", "chroma_db")).expanduser()
if not _database_path.is_absolute():
    _database_path = PROJECT_ROOT / _database_path
CHROMA_PERSIST_DIRECTORY = str(_database_path.resolve())
