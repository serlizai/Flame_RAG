"""Agent 后端使用的提示词。

与经典 RAG 链路共用顶层 prompts 模块中的统一定义，避免重复维护。
"""

from prompts import SYSTEM_PROMPT, QA_PROMPT

__all__ = ["SYSTEM_PROMPT", "QA_PROMPT"]
