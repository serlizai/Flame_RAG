"""Agent 后端使用的提示词。

提示词统一定义在顶层 prompts 模块，本模块提供后端导入入口。
"""

from prompts import SYSTEM_PROMPT, QA_PROMPT

__all__ = ["SYSTEM_PROMPT", "QA_PROMPT"]
