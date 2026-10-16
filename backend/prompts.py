"""Agent 后端使用的提示词。

提示词统一定义在顶层 prompts 模块，本模块提供后端导入入口。
"""

from prompts import PROMPT_VERSION, SYSTEM_PROMPT, QA_PROMPT, build_system_prompt

__all__ = ["PROMPT_VERSION", "SYSTEM_PROMPT", "QA_PROMPT", "build_system_prompt"]
