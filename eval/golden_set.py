"""读取、校验和保存检索评估的标准问答集。"""

import json
from pathlib import Path

from pydantic import BaseModel


class GoldenQAItem(BaseModel):
    """标准问答条目；chunk_text 必须原样取自源库，评估按文本完全一致判定命中。"""

    question: str
    chunk_text: str


def load_golden_qa_set(path: Path) -> list[GoldenQAItem]:
    """从 JSON 文件读取并校验标准问答集。"""
    raw_items = json.loads(path.read_text())
    return [GoldenQAItem.model_validate(raw_item) for raw_item in raw_items]


def save_candidate_qa_items(items: list[GoldenQAItem], path: Path) -> None:
    """把候选问答写入 JSON 文件，供人工接受或修改。"""
    path.write_text(json.dumps([item.model_dump() for item in items], indent=2))
