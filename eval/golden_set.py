import json
from pathlib import Path

from pydantic import BaseModel


class GoldenQAItem(BaseModel):
    """chunk_text must be captured verbatim from the source DB — scoring matches on it exactly, and the retriever exposes no stable chunk id."""

    question: str
    chunk_text: str


def load_golden_qa_set(path: Path) -> list[GoldenQAItem]:
    """Load and validate the golden QA set from a JSON file."""
    raw_items = json.loads(path.read_text())
    return [GoldenQAItem.model_validate(raw_item) for raw_item in raw_items]


def save_candidate_qa_items(items: list[GoldenQAItem], path: Path) -> None:
    """Write candidate QA items to a JSON file for human accept/edit review."""
    path.write_text(json.dumps([item.model_dump() for item in items], indent=2))
