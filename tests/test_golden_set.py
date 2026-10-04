"""验证标准问答集的字段校验和读写。"""

import json

import pytest
from pydantic import ValidationError

from eval.golden_set import GoldenQAItem, load_golden_qa_set, save_candidate_qa_items


def test_loads_valid_items(tmp_path):
    path = tmp_path / "golden_qa.json"
    path.write_text(
        json.dumps(
            [
                {
                    "question": "What does Flame use to retrieve documents?",
                    "chunk_text": "Flame uses a vector database to retrieve documents.",
                }
            ]
        )
    )

    items = load_golden_qa_set(path)

    assert len(items) == 1
    assert items[0].question == "What does Flame use to retrieve documents?"
    assert items[0].chunk_text == "Flame uses a vector database to retrieve documents."


def test_raises_on_item_missing_required_field(tmp_path):
    path = tmp_path / "golden_qa.json"
    path.write_text(json.dumps([{"question": "What does Flame use?"}]))

    with pytest.raises(ValidationError):
        load_golden_qa_set(path)


def test_save_then_load_round_trips(tmp_path):
    path = tmp_path / "candidates.json"
    items = [
        GoldenQAItem(
            question="What does Flame use?", chunk_text="Flame uses a vector database."
        )
    ]

    save_candidate_qa_items(items, path)

    assert load_golden_qa_set(path) == items
