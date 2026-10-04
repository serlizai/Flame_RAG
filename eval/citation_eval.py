"""评估回答的引用有效性、覆盖情况与标准块命中情况。"""

import re
from dataclasses import dataclass
from typing import Callable

from citations import Citation
from eval.golden_set import GoldenQAItem

_MARKER_PATTERN = re.compile(r"\[(\d+)\]")


@dataclass
class CitationEvalResults:
    citation_validity_rate: float
    answer_citation_coverage: float
    golden_chunk_citation_rate: float


def _marker_numbers(answer: str) -> list[int]:
    return [int(match.group(1)) for match in _MARKER_PATTERN.finditer(answer)]


def run_citation_eval(
    golden_set: list[GoldenQAItem],
    answer_fn: Callable[[str], tuple[str, list[Citation]]],
) -> CitationEvalResults:
    """用标准问答集评估 answer_fn 返回的引用。

    citation_validity_rate 汇总所有问题的引用编号后计算，
    并非逐题取平均；引用编号较多的问题对最终比例影响更大。
    """
    total_markers = 0
    valid_markers = 0
    questions_with_valid_marker = 0
    questions_citing_golden_chunk = 0

    for item in golden_set:
        answer, citations = answer_fn(item.question)
        citation_numbers = {citation.number for citation in citations}

        markers = _marker_numbers(answer)
        valid_in_answer = [n for n in markers if n in citation_numbers]

        total_markers += len(markers)
        valid_markers += len(valid_in_answer)

        if valid_in_answer:
            questions_with_valid_marker += 1

        if any(citation.snippet == item.chunk_text for citation in citations):
            questions_citing_golden_chunk += 1

    num_questions = len(golden_set)

    return CitationEvalResults(
        citation_validity_rate=(
            valid_markers / total_markers if total_markers else 0.0
        ),
        answer_citation_coverage=(
            questions_with_valid_marker / num_questions if num_questions else 0.0
        ),
        golden_chunk_citation_rate=(
            questions_citing_golden_chunk / num_questions if num_questions else 0.0
        ),
    )
