"""验证引用评估指标对不同回答情况的处理。"""

from citations import Citation
from eval.citation_eval import run_citation_eval
from eval.golden_set import GoldenQAItem


def build_citation(number: int, snippet: str = "passage") -> Citation:
    return Citation(
        number=number,
        label=f"Product Guide, PART {number}",
        snippet=snippet,
        source="https://example.com/product-guide.pdf",
    )


def test_returns_all_zero_metrics_for_an_empty_golden_set():
    def unused_answer_fn(question: str) -> tuple[str, list[Citation]]:
        raise AssertionError("answer_fn must not be called for an empty golden set")

    results = run_citation_eval([], unused_answer_fn)

    assert results.citation_validity_rate == 0.0
    assert results.answer_citation_coverage == 0.0
    assert results.golden_chunk_citation_rate == 0.0


def test_scores_a_fully_valid_single_question_answer():
    golden_set = [GoldenQAItem(question="q1", chunk_text="correct chunk")]

    def fake_answer_fn(question: str) -> tuple[str, list[Citation]]:
        return "Backups are available [1].", [build_citation(1, "correct chunk")]

    results = run_citation_eval(golden_set, fake_answer_fn)

    assert results.citation_validity_rate == 1.0
    assert results.answer_citation_coverage == 1.0
    assert results.golden_chunk_citation_rate == 1.0


def test_does_not_count_an_answer_with_no_markers_toward_coverage():
    golden_set = [GoldenQAItem(question="q1", chunk_text="correct chunk")]

    def fake_answer_fn(question: str) -> tuple[str, list[Citation]]:
        return "I cannot answer that.", [build_citation(1, "some other chunk")]

    results = run_citation_eval(golden_set, fake_answer_fn)

    assert results.answer_citation_coverage == 0.0
    assert results.golden_chunk_citation_rate == 0.0


def test_an_answer_with_no_markers_does_not_affect_the_pooled_validity_rate():
    """没有引用编号表示没有可计分的输出，不应按无效编号处理。"""
    golden_set = [
        GoldenQAItem(question="q1", chunk_text="chunk 1"),
        GoldenQAItem(question="q2", chunk_text="chunk 2"),
    ]

    def fake_answer_fn(question: str) -> tuple[str, list[Citation]]:
        if question == "q1":
            return "I cannot answer that.", [build_citation(1, "chunk 1")]
        return "Cited correctly [1].", [build_citation(1, "chunk 2")]

    results = run_citation_eval(golden_set, fake_answer_fn)

    assert results.citation_validity_rate == 1.0
    assert results.answer_citation_coverage == 0.5


def test_a_marker_pointing_at_a_number_never_retrieved_is_invalid():
    golden_set = [GoldenQAItem(question="q1", chunk_text="correct chunk")]

    def fake_answer_fn(question: str) -> tuple[str, list[Citation]]:
        return "Recovery is supported [7].", [build_citation(1, "correct chunk")]

    results = run_citation_eval(golden_set, fake_answer_fn)

    assert results.citation_validity_rate == 0.0
    assert results.answer_citation_coverage == 0.0


def test_duplicate_markers_are_each_counted_toward_the_pooled_rate():
    golden_set = [GoldenQAItem(question="q1", chunk_text="correct chunk")]

    def fake_answer_fn(question: str) -> tuple[str, list[Citation]]:
        return "First [1] and again [1].", [build_citation(1, "correct chunk")]

    results = run_citation_eval(golden_set, fake_answer_fn)

    assert results.citation_validity_rate == 1.0
    assert results.answer_citation_coverage == 1.0


def test_pools_citation_validity_across_questions_rather_than_averaging():
    golden_set = [
        GoldenQAItem(question="q1", chunk_text="chunk 1"),
        GoldenQAItem(question="q2", chunk_text="chunk 2"),
    ]

    def fake_answer_fn(question: str) -> tuple[str, list[Citation]]:
        if question == "q1":
            # 两个引用编号均有效。
            return "First [1] and second [2].", [
                build_citation(1, "chunk 1"),
                build_citation(2, "other"),
            ]
        # 一个编号有效，另一个编号不存在。
        return "Cited [1] and invented [9].", [build_citation(1, "chunk 2")]

    results = run_citation_eval(golden_set, fake_answer_fn)

    assert results.citation_validity_rate == 0.75
    assert results.answer_citation_coverage == 1.0


def test_golden_chunk_citation_rate_matches_on_exact_chunk_text():
    """直接比较 GoldenQAItem.chunk_text，沿用 page_content 完全一致的命中规则。"""
    golden_set = [GoldenQAItem(question="q1", chunk_text="the golden chunk")]

    def fake_answer_fn(question: str) -> tuple[str, list[Citation]]:
        return "Cited [1].", [build_citation(1, "a different chunk")]

    results = run_citation_eval(golden_set, fake_answer_fn)

    assert results.golden_chunk_citation_rate == 0.0


def test_golden_chunk_citation_rate_does_not_require_the_golden_chunk_be_cited():
    """按当前指标定义，检索到但未在正文引用的标准块仍计为命中。"""
    golden_set = [GoldenQAItem(question="q1", chunk_text="the golden chunk")]

    def fake_answer_fn(question: str) -> tuple[str, list[Citation]]:
        return "No markers here.", [build_citation(1, "the golden chunk")]

    results = run_citation_eval(golden_set, fake_answer_fn)

    assert results.golden_chunk_citation_rate == 1.0
    assert results.answer_citation_coverage == 0.0


def test_handles_an_answer_with_no_citations_retrieved_at_all():
    golden_set = [GoldenQAItem(question="q1", chunk_text="correct chunk")]

    def fake_answer_fn(question: str) -> tuple[str, list[Citation]]:
        return "Hello there.", []

    results = run_citation_eval(golden_set, fake_answer_fn)

    assert results.citation_validity_rate == 0.0
    assert results.answer_citation_coverage == 0.0
    assert results.golden_chunk_citation_rate == 0.0
