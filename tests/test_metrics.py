import pytest

from eval.metrics import (
    mean_reciprocal_rank,
    rank_of_correct_chunk,
    recall,
    recall_confidence_interval,
)


def test_rank_of_correct_chunk_returns_1_indexed_position():
    retrieved = ["chunk_c", "chunk_a", "chunk_b"]

    assert rank_of_correct_chunk(retrieved, "chunk_a") == 2


def test_rank_of_correct_chunk_returns_none_when_absent():
    retrieved = ["chunk_c", "chunk_b"]

    assert rank_of_correct_chunk(retrieved, "chunk_a") is None


def test_recall_is_fraction_of_ranks_found():
    ranks = [1, None, 3, None]

    assert recall(ranks) == 0.5


def test_recall_raises_on_empty_ranks():
    with pytest.raises(ValueError):
        recall([])


def test_mean_reciprocal_rank_averages_reciprocals_treating_none_as_zero():
    ranks = [1, 4, None]

    assert mean_reciprocal_rank(ranks) == pytest.approx((1 / 1 + 1 / 4 + 0) / 3)


def test_mean_reciprocal_rank_raises_on_empty_ranks():
    with pytest.raises(ValueError):
        mean_reciprocal_rank([])


def test_recall_confidence_interval_matches_wilson_score_formula():
    ranks = [1] * 15 + [None] * 3  # 15/18 hits

    lower, upper = recall_confidence_interval(ranks)

    assert lower == pytest.approx(0.6077750036164076)
    assert upper == pytest.approx(0.9416352959953425)
