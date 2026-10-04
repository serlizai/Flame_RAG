"""计算召回率、平均倒数排名及召回率置信区间。"""

import math


def rank_of_correct_chunk(
    retrieved_chunk_ids: list[str], correct_chunk_id: str
) -> int | None:
    """返回目标块在检索结果中从 1 开始的位置；未命中时返回 None。"""
    if correct_chunk_id not in retrieved_chunk_ids:
        return None
    return retrieved_chunk_ids.index(correct_chunk_id) + 1


def recall(ranks: list[int | None]) -> float:
    """计算排名列表中非 None 项所占的比例。"""
    if not ranks:
        raise ValueError("ranks must not be empty")
    return sum(1 for r in ranks if r is not None) / len(ranks)


def mean_reciprocal_rank(ranks: list[int | None]) -> float:
    """计算排名倒数的平均值，未命中的 None 按 0 处理。"""
    if not ranks:
        raise ValueError("ranks must not be empty")
    return sum(1 / r if r is not None else 0 for r in ranks) / len(ranks)


def recall_confidence_interval(ranks: list[int | None]) -> tuple[float, float]:
    """计算召回率的 95% Wilson 置信区间，在样本较少时比正态近似更可靠。"""
    z = 1.96
    n = len(ranks)
    p_hat = recall(ranks)
    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denominator
    half_width = (z / denominator) * math.sqrt(
        p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)
    )
    return center - half_width, center + half_width
