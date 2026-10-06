"""构建真实检索器并计算标准问答集上的检索指标。"""

from dataclasses import dataclass
from typing import Callable

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from backend.embeddings import create_embeddings

from eval.chunk_sampling import PRODUCT_COLLECTION_NAME, PRODUCT_PERSIST_DIRECTORY
from eval.golden_set import GoldenQAItem
from eval.metrics import (
    mean_reciprocal_rank,
    rank_of_correct_chunk,
    recall,
    recall_confidence_interval,
)

# 与 backend.tools.retrieve_docs 中的商品检索配置保持一致。
RETRIEVER_SEARCH_TYPE = "similarity"
RETRIEVER_SEARCH_KWARGS = {"k": 5}


@dataclass
class EvalResults:
    recall: float
    mrr: float
    ranks: list[int | None]
    recall_ci: tuple[float, float]


def build_real_retriever(
    persist_directory: str = PRODUCT_PERSIST_DIRECTORY,
) -> VectorStoreRetriever:
    """使用共享的本地 embedding 模型创建 Chroma 检索器。"""
    embeddings = create_embeddings()
    vector_store = Chroma(
        collection_name=PRODUCT_COLLECTION_NAME,
        persist_directory=persist_directory,
        embedding_function=embeddings,
        create_collection_if_not_exists=False,
    )
    return vector_store.as_retriever(
        search_type=RETRIEVER_SEARCH_TYPE, search_kwargs=RETRIEVER_SEARCH_KWARGS
    )


def run_retrieval_eval(
    golden_set: list[GoldenQAItem],
    retrieve_fn: Callable[[str], list[Document]],
) -> EvalResults:
    """用标准问答集评估检索函数，按 page_content 完全一致判定命中。"""
    ranks = [
        rank_of_correct_chunk(
            [doc.page_content for doc in retrieve_fn(item.question)],
            item.chunk_text,
        )
        for item in golden_set
    ]
    return EvalResults(
        recall=recall(ranks),
        mrr=mean_reciprocal_rank(ranks),
        ranks=ranks,
        recall_ci=recall_confidence_interval(ranks),
    )
