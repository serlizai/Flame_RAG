"""读取语料并筛选、抽样适合生成评估问题的文本块。"""

import random

from langchain_chroma import Chroma
from settings import PROJECT_ROOT

PRODUCT_PERSIST_DIRECTORY = str(PROJECT_ROOT / "chroma_db_products")
PRODUCT_COLLECTION_NAME = "products"

TERMINAL_PUNCTUATION = ".!?"

# 排除过短的问候语和无法独立支撑问题的文本片段。
MIN_CHUNK_LENGTH = 200


def looks_self_contained(chunk_text: str) -> bool:
    """根据首尾字符和长度筛选较完整的文本块，用于缩小候选范围，不能代替人工复核。"""
    stripped = chunk_text.strip()
    if not stripped or len(stripped) < MIN_CHUNK_LENGTH:
        return False
    starts_clean = stripped[0].isupper() or stripped[0].isdigit()
    ends_clean = stripped[-1] in TERMINAL_PUNCTUATION
    return starts_clean and ends_clean


def load_all_chunk_texts(
    persist_directory: str = PRODUCT_PERSIST_DIRECTORY,
    *,
    collection_name: str = PRODUCT_COLLECTION_NAME,
) -> list[str]:
    """默认读取商品集合；其他语料需显式指定集合，不创建空集合或加载模型。"""
    vector_store = Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=None,
        create_collection_if_not_exists=False,
    )
    return vector_store.get()["documents"]


def sample_clean_chunks(
    chunk_texts: list[str], n: int, seed: int | None = None
) -> list[str]:
    """从通过完整性筛选的文本块中抽取 n 个；候选不足时抛出异常。"""
    clean_chunks = [chunk for chunk in chunk_texts if looks_self_contained(chunk)]
    return random.Random(seed).sample(clean_chunks, n)
