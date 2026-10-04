"""读取语料并筛选、抽样适合生成评估问题的文本块。"""

import random

from langchain_chroma import Chroma
from settings import CHROMA_PERSIST_DIRECTORY

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
    persist_directory: str = CHROMA_PERSIST_DIRECTORY,
) -> list[str]:
    """读取库中已有文本，不加载 embedding 模型，也不调用聊天 API。"""
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=None)
    return vector_store.get()["documents"]


def sample_clean_chunks(
    chunk_texts: list[str], n: int, seed: int | None = None
) -> list[str]:
    """从通过完整性筛选的文本块中抽取 n 个；候选不足时抛出异常。"""
    clean_chunks = [chunk for chunk in chunk_texts if looks_self_contained(chunk)]
    return random.Random(seed).sample(clean_chunks, n)
