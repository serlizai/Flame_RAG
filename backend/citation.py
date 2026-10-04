"""为基于检索结果的回答提供引用标签和编号。

Agent 后端与经典 RAG 链路共用同一套引用模型。
本模块重新导出顶层 citations 模块的实现，避免重复维护。
"""

from citations import (
    CITATION_MARKER_KEY,
    LOCATOR_KEYS,
    TITLE_KEYS,
    Citation,
    annotate_documents_for_citation,
    cache_payload_to_result,
    citations_to_cache_payload,
    format_citation_label,
    resolve_answer_citations,
)

__all__ = [
    "CITATION_MARKER_KEY",
    "LOCATOR_KEYS",
    "TITLE_KEYS",
    "Citation",
    "annotate_documents_for_citation",
    "cache_payload_to_result",
    "citations_to_cache_payload",
    "format_citation_label",
    "resolve_answer_citations",
]
