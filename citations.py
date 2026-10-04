"""为检索回答生成引用标签、编号和缓存载荷。

标签使用文档标题与位置 metadata，可用于不同语料。
缺少可识别身份信息的文本块不会生成引用。
"""

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from langchain_core.documents import Document

# 正则同时匹配引用编号前的一个可选空格，
# 避免移除无依据的标记后留下连续空格。
_MARKER_PATTERN = re.compile(r" ?\[(\d+)\]")

TITLE_KEYS = ("product_name", "source_name", "title", "document_title")

LOCATOR_KEYS = (
    "section",
    "page_label",
    "page",
    "part",
    "chapter",
)

CITATION_MARKER_KEY = "citation_marker"


@dataclass(frozen=True)
class Citation:
    """一个带编号、可追溯到具体检索文本块的引用。"""

    number: int
    label: str
    snippet: str
    source: str | None


def _normalise(value: Any) -> str | None:
    """合并连续空白，保留大小写，并把空值或空字符串视为缺失。"""
    if value is None:
        return None
    text = " ".join(str(value).split())
    return text or None


def _first_present(
    metadata: Mapping[str, Any], keys: tuple[str, ...]
) -> tuple[str | None, str | None]:
    for key in keys:
        value = _normalise(metadata.get(key))
        if value is not None:
            return key, value
    return None, None


def format_citation_label(metadata: Mapping[str, Any]) -> str | None:
    """生成可读的文档标签；缺少身份 metadata 时返回 None。"""
    _, title = _first_present(metadata, TITLE_KEYS)
    if title is None:
        return None

    locator_key, locator = _first_present(metadata, LOCATOR_KEYS)
    if locator is None:
        return title

    if locator_key == "page":
        page = metadata["page"]
        locator = f"page {page + 1 if isinstance(page, int) else locator}"
    elif locator_key == "page_label":
        locator = f"page {locator}"

    locator_name = _normalise(metadata.get(f"{locator_key}_name"))
    if locator_name is not None:
        return f"{title}, {locator} ({locator_name})"
    return f"{title}, {locator}"


def annotate_documents_for_citation(
    documents: list[Document],
) -> tuple[list[Document], list[Citation]]:
    """复制文档并添加 citation_marker，只对可以引用的文档连续编号。"""
    annotated: list[Document] = []
    citations: list[Citation] = []

    for document in documents:
        label = format_citation_label(document.metadata)
        marker = ""

        if label is not None:
            citation = Citation(
                number=len(citations) + 1,
                label=label,
                snippet=document.page_content,
                source=_normalise(document.metadata.get("source")),
            )
            citations.append(citation)
            marker = f"[{citation.number}] {citation.label}"

        annotated.append(
            Document(
                page_content=document.page_content,
                metadata={**document.metadata, CITATION_MARKER_KEY: marker},
            )
        )

    return annotated, citations


def resolve_answer_citations(
    answer: str, citations: list[Citation]
) -> tuple[str, list[Citation]]:
    """移除没有检索依据的编号，并剔除回答中未使用的引用。"""
    citations_by_number = {citation.number: citation for citation in citations}
    cited_numbers: set[int] = set()

    def keep_or_strip(match: re.Match[str]) -> str:
        number = int(match.group(1))
        if number not in citations_by_number:
            return ""
        cited_numbers.add(number)
        return match.group(0)

    resolved = _MARKER_PATTERN.sub(keep_or_strip, answer)
    kept = [citations_by_number[number] for number in sorted(cited_numbers)]
    return resolved, kept


def citations_to_cache_payload(answer: str, citations: list[Citation]) -> str:
    """把回答和引用序列化为可写入缓存的 JSON 字符串。"""
    return json.dumps(
        {"answer": answer, "citations": [asdict(c) for c in citations]},
        ensure_ascii=False,
    )


def cache_payload_to_result(payload: str) -> tuple[str, list[Citation]]:
    """从缓存载荷恢复回答及引用列表。"""
    data = json.loads(payload)
    return data["answer"], [Citation(**item) for item in data["citations"]]
