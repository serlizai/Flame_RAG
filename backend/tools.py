"""定义从商品知识库检索文档的模型工具。"""

# 导入与商品建库脚本对应的向量库
from backend.config import product_vector_store
from backend.citation import annotate_documents_for_citation, CITATION_MARKER_KEY
from typing import Annotated

from langchain.tools import InjectedToolArg, tool


@tool(response_format="content_and_artifact")
def retrieve_product_docs(
    query: str,
    product_id: Annotated[str | None, InjectedToolArg] = None,
    citation_start: Annotated[int, InjectedToolArg] = 1,
):
    """用于查询当前商品的功能、参数说明、特点、使用方法、FAQ 和商品知识。

    涉及商品事实且已有资料不足时调用；价格或售后问题可查文档中的说明。
    当前商品由后端确定，未选商品时查询整个商品库。返回带引用编号的原文，
    只用于支持原文明示的事实，未提供所问信息时应说明暂未查询到。

    参数：
        query：待查询的具体商品问题，保留用户关心的型号、功能或参数。
        product_id：由工具节点从 State 注入，模型不能选择检索商品。
        citation_start：由工具节点注入，避免连续检索的引用编号重复。
    """

    search_options = {"k": 5}
    if product_id is not None:
        search_options["filter"] = {"product_id": product_id}
    # 不指定商品时保留全库检索；指定商品却没有结果时，不扩大检索范围。
    result = product_vector_store.similarity_search(query, **search_options)
    annotated_docs, citations = annotate_documents_for_citation(
        result, start_number=citation_start
    )
    content = "\n\n".join(
        f"{doc.metadata[CITATION_MARKER_KEY]}\n{doc.page_content}".strip()
        for doc in annotated_docs
    )

    return content or "暂未查询到当前检索范围内的相关信息。", citations
