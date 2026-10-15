"""定义从商品知识库检索文档的模型工具。"""

# 导入与商品建库脚本对应的向量库
from backend.config import product_vector_store
from backend.citation import annotate_documents_for_citation, CITATION_MARKER_KEY
from typing import Annotated

from langchain.tools import InjectedToolArg, tool


@tool(response_format="content_and_artifact")
def retrieve_docs(
    query: str,
    product_id: Annotated[str | None, InjectedToolArg] = None,
    citation_start: Annotated[int, InjectedToolArg] = 1,
):
    """根据语义相似度从商品知识库检索相关文档，并返回带编号的文本和引用。

    参数：
        query：用于语义检索的完整用户问题，宜使用自然语言句子。
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
