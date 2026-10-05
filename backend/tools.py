"""定义从商品知识库检索文档的模型工具。"""

# 导入与商品建库脚本对应的向量库
from backend.config import product_vector_store
from backend.citation import annotate_documents_for_citation, CITATION_MARKER_KEY
from langchain.tools import tool


@tool(response_format="content_and_artifact")
def retrieve_docs(query: str):
    """根据语义相似度从商品知识库检索相关文档，并返回带编号的文本和引用。

    参数：
        query：用于语义检索的完整用户问题，宜使用自然语言句子。
    """

    result = product_vector_store.similarity_search(query, k=5)
    annotated_docs, citations = annotate_documents_for_citation(result)
    content = "\n\n".join(
        f"{doc.metadata[CITATION_MARKER_KEY]}\n{doc.page_content}".strip()
        for doc in annotated_docs
    )

    return content, citations
