"""验证检索评估按原始文本匹配目标块。"""

import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from eval import retrieval_eval
from eval.chunk_sampling import load_all_chunk_texts
from eval.golden_set import GoldenQAItem
from eval.metrics import recall_confidence_interval
from eval.retrieval_eval import run_retrieval_eval


def test_run_retrieval_eval_scores_against_retrieved_chunks():
    golden_set = [
        GoldenQAItem(question="q1", chunk_text="correct chunk 1"),
        GoldenQAItem(question="q2", chunk_text="correct chunk 2"),
    ]

    def fake_retrieve(question: str) -> list[Document]:
        if question == "q1":
            return [
                Document(page_content="correct chunk 1"),
                Document(page_content="other"),
            ]
        return [Document(page_content="unrelated")]

    results = run_retrieval_eval(golden_set, fake_retrieve)

    assert results.ranks == [1, None]
    assert results.recall == 0.5
    assert results.mrr == pytest.approx(0.5)
    assert results.recall_ci == recall_confidence_interval([1, None])


def test_real_retriever_and_sampling_read_product_collection(tmp_path, monkeypatch):
    """商品评测应读取产品集合并召回五条，排除同目录默认集合中的其他资料。"""

    class OfflineEmbeddings(Embeddings):
        """用固定向量验证集合选择和召回数量，不评估语义相关性。"""

        def embed_documents(self, texts):
            return [[1.0, 0.0, 0.0] for _ in texts]

        def embed_query(self, text):
            return [1.0, 0.0, 0.0]

    directory = str(tmp_path / "products")
    embeddings = OfflineEmbeddings()
    product_texts = {f"商品参数第 {index} 块" for index in range(6)}
    Chroma(
        collection_name="products",
        persist_directory=directory,
        embedding_function=embeddings,
    ).add_documents([Document(page_content=text) for text in sorted(product_texts)])
    Chroma(persist_directory=directory, embedding_function=embeddings).add_documents(
        [Document(page_content="其他语料")]
    )
    monkeypatch.setattr(retrieval_eval, "create_embeddings", lambda: embeddings)

    retrieved = retrieval_eval.build_real_retriever(directory).invoke("商品配置")

    assert len(retrieved) == 5
    assert {doc.page_content for doc in retrieved} <= product_texts
    assert set(load_all_chunk_texts(directory)) == product_texts
