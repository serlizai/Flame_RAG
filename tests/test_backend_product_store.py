"""验证后端配置和检索工具连接商品集合，避免读到其他语料或空集合。"""

import importlib.util
from pathlib import Path
import sys

from langchain import chat_models
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from backend import embeddings as embedding_config
from backend.product_documents import (
    DEFAULT_PRODUCTS_DIRECTORY,
    load_product_documents,
    split_product_documents,
)
from scripts.build_product_db import sync_product_chunks
import settings


class OfflineEmbeddings(Embeddings):
    """提供固定向量，只验证后端接线，不用于判断语义检索质量。"""

    def embed_documents(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0, 0.0]


def load_module(name, path, monkeypatch):
    """使用独立模块加载配置，避免模型初始化影响其他测试。"""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_backend_tool_reads_product_collection_from_project_root(tmp_path, monkeypatch):
    source_root = Path(__file__).resolve().parents[1]
    embeddings = OfflineEmbeddings()
    chunks = split_product_documents(load_product_documents(DEFAULT_PRODUCTS_DIRECTORY))
    product_directory = tmp_path / "chroma_db_products"
    sync_product_chunks(chunks, embeddings, product_directory)

    # 在两个目录的默认集合中放入其他语料，检查后端是否选对路径和集合。
    for directory in (tmp_path / "chroma_db", product_directory):
        Chroma(
            persist_directory=str(directory), embedding_function=embeddings
        ).add_documents(
            [Document(page_content="其他语料", metadata={"source": "legacy.pdf"})],
            ids=["legacy"],
        )

    monkeypatch.setattr(settings, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(chat_models, "init_chat_model", lambda **kwargs: object())
    monkeypatch.setattr(embedding_config, "create_embeddings", lambda: embeddings)
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    other_directory = tmp_path / "elsewhere"
    other_directory.mkdir()
    monkeypatch.chdir(other_directory)

    config = load_module(
        "flame_product_config_test", source_root / "backend" / "config.py", monkeypatch
    )
    stored = config.product_vector_store.get()
    assert set(stored["ids"]) == {chunk.metadata["chunk_id"] for chunk in chunks}

    monkeypatch.setitem(sys.modules, "backend.config", config)
    tools = load_module(
        "flame_product_tools_test", source_root / "backend" / "tools.py", monkeypatch
    )
    result = tools.retrieve_docs.invoke(
        {
            "type": "tool_call",
            "name": "retrieve_docs",
            "id": "product-retrieval",
            "args": {"query": "X500 Pro 有什么卖点？"},
        }
    )
    sources = {chunk.metadata["source"] for chunk in chunks}
    assert len(result.artifact) == 5
    assert all(citation.source in sources for citation in result.artifact)
    assert all(citation.label.startswith("vivo X500") for citation in result.artifact)
    assert all(citation.snippet in result.content for citation in result.artifact)
    assert "其他语料" not in result.content
