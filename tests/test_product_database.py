"""验证商品向量库写入、更新、清理及失败处理。"""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
import pytest

from backend import embeddings as embedding_config
from backend.product_documents import (
    DEFAULT_PRODUCTS_DIRECTORY,
    load_product_documents,
    split_product_documents,
)
from scripts.build_product_db import (
    INGESTION_PIPELINE,
    PRODUCT_COLLECTION_NAME,
    main,
    sync_product_chunks,
)
import settings


class OfflineEmbeddings(Embeddings):
    """使用真实 Chroma 验证持久化，不下载模型，也不据此判断语义检索质量。"""

    def embed_documents(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0, 0.0]


def product_store(directory, embeddings):
    return Chroma(
        collection_name=PRODUCT_COLLECTION_NAME,
        persist_directory=str(directory),
        embedding_function=embeddings,
    )


def write_product(path, product_id="sample_product", sku="001001", description=None):
    path.write_text(
        f"# Sample Product\n\nSKU: {sku}\nproduct_id: {product_id}\n\n"
        "## 商品简介\n\n" + (description or "支持无线充电与双卡。") + "\n",
        encoding="utf-8",
    )


def test_real_product_metadata_survives_persistence_and_repeated_builds(tmp_path):
    chunks = split_product_documents(load_product_documents(DEFAULT_PRODUCTS_DIRECTORY))
    embeddings = OfflineEmbeddings()
    directory = tmp_path / "products"

    assert sync_product_chunks(chunks, embeddings, directory) == 0
    assert sync_product_chunks(chunks, embeddings, directory) == 0

    stored = product_store(directory, embeddings).get()
    assert len(stored["ids"]) == len(chunks) == 22
    expected = {chunk.metadata["chunk_id"]: chunk for chunk in chunks}
    for chunk_id, text, metadata in zip(
        stored["ids"], stored["documents"], stored["metadatas"]
    ):
        assert text == expected[chunk_id].page_content
        assert metadata == {
            **expected[chunk_id].metadata,
            "ingestion_pipeline": INGESTION_PIPELINE,
        }
    pro_records = product_store(directory, embeddings).get(
        where={"product_id": "vivo_x500_pro"}
    )
    assert len(pro_records["ids"]) == 7
    assert all(metadata["sku_id"] == "100001" for metadata in pro_records["metadatas"])


def test_changed_and_removed_products_remove_old_chunks_but_preserve_other_data(
    tmp_path,
):
    products = tmp_path / "documents"
    products.mkdir()
    first = products / "first.md"
    second = products / "second.md"
    write_product(first, description="商品支持无线充电。" * 40)
    write_product(second, product_id="second_product", sku="001002")
    original = split_product_documents(
        load_product_documents(products), chunk_size=120, chunk_overlap=20
    )
    embeddings = OfflineEmbeddings()
    database = tmp_path / "database"
    sync_product_chunks(original, embeddings, database)
    marker = database / "keep.txt"
    marker.write_text("keep other local files")

    store = product_store(database, embeddings)
    store.add_documents(
        [Document(page_content="Manual entry", metadata={"source": "manual"})],
        ids=["manual"],
    )
    other_collection = Chroma(
        collection_name="other_documents",
        persist_directory=str(database),
        embedding_function=embeddings,
    )
    other_collection.add_documents(
        [Document(page_content="Other collection")], ids=["other"]
    )

    write_product(first, description="更新后的商品简介。")
    second.unlink()
    updated = split_product_documents(
        load_product_documents(products), chunk_size=120, chunk_overlap=20
    )
    removed = sync_product_chunks(updated, embeddings, database)

    old_ids = {chunk.metadata["chunk_id"] for chunk in original}
    new_ids = {chunk.metadata["chunk_id"] for chunk in updated}
    assert removed == len(old_ids - new_ids)
    assert set(store.get()["ids"]) == new_ids | {"manual"}
    assert store.get(where={"product_id": "second_product"})["ids"] == []
    assert other_collection.get()["ids"] == ["other"]
    assert marker.read_text() == "keep other local files"


def test_embedding_failure_does_not_delete_existing_chunks(tmp_path):
    class FailingEmbeddings(OfflineEmbeddings):
        def embed_documents(self, texts):
            raise RuntimeError("Embedding failed")

    products = tmp_path / "documents"
    products.mkdir()
    path = products / "sample.md"
    write_product(path)
    original = split_product_documents(load_product_documents(products))
    embeddings = OfflineEmbeddings()
    database = tmp_path / "database"
    sync_product_chunks(original, embeddings, database)
    before = product_store(database, embeddings).get()

    write_product(path, description="新的商品内容。")
    updated = split_product_documents(load_product_documents(products))
    with pytest.raises(RuntimeError, match="Embedding failed"):
        sync_product_chunks(updated, FailingEmbeddings(), database)

    assert product_store(database, embeddings).get() == before


def test_dry_run_from_another_directory_does_not_load_model_or_create_database(
    tmp_path, monkeypatch, capsys
):
    products = tmp_path / "data" / "products"
    products.mkdir(parents=True)
    write_product(products / "sample.md")
    monkeypatch.setattr(settings, "PROJECT_ROOT", tmp_path)
    monkeypatch.chdir(products)

    def unexpected_model_load(**kwargs):
        raise AssertionError("Dry run must not load an embedding model.")

    monkeypatch.setattr(embedding_config, "create_embeddings", unexpected_model_load)
    main(["--dry-run"])

    assert not (tmp_path / "chroma_db_products").exists()
    assert str(tmp_path / "chroma_db_products") in capsys.readouterr().out


def test_cli_builds_default_database_and_forwards_model_options(tmp_path, monkeypatch):
    products = tmp_path / "data" / "products"
    products.mkdir(parents=True)
    write_product(products / "sample.md")
    (tmp_path / "models" / "bge").mkdir(parents=True)
    monkeypatch.setattr(settings, "PROJECT_ROOT", tmp_path)
    monkeypatch.chdir(products)
    embeddings = OfflineEmbeddings()
    options = {}

    def create_embeddings(**kwargs):
        options.update(kwargs)
        return embeddings

    monkeypatch.setattr(embedding_config, "create_embeddings", create_embeddings)
    main(["--model-path", "models/bge", "--device", "cpu"])

    assert options == {"model_path": tmp_path / "models" / "bge", "device": "cpu"}
    stored = product_store(tmp_path / "chroma_db_products", embeddings).get()
    assert stored["ids"]
    assert all(metadata["sku_id"] == "001001" for metadata in stored["metadatas"])


def test_invalid_product_fails_before_model_loading_or_database_creation(
    tmp_path, monkeypatch
):
    products = tmp_path / "data" / "products"
    products.mkdir(parents=True)
    (products / "invalid.md").write_text("# Missing SKU\n")
    monkeypatch.setattr(settings, "PROJECT_ROOT", tmp_path)

    def unexpected_model_load(**kwargs):
        raise AssertionError("Invalid input must not load the model.")

    monkeypatch.setattr(embedding_config, "create_embeddings", unexpected_model_load)
    with pytest.raises(SystemExit) as error:
        main([])
    assert error.value.code == 2
    assert not (tmp_path / "chroma_db_products").exists()


def test_missing_model_configuration_does_not_create_database(
    tmp_path, monkeypatch, capsys
):
    products = tmp_path / "data" / "products"
    products.mkdir(parents=True)
    write_product(products / "sample.md")
    monkeypatch.setattr(settings, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("BGE_M3_PATH", raising=False)

    with pytest.raises(SystemExit) as error:
        main([])
    assert error.value.code == 2
    assert "BGE_M3_PATH" in capsys.readouterr().err
    assert not (tmp_path / "chroma_db_products").exists()


def test_empty_sync_does_not_create_database(tmp_path):
    database = tmp_path / "database"
    with pytest.raises(ValueError, match="No product chunks"):
        sync_product_chunks([], OfflineEmbeddings(), database)
    assert not database.exists()
