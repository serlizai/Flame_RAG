"""读取本地商品 Markdown，使用本地 BGE 构建 Chroma 商品集合。

在项目根目录执行：
    python -m scripts.build_product_db
    python -m scripts.build_product_db --model-path /path/to/bge-m3
"""

import argparse
from collections import Counter
from pathlib import Path

PRODUCT_COLLECTION_NAME = "products"
INGESTION_PIPELINE = "flame_product_markdown_v1"


def sync_product_chunks(chunks, embeddings, persist_directory: Path) -> int:
    """写入或更新当前文本块，成功后移除本脚本管理的过期块。"""
    if not chunks:
        raise ValueError(
            "No product chunks to write; the existing database is unchanged."
        )
    ids = [chunk.metadata["chunk_id"] for chunk in chunks]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate product chunk IDs.")

    from langchain_chroma import Chroma
    from langchain_core.documents import Document

    vector_store = Chroma(
        collection_name=PRODUCT_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )
    existing_ids = vector_store.get(
        where={"ingestion_pipeline": INGESTION_PIPELINE}, include=[]
    )["ids"]
    documents = [
        Document(
            page_content=chunk.page_content,
            metadata={**chunk.metadata, "ingestion_pipeline": INGESTION_PIPELINE},
        )
        for chunk in chunks
    ]

    # 只有向量生成和新块写入成功后，才清理旧数据。
    vector_store.add_documents(documents=documents, ids=ids)
    obsolete_ids = sorted(set(existing_ids) - set(ids))
    if obsolete_ids:
        vector_store.delete(ids=obsolete_ids)
    return len(obsolete_ids)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--products-directory", type=Path, default=Path("data/products")
    )
    parser.add_argument(
        "--persist-directory", type=Path, default=Path("chroma_db_products")
    )
    parser.add_argument("--model-path", type=Path, help="Override BGE_M3_PATH")
    parser.add_argument("--device", help="Override EMBEDDING_DEVICE, e.g. cpu or mps")
    parser.add_argument("--category", default="smartphone")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and preview without loading BGE or writing Chroma",
    )
    args = parser.parse_args(argv)

    from backend.product_documents import (
        load_product_documents,
        split_product_documents,
    )
    from settings import PROJECT_ROOT

    def project_path(path: Path) -> Path:
        path = path.expanduser()
        return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()

    products_directory = project_path(args.products_directory)
    persist_directory = project_path(args.persist_directory)
    try:
        products = load_product_documents(products_directory, category=args.category)
        chunks = split_product_documents(
            products, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
        )
        if not chunks:
            raise ValueError("Product documents did not produce any chunks.")
    except ValueError as error:
        parser.error(str(error))

    counts = Counter(chunk.metadata["source"] for chunk in chunks)
    print(f"Prepared {len(chunks)} chunks from {len(products)} products.")
    for source, count in sorted(counts.items()):
        print(f"  {source}: {count} chunks")
    print(f"Database: {persist_directory}")
    print(f"Collection: {PRODUCT_COLLECTION_NAME}")
    if args.dry_run:
        print("Dry run complete; no embedding model loaded or database written.")
        return

    from backend.embeddings import create_embeddings

    model_path = project_path(args.model_path) if args.model_path else None
    try:
        embeddings = create_embeddings(model_path=model_path, device=args.device)
    except ValueError as error:
        parser.error(str(error))
    removed_count = sync_product_chunks(chunks, embeddings, persist_directory)
    print(
        f"Stored {len(chunks)} product chunks; removed {removed_count} obsolete chunks."
    )


if __name__ == "__main__":
    main()
