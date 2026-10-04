"""生成商品文本块及 metadata，不加载 BGE，也不写入 Chroma。"""

import argparse
from collections import Counter
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--products-directory", type=Path, default=Path("data/products")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/product_chunks.jsonl")
    )
    parser.add_argument("--category", default="smartphone")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    args = parser.parse_args(argv)

    from backend.product_documents import (
        load_product_documents,
        split_product_documents,
    )
    from settings import PROJECT_ROOT

    directory = args.products_directory.expanduser()
    if not directory.is_absolute():
        directory = PROJECT_ROOT / directory
    output = args.output.expanduser()
    if not output.is_absolute():
        output = PROJECT_ROOT / output

    try:
        documents = load_product_documents(directory, category=args.category)
        chunks = split_product_documents(
            documents, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
        )
    except ValueError as error:
        parser.error(str(error))

    payload = "".join(
        json.dumps(
            {"page_content": chunk.page_content, "metadata": chunk.metadata},
            ensure_ascii=False,
        )
        + "\n"
        for chunk in chunks
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")
    counts = Counter(chunk.metadata["source"] for chunk in chunks)
    print(f"Prepared {len(chunks)} chunks from {len(documents)} products: {output}")
    for source, count in sorted(counts.items()):
        print(f"  {source}: {count} chunks")
    print("Example metadata:")
    print(json.dumps(chunks[0].metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
