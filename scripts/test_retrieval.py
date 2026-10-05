"""使用真实本地 BGE 和商品库，验收第二周的三个检索问题。

在项目根目录执行：
    python -m scripts.test_retrieval
也支持直接执行文件：
    python scripts/test_retrieval.py
按测试用例中的目标商品过滤：
    python scripts/test_retrieval.py --filter-products
本脚本只检索，不调用聊天模型或 Redis。
单商品问题要求首条结果属于目标商品，对比问题要求前 k 条包含两个商品。
"""

import argparse
from pathlib import Path
import sys

# 单商品问题检查首条结果；对比问题检查两个商品；章节命中单独诊断。
RETRIEVAL_CASES = (
    ("X500 Pro 有什么卖点？", {"vivo_x500_pro"}, "核心卖点"),
    ("X500 Pro 有哪些配置？", {"vivo_x500_pro"}, "核心参数"),
    (
        "X500 Pro Max 和 Pro 有什么区别？",
        {"vivo_x500_pro", "vivo_x500_pro_max"},
        None,
    ),
)


def main(argv=None) -> int:
    """打印召回明细；全部通过返回 0，任一问题未达标返回 1。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--persist-directory", type=Path, default=Path("chroma_db_products")
    )
    parser.add_argument("--model-path", type=Path, help="覆盖 BGE_M3_PATH")
    parser.add_argument("--device", help="覆盖 EMBEDDING_DEVICE，例如 cpu 或 mps")
    parser.add_argument("--k", type=int, default=5, help="每个问题召回的块数，默认 5")
    parser.add_argument(
        "--filter-products",
        action="store_true",
        help="按测试用例的目标商品过滤，对比问题同时允许两个商品",
    )
    args = parser.parse_args(argv)
    if args.k <= 0:
        parser.error("--k 必须大于 0。")

    # 直接执行文件时，Python 只将 scripts 目录加入导入路径。
    # 根据脚本位置补充项目根目录，避免依赖终端的当前工作目录。
    if not __package__:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from settings import PROJECT_ROOT

    def project_path(path: Path) -> Path:
        """相对路径从项目根目录解析，避免运行位置影响测试。"""
        path = path.expanduser()
        return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()

    directory = project_path(args.persist_directory)
    if not (directory / "chroma.sqlite3").is_file():
        parser.error("商品库不存在，请先运行 python -m scripts.build_product_db。")

    from chromadb.errors import NotFoundError
    from langchain_chroma import Chroma

    from backend.embeddings import create_embeddings
    from scripts.build_product_db import PRODUCT_COLLECTION_NAME

    # 先检查已有集合，避免测试误建空库或加载模型后才发现没有数据。
    try:
        stored = Chroma(
            collection_name=PRODUCT_COLLECTION_NAME,
            persist_directory=str(directory),
            embedding_function=None,
            create_collection_if_not_exists=False,
        ).get(include=[])
    except NotFoundError:
        parser.error("商品库中没有 products 集合，请重新运行商品建库脚本。")
    if not stored["ids"]:
        parser.error("products 集合没有数据，请先运行商品建库脚本。")

    model_path = project_path(args.model_path) if args.model_path else None
    try:
        embeddings = create_embeddings(model_path=model_path, device=args.device)
    except ValueError as error:
        parser.error(str(error))
    product_vector_store = Chroma(
        collection_name=PRODUCT_COLLECTION_NAME,
        persist_directory=str(directory),
        embedding_function=embeddings,
        create_collection_if_not_exists=False,
    )

    print(f"商品库：{directory}；集合：{PRODUCT_COLLECTION_NAME}")
    print(f"已有 {len(stored['ids'])} 个块；每次检索 k={args.k}。")
    print(f"商品过滤：{'按测试用例启用' if args.filter_products else '未启用'}。")
    passed_count = 0
    for question, expected_products, expected_section in RETRIEVAL_CASES:
        retrieval_filter = None
        if args.filter_products:
            # 测试用例明确指定候选商品；实际业务中应由当前商品上下文提供。
            product_ids = sorted(expected_products)
            retrieval_filter = {
                "product_id": (
                    product_ids[0] if len(product_ids) == 1 else {"$in": product_ids}
                )
            }
        docs = product_vector_store.similarity_search(
            question, k=args.k, filter=retrieval_filter
        )
        print(f"\n问题：{question}")
        if retrieval_filter:
            print(f"过滤条件：{retrieval_filter}")
        for rank, doc in enumerate(docs, start=1):
            metadata = doc.metadata
            preview = " ".join(doc.page_content.split())[:120]
            print(
                f"  {rank}. {metadata.get('product_name')} | "
                f"{metadata.get('section')} | {metadata.get('source')}"
            )
            print(f"     {preview}")

        matched_products = {doc.metadata.get("product_id") for doc in docs}
        passed = expected_products <= matched_products
        if args.filter_products:
            # 开启过滤时，除命中目标商品外，还要求结果不混入其他商品。
            passed = passed and not (matched_products - expected_products)
        if len(expected_products) == 1:
            passed = (
                passed
                and bool(docs)
                and (docs[0].metadata.get("product_id") in expected_products)
            )
        passed_count += int(passed)
        target = "、".join(sorted(expected_products))
        print(f"商品召回：{'通过' if passed else '失败'}；要求召回：{target}")
        if expected_section:
            section_hit = any(
                doc.metadata.get("product_id") in expected_products
                and doc.metadata.get("section") == expected_section
                for doc in docs
            )
            print(
                f"章节诊断：“{expected_section}”"
                f"{'已命中' if section_hit else '未命中'}（不影响基础商品召回验收）。"
            )

    print(f"\n基础商品召回验收：{passed_count}/{len(RETRIEVAL_CASES)} 通过。")
    return 0 if passed_count == len(RETRIEVAL_CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
