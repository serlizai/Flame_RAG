"""读取商品 Markdown，并让每个文本块保留所属商品的身份信息。"""

from collections import Counter
import hashlib
from pathlib import Path
import re

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from settings import PROJECT_ROOT

DEFAULT_PRODUCTS_DIRECTORY = PROJECT_ROOT / "data" / "products"
REQUIRED_METADATA_KEYS = (
    "product_id",
    "sku_id",
    "product_name",
    "category",
    "source",
)
_FIELD_PATTERN = re.compile(r"^(SKU|product_id):[ \t]*(.*?)[ \t]*$", re.MULTILINE)


def load_product_document(path: Path, category: str = "smartphone") -> Document:
    """从文档头部读取商品信息；SKU 保持字符串，不转换为数字。"""
    content = path.read_text(encoding="utf-8")
    header = re.split(r"^##[ \t]+", content, maxsplit=1, flags=re.MULTILINE)[0]
    title = re.search(r"^#[ \t]+(.+?)[ \t]*$", header, re.MULTILINE)
    if title is None:
        raise ValueError(f"{path.name}: missing product name (# heading).")

    pairs = _FIELD_PATTERN.findall(header)
    duplicate_fields = [
        key for key, count in Counter(key for key, _ in pairs).items() if count > 1
    ]
    if duplicate_fields:
        raise ValueError(
            f"{path.name}: duplicate header fields: {', '.join(duplicate_fields)}."
        )
    fields = dict(pairs)
    for key in ("SKU", "product_id"):
        if not fields.get(key):
            raise ValueError(
                f"{path.name}: missing or empty {key} in the product header."
            )
    if not category.strip():
        raise ValueError("Product category must not be empty.")

    return Document(
        page_content=content,
        metadata={
            "product_id": fields["product_id"],
            "sku_id": fields["SKU"],
            "product_name": title.group(1).strip(),
            "category": category.strip(),
            "source": path.name,
        },
    )


def load_product_documents(
    directory: Path = DEFAULT_PRODUCTS_DIRECTORY,
    category: str = "smartphone",
) -> list[Document]:
    """每个 Markdown 文件读取一个商品，并拒绝重复的商品 ID 或 SKU。"""
    paths = sorted(directory.glob("*.md"))
    if not paths:
        raise ValueError(f"No product Markdown files found in {directory}.")
    documents = [load_product_document(path, category=category) for path in paths]
    for key in ("product_id", "sku_id"):
        seen: dict[str, str] = {}
        for document in documents:
            value = document.metadata[key]
            if value in seen:
                raise ValueError(
                    f"Duplicate {key} {value!r}: {seen[value]} and {document.metadata['source']}."
                )
            seen[value] = document.metadata["source"]
    return documents


def split_product_documents(
    documents: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> list[Document]:
    """先按章节分组，再切分文本，并在每个块上保留商品 metadata。"""
    if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= chunk_overlap < chunk_size.")
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("##", "section")], strip_headers=False
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n### ",
            "\n\n",
            "\n",
            "。",
            "！",
            "？",
            "；",
            "，",
            " ",
            "",
        ],  # 按顺序尝试
    )
    chunks = []
    for document in documents:
        for key in REQUIRED_METADATA_KEYS:
            if (
                not isinstance(document.metadata.get(key), str)
                or not document.metadata[key].strip()
            ):
                raise ValueError(f"Missing or invalid product metadata: {key}.")
        sections = [
            Document(
                page_content=section.page_content,
                metadata={
                    **document.metadata,  # 复制商品的 metadata
                    "section": section.metadata.get("section", "商品信息"),
                },
            )
            for section in header_splitter.split_text(document.page_content)
        ]
        product_chunks = text_splitter.split_documents(sections)
        # 给chunks编号
        for index, chunk in enumerate(product_chunks):
            # 生成每个块的唯一身份标识符，确保即使内容相同但属于不同商品或不同块，也能区分开来
            identity = "\0".join(
                [chunk.metadata[key] for key in REQUIRED_METADATA_KEYS]
                + [str(index), chunk.page_content]
            )
            chunk.metadata.update(
                chunk_index=index,
                chunk_id=hashlib.sha256(
                    identity.encode("utf-8")
                ).hexdigest(),  # SHA256保证内容不变id也不会变
            )
        chunks.extend(product_chunks)
    return chunks
