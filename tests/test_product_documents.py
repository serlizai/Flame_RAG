"""验证商品头部字段解析、文本切分和 metadata 保留。"""

import json

import pytest

from backend.product_documents import (
    DEFAULT_PRODUCTS_DIRECTORY,
    REQUIRED_METADATA_KEYS,
    load_product_document,
    load_product_documents,
    split_product_documents,
)
from citations import annotate_documents_for_citation
from scripts.prepare_product_chunks import main
import settings


def write_product(path, sku="001001", product_id="sample_product"):
    path.write_text(
        f"# Sample Product\n\nSKU: {sku}\nproduct_id: {product_id}\n"
        "品牌: Sample\n型号: Product\n\n"
        "## 核心参数\n\n支持无线充电。\n\n"
        "## FAQ\n\n### 支持什么功能？\n\n支持无线充电和双卡。\n",
        encoding="utf-8",
    )


def test_reads_identity_from_header_and_preserves_leading_zero_in_sku(tmp_path):
    path = tmp_path / "sample.md"
    write_product(path)
    product = load_product_document(path)

    assert product.metadata == {
        "product_id": "sample_product",
        "sku_id": "001001",
        "product_name": "Sample Product",
        "category": "smartphone",
        "source": "sample.md",
    }


@pytest.mark.parametrize(
    "header",
    [
        "# Sample Product\nproduct_id: sample_product\n",
        "# Sample Product\nSKU:\nproduct_id: sample_product\n",
        "# Sample Product\nSKU: 100001\nproduct_id:\n",
        "# Sample Product\nSKU: 100001\nSKU: 100002\nproduct_id: sample_product\n",
        "SKU: 100001\nproduct_id: sample_product\n",
    ],
)
def test_rejects_invalid_headers_instead_of_using_fields_from_body(tmp_path, header):
    path = tmp_path / "sample.md"
    path.write_text(
        header + "\n## FAQ\nSKU: 100001\nproduct_id: body_example\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_product_document(path)


@pytest.mark.parametrize("key", ["sku_id", "product_id"])
def test_rejects_identifiers_shared_by_two_product_files(tmp_path, key):
    write_product(tmp_path / "first.md", sku="100001", product_id="first_product")
    write_product(
        tmp_path / "second.md",
        sku="100001" if key == "sku_id" else "100002",
        product_id="first_product" if key == "product_id" else "second_product",
    )
    with pytest.raises(ValueError, match=f"Duplicate {key}"):
        load_product_documents(tmp_path)


def test_real_product_chunks_keep_their_own_metadata_and_section():
    products = load_product_documents(DEFAULT_PRODUCTS_DIRECTORY)
    chunks = split_product_documents(products)
    assert len(products) == 3
    assert len(chunks) > len(products)
    assert len({chunk.metadata["chunk_id"] for chunk in chunks}) == len(chunks)
    expected = {
        product.metadata["product_id"]: product.metadata for product in products
    }

    for chunk in chunks:
        metadata = chunk.metadata
        assert {key: metadata[key] for key in REQUIRED_METADATA_KEYS} == expected[
            metadata["product_id"]
        ]
        assert 0 < len(chunk.page_content) <= 800
        assert isinstance(metadata["chunk_index"], int)
        assert metadata["section"]
        # 同一个文本块不能混入其他章节的标题。
        headings = [
            line[3:].strip()
            for line in chunk.page_content.splitlines()
            if line.startswith("## ")
        ]
        assert all(heading == metadata["section"] for heading in headings)

    _, citations = annotate_documents_for_citation(chunks)
    assert len(citations) == len(chunks)
    assert any(citation.label == "vivo X500 Pro, 核心参数" for citation in citations)
    assert [chunk.metadata["chunk_id"] for chunk in chunks] == [
        chunk.metadata["chunk_id"] for chunk in split_product_documents(products)
    ]


def test_exports_metadata_with_text_when_invoked_from_another_directory(
    tmp_path, monkeypatch
):
    products_directory = tmp_path / "data" / "products"
    products_directory.mkdir(parents=True)
    write_product(products_directory / "sample.md")
    monkeypatch.setattr(settings, "PROJECT_ROOT", tmp_path)
    monkeypatch.chdir(products_directory)

    main([])

    output = tmp_path / "data" / "product_chunks.jsonl"
    records = [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert records
    assert all(record["page_content"] for record in records)
    assert all(record["metadata"]["sku_id"] == "001001" for record in records)
    assert all(record["metadata"]["source"] == "sample.md" for record in records)


def test_invalid_input_does_not_replace_a_previous_chunk_export(tmp_path, monkeypatch):
    products_directory = tmp_path / "data" / "products"
    products_directory.mkdir(parents=True)
    (products_directory / "invalid.md").write_text("# Missing identifiers\n")
    output = tmp_path / "data" / "product_chunks.jsonl"
    output.write_text("previous export\n")
    monkeypatch.setattr(settings, "PROJECT_ROOT", tmp_path)

    with pytest.raises(SystemExit) as error:
        main([])

    assert error.value.code == 2
    assert output.read_text() == "previous export\n"


@pytest.mark.parametrize("chunk_size, overlap", [(0, 0), (800, 800), (800, -1)])
def test_rejects_invalid_chunk_settings(chunk_size, overlap):
    with pytest.raises(ValueError):
        split_product_documents([], chunk_size=chunk_size, chunk_overlap=overlap)


def test_rejects_missing_product_metadata_before_splitting():
    from langchain_core.documents import Document

    with pytest.raises(ValueError, match="product_id"):
        split_product_documents([Document(page_content="No product identity")])


def test_empty_product_directory_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="No product Markdown"):
        load_product_documents(tmp_path)
