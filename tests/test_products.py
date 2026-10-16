"""验证商品身份读取、非法商品表拒绝和项目根目录路径解析。"""

import json

import pytest

from backend.products import Product, load_products


def test_catalog_keeps_sku_string_and_does_not_depend_on_current_directory(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    products = load_products()
    assert products["vivo_x500_pro"] == Product(
        id="vivo_x500_pro", sku="100001", name="vivo X500 Pro"
    )

    path = tmp_path / "products.json"
    path.write_text(
        json.dumps([{"id": "sample", "sku": "001001", "name": "示例商品"}]),
        encoding="utf-8",
    )
    assert load_products(path)["sample"].sku == "001001"


@pytest.mark.parametrize("category", ["", "  ", None, 123])
def test_catalog_rejects_explicit_invalid_category(tmp_path, category):
    path = tmp_path / "products.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "sample",
                    "sku": "001001",
                    "name": "示例商品",
                    "category": category,
                }
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="category"):
        load_products(path)


@pytest.mark.parametrize(
    "records",
    [
        [],
        {},
        ["示例商品"],
        [{"id": "sample", "sku": 100001, "name": "示例商品"}],
        [{"id": "sample", "sku": "100001", "name": " "}],
        [{"id": "sample", "sku": "100001"}],
        [
            {"id": "sample", "sku": "100001", "name": "商品一"},
            {"id": "sample", "sku": "100002", "name": "商品二"},
        ],
        [
            {"id": "first", "sku": "100001", "name": "商品一"},
            {"id": "second", "sku": "100001", "name": "商品二"},
        ],
    ],
)
def test_catalog_rejects_invalid_fields_and_duplicate_identity(tmp_path, records):
    path = tmp_path / "products.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    with pytest.raises(ValueError):
        load_products(path)


@pytest.mark.parametrize("content", [None, "{", b"\xff"])
def test_catalog_reports_missing_or_unreadable_file(tmp_path, content):
    path = tmp_path / "products.json"
    if isinstance(content, str):
        path.write_text(content, encoding="utf-8")
    elif isinstance(content, bytes):
        path.write_bytes(content)
    with pytest.raises(ValueError, match="无法读取商品表"):
        load_products(path)
