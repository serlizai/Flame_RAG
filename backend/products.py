"""读取基础商品表，为 API 校验和后续当前商品上下文提供商品身份。"""

from dataclasses import dataclass
import json
from pathlib import Path

from settings import PROJECT_ROOT

PRODUCTS_PATH = PROJECT_ROOT / "data" / "products.json"


@dataclass(frozen=True)
class Product:
    """商品表中的基础身份字段；SKU 保持字符串。"""

    id: str
    sku: str
    name: str


def load_products(path: Path | None = None) -> dict[str, Product]:
    """读取并校验商品表，以商品 ID 建立索引，拒绝重复 ID 和 SKU。"""
    path = path or PRODUCTS_PATH
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("无法读取商品表，请检查 data/products.json。") from error
    if not isinstance(records, list) or not records:
        raise ValueError("商品表必须是非空列表。")

    products: dict[str, Product] = {}
    skus: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or any(
            not isinstance(record.get(key), str) or not record[key].strip()
            for key in ("id", "sku", "name")
        ):
            raise ValueError("每个商品必须包含非空字符串 id、sku、name。")
        product = Product(**{key: record[key].strip() for key in ("id", "sku", "name")})
        if product.id in products or product.sku in skus:
            raise ValueError("商品表中的商品 ID 和 SKU 必须唯一。")
        products[product.id] = product
        skus.add(product.sku)
    return products
