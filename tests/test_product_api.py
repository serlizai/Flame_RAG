"""验证正式商品列表、单商品基础信息和独立于模型服务的读取链路。"""

import json
import os
from pathlib import Path
import subprocess
import sys

from fastapi.testclient import TestClient
import pytest

from backend import main, products


@pytest.fixture
def product_client(monkeypatch):
    """商品查询只读商品表，若误入问答流程则立即失败。"""

    def unexpected_agent_call(*args, **kwargs):
        raise AssertionError("商品查询不应调用 Agent。")

    monkeypatch.setattr(main, "_invoke_agent", unexpected_agent_call)
    with TestClient(main.app) as client:
        yield client


def test_product_list_matches_ten_product_catalog(product_client):
    expected = json.loads(products.PRODUCTS_PATH.read_text(encoding="utf-8"))
    response = product_client.get("/api/products")
    assert response.status_code == 200
    assert response.json() == expected
    assert len(response.json()) == 10
    assert len({product["category"] for product in response.json()}) == 6
    assert all(isinstance(product["sku"], str) for product in response.json())


@pytest.mark.parametrize("product_id", list(products.load_products()))
def test_product_detail_matches_catalog(product_client, product_id):
    catalog = json.loads(products.PRODUCTS_PATH.read_text(encoding="utf-8"))
    expected = next(product for product in catalog if product["id"] == product_id)
    response = product_client.get(f"/api/products/{product_id}")
    assert response.status_code == 200
    assert response.json() == expected


@pytest.mark.parametrize("product_id", ["unknown-product", "200001"])
def test_product_detail_rejects_unknown_id_and_sku(product_client, product_id):
    response = product_client.get(f"/api/products/{product_id}")
    assert response.status_code == 404
    assert response.json() == {"detail": "当前商品不存在。"}


def test_product_detail_rejects_blank_path_id(product_client):
    response = product_client.get("/api/products/%20")
    assert response.status_code == 422


@pytest.mark.parametrize("endpoint", ["/api/products", "/api/products/flame_kettle_k1"])
@pytest.mark.parametrize("content", [None, "{", "[]"])
def test_invalid_catalog_returns_generic_error(
    product_client, tmp_path, monkeypatch, content, endpoint
):
    catalog = tmp_path / "broken-products.json"
    if content is not None:
        catalog.write_text(content, encoding="utf-8")
    monkeypatch.setattr(products, "PRODUCTS_PATH", catalog)
    response = product_client.get(endpoint)
    assert response.status_code == 500
    assert response.json() == {"detail": "商品表配置错误。"}


def test_product_list_reads_updates_and_keeps_sku_leading_zero(
    product_client, tmp_path, monkeypatch
):
    catalog = tmp_path / "products.json"
    records = [
        {"id": "cup", "sku": "001001", "name": "示例杯子", "category": "drinkware"}
    ]
    catalog.write_text(json.dumps(records), encoding="utf-8")
    monkeypatch.setattr(products, "PRODUCTS_PATH", catalog)
    assert product_client.get("/api/products").json() == records
    assert product_client.get("/api/products/cup").json() == records[0]

    records[0]["name"] = "更新后的示例杯子"
    records.append(
        {"id": "lamp", "sku": "001002", "name": "示例台灯", "category": "lighting"}
    )
    catalog.write_text(json.dumps(records), encoding="utf-8")
    assert product_client.get("/api/products").json() == records
    assert product_client.get("/api/products/cup").json() == records[0]


def test_openapi_describes_product_array_and_fields(product_client):
    schema = product_client.get("/openapi.json").json()
    operation = schema["paths"]["/api/products"]["get"]
    response = operation["responses"]["200"]["content"]["application/json"]["schema"]
    assert response["type"] == "array"
    assert response["items"]["$ref"] == "#/components/schemas/ProductResponse"
    fields = schema["components"]["schemas"]["ProductResponse"]["properties"]
    assert set(fields) == {"id", "sku", "name", "category"}
    assert all(field["type"] == "string" for field in fields.values())

    detail = schema["paths"]["/api/products/{product_id}"]["get"]
    body = detail["responses"]["200"]["content"]["application/json"]["schema"]
    assert body["$ref"] == "#/components/schemas/ProductResponse"
    parameter = next(
        field for field in detail["parameters"] if field["name"] == "product_id"
    )
    assert parameter["in"] == "path" and parameter["required"] is True
    assert {"404", "500"} <= set(detail["responses"])


def test_product_queries_in_fresh_process_do_not_import_model_or_redis(tmp_path):
    """全新进程禁止导入重依赖，防止测试进程已加载的模块掩盖问题。"""
    project_root = Path(__file__).resolve().parents[1]
    script = """
import builtins
import sys

sys.path.insert(0, sys.argv[1])
original_import = builtins.__import__
blocked = (
    "backend.config", "backend.retrieval", "backend.embeddings", "backend.cache",
    "torch", "redis", "langchain_chroma", "langchain_openai",
)

def guarded_import(name, *args, **kwargs):
    if any(name == module or name.startswith(module + ".") for module in blocked):
        raise AssertionError("商品查询不应导入 " + name)
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from fastapi.testclient import TestClient
from backend.main import app

with TestClient(app) as client:
    response = client.get("/api/products")
    assert response.status_code == 200
    assert len(response.json()) == 10
    for product in response.json():
        detail = client.get("/api/products/" + product["id"])
        assert detail.status_code == 200
        assert detail.json() == product
    assert client.get("/api/products/unknown-product").status_code == 404
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "flame"}
    response = client.post("/api/chat", json={"message": "问题", "product_id": "unknown-product"})
    assert response.status_code == 404
    assert client.post("/api/chat", json={"message": " "}).status_code == 422
print("商品查询独立读取通过")
"""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.update(
        OPENAI_API_KEY="",
        BGE_M3_PATH=str(tmp_path / "missing-model"),
        REDIS_URL="redis://127.0.0.1:1/0",
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(project_root)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "商品查询独立读取通过" in result.stdout
