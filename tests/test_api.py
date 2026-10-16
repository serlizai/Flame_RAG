"""验证真实 HTTP 路由和输入校验，问答服务使用替身以避免调用外部模型。"""

from fastapi.testclient import TestClient
import pytest

from backend import main
from backend.citation import Citation
from backend.products import load_products


@pytest.fixture
def api_client(monkeypatch):
    """记录路由传给 Agent 的输入，并提供可序列化的回答和引用。"""
    calls = []

    def invoke(query, session_id, product_id):
        calls.append((query, session_id, product_id))
        return (
            "示例回答。[1]",
            [Citation(1, "示例商品, FAQ", "引用原文", "sample.md")],
            session_id or "generated-session",
        )

    monkeypatch.setattr(main, "_invoke_agent", invoke)
    with TestClient(main.app) as client:
        yield client, calls


def test_post_passes_product_message_and_session_to_agent(api_client):
    client, calls = api_client
    response = client.post(
        "/query",
        json={
            "product_id": " vivo_x500_pro ",
            "message": " 这款支持无线充电吗？ ",
            "session_id": " existing-session ",
        },
    )
    assert response.status_code == 200
    assert calls == [("这款支持无线充电吗？", "existing-session", "vivo_x500_pro")]
    assert response.json() == {
        "answer": "示例回答。[1]",
        "citations": [
            {
                "number": 1,
                "label": "示例商品, FAQ",
                "snippet": "引用原文",
                "source": "sample.md",
            }
        ],
        "session_id": "existing-session",
        "product_id": "vivo_x500_pro",
    }


def test_get_keeps_original_query_and_optional_session(api_client):
    client, calls = api_client
    response = client.get("/query", params={"query": "介绍一下商品"})
    assert response.status_code == 200
    assert calls == [("介绍一下商品", None, None)]
    assert response.json()["session_id"] == "generated-session"
    assert response.json()["product_id"] is None


def test_get_also_accepts_current_product(api_client):
    client, calls = api_client
    response = client.get(
        "/query",
        params={"query": "功能有哪些？", "product_id": "vivo_x500"},
    )
    assert response.status_code == 200
    assert calls == [("功能有哪些？", None, "vivo_x500")]


@pytest.mark.parametrize("product_id", list(load_products()))
def test_post_accepts_every_catalog_product(api_client, product_id):
    client, calls = api_client
    response = client.post(
        "/query", json={"message": "这款有什么特点？", "product_id": product_id}
    )
    assert response.status_code == 200
    assert response.json()["product_id"] == product_id
    assert calls == [("这款有什么特点？", None, product_id)]


@pytest.mark.parametrize("method", ["get", "post"])
def test_unknown_product_is_rejected_before_agent_call(api_client, method):
    client, calls = api_client
    if method == "get":
        response = client.get(
            "/query", params={"query": "商品有什么功能？", "product_id": "unknown"}
        )
    else:
        response = client.post(
            "/query", json={"message": "商品有什么功能？", "product_id": "unknown"}
        )
    assert response.status_code == 404
    assert response.json()["detail"] == "当前商品不存在。"
    assert calls == []


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"message": " "},
        {"message": 123},
        {"message": "问题", "product_id": " "},
        {"message": "问题", "product_id": 123},
        {"message": "问题", "session_id": " "},
        {"message": "问题", "unexpected": "字段拼写错误"},
    ],
)
def test_post_rejects_invalid_input_before_agent_call(api_client, payload):
    client, calls = api_client
    response = client.post("/query", json=payload)
    assert response.status_code == 422
    assert calls == []


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"query": " "},
        {"query": "问题", "product_id": " "},
        {"query": "问题", "session_id": " "},
    ],
)
def test_get_rejects_empty_fields_before_agent_call(api_client, params):
    client, calls = api_client
    response = client.get("/query", params=params)
    assert response.status_code == 422
    assert calls == []


def test_post_can_omit_product_for_original_whole_library_mode(api_client):
    client, calls = api_client
    response = client.post("/query", json={"message": "介绍一下商品"})
    assert response.status_code == 200
    assert calls == [("介绍一下商品", None, None)]


def test_invalid_catalog_reports_configuration_error_before_agent_call(
    api_client, monkeypatch
):
    client, calls = api_client

    def invalid_catalog():
        raise ValueError("商品表字段不完整。")

    monkeypatch.setattr(main, "load_products", invalid_catalog)
    response = client.post(
        "/query", json={"message": "问题", "product_id": "vivo_x500"}
    )
    assert response.status_code == 500
    assert response.json()["detail"] == "商品表配置错误。"
    assert calls == []


def test_openapi_describes_json_body_and_product_response(api_client):
    client, _ = api_client
    schema = client.get("/openapi.json").json()
    assert {"get", "post"} <= set(schema["paths"]["/query"])
    properties = schema["components"]["schemas"]["QueryRequest"]["properties"]
    assert {"message", "session_id", "product_id"} <= set(properties)
