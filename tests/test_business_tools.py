"""验证结构化业务工具及正式 LangGraph 工具节点，离线运行。"""

import json
from pathlib import Path
import uuid

import pytest

from backend import business_tools
from backend.products import load_products


TOOLS = [
    business_tools.product_info_tool,
    business_tools.price_tool,
    business_tools.after_sales_tool,
]


def invoke(tool, **args):
    return tool.invoke(
        {"type": "tool_call", "name": tool.name, "id": uuid.uuid4().hex, "args": args}
    )


@pytest.mark.parametrize("tool", TOOLS)
@pytest.mark.parametrize("product_id", list(load_products()))
def test_business_records_cover_catalog(tool, product_id):
    result = invoke(tool, product_id=product_id, citation_start=7)
    assert result.artifact and result.artifact[0].number == 7
    assert result.artifact[0].source == "product_business.json"
    data = json.loads(result.artifact[0].snippet.split("\n", 1)[1])
    assert data["product_id"] == product_id and data["demo"] is True
    if tool.name == "product_info_tool":
        assert data["colors"] and data["specifications"] and data["variants"]
        assert "price_fen" not in result.content and "policy" not in data
    elif tool.name == "price_tool":
        assert all(type(v["price_fen"]) is int for v in data["variants"])
    else:
        assert "7 天" in data["policy"] and "演示" in result.content


@pytest.mark.parametrize("tool", TOOLS)
def test_business_tools_hide_injected_identity(tool):
    properties = tool.tool_call_schema.model_json_schema()["properties"]
    assert "product_id" not in properties and "citation_start" not in properties


@pytest.mark.parametrize(
    "alias", ["X500 Pro", "vivo X500 Pro", "VIVO X500PRO", "vivo_x500_pro", "100001"]
)
def test_global_mode_resolves_exact_product_without_confusing_max(alias):
    result = invoke(business_tools.product_info_tool, product_name=alias)
    assert '"product_id": "vivo_x500_pro"' in result.content
    assert "vivo_x500_pro_max" not in result.content


@pytest.mark.parametrize(
    "args",
    [
        {},
        {"product_name": "Pro"},
        {"product_id": "missing", "product_name": "X500 Pro"},
    ],
)
def test_unknown_or_unselected_product_never_falls_back(args):
    result = invoke(business_tools.price_tool, **args)
    assert not result.artifact and "price_fen" not in result.content


@pytest.mark.parametrize(
    "variant", ["16+512", "16GB+512GB", "16GB RAM + 512GB ROM", "100001", "月光银"]
)
def test_price_matches_requested_configuration(variant):
    result = invoke(
        business_tools.price_tool, product_id="vivo_x500_pro", variant=variant
    )
    assert '"price_fen": 399900' in result.content


def test_unavailable_configuration_has_no_substitute_price():
    result = invoke(
        business_tools.price_tool, product_id="vivo_x500_pro", variant="12+512"
    )
    assert "暂未查询到指定规格" in result.content and not result.artifact
    assert "3999" not in result.content


@pytest.mark.parametrize(
    "contents",
    [None, "{", "[]", '{"demo":true,"products":[]}', '{"demo":true,"products":{}}'],
)
def test_unavailable_business_data_returns_no_citations(
    tmp_path, monkeypatch, contents
):
    path = tmp_path / "business.json"
    if contents is not None:
        path.write_text(contents, encoding="utf-8")
    monkeypatch.setattr(business_tools, "BUSINESS_DATA_PATH", path)
    result = invoke(business_tools.price_tool, product_id="vivo_x500_pro")
    assert not result.artifact and "price_fen" not in result.content


@pytest.mark.parametrize(
    "name,variant,price",
    [("A1", "5L", 29900), ("R1", "3L", 22900), ("T1", "500mL", 8900)],
)
def test_household_model_and_capacity_price(name, variant, price):
    result = invoke(business_tools.price_tool, product_name=name, variant=variant)
    assert f'"price_fen": {price}' in result.content


def test_business_data_is_consistent_with_markdown_sources():
    data = json.loads(
        Path(business_tools.BUSINESS_DATA_PATH).read_text(encoding="utf-8")
    )
    catalog = load_products()
    assert set(data["products"]) == set(catalog)
    documents = {
        p.read_text(encoding="utf-8")
        .split("product_id: ")[1]
        .splitlines()[0]: p.read_text(encoding="utf-8")
        for p in Path("data/products").glob("*.md")
    }
    for product_id, record in data["products"].items():
        assert record["after_sales"] in documents[product_id]
        for item in record["variants"]:
            assert item["sku"] == catalog[product_id].sku
            assert str(item["price_fen"] // 100) in documents[product_id]
            assert item["color"] in documents[product_id]
