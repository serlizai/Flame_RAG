"""商品、价格及售后演示接口；直接读取结构化数据，不进行向量检索。"""

import json
import re
from pathlib import Path
from typing import Annotated

from langchain.tools import InjectedToolArg, tool
from langchain_core.documents import Document

from backend.citation import annotate_documents_for_citation, CITATION_MARKER_KEY
from backend.products import load_products
from settings import PROJECT_ROOT

BUSINESS_DATA_PATH = PROJECT_ROOT / "data" / "product_business.json"


def _normalise_name(value):
    return re.sub(r"\s+", "", value).casefold()


def _lookup(product_id, product_name):
    catalog = load_products()
    if product_id is None:
        if not product_name:
            return None, None, "请先指定商品名称或选择当前商品。"
        matches = [
            product
            for product in catalog.values()
            if _normalise_name(product_name)
            in {
                _normalise_name(product.id),
                _normalise_name(product.name),
                _normalise_name(
                    product.name.removeprefix("vivo ").removeprefix("Flame ")
                ),
                _normalise_name(product.name.split()[1])
                if product.name.startswith("Flame ")
                else _normalise_name(product.name),
                product.sku,
            }
        ]
        if len(matches) != 1:
            return None, None, "暂未查询到该商品，请提供商品列表中的完整名称。"
        product_id = matches[0].id
    if product_id not in catalog:
        return None, None, "暂未查询到当前商品，不会查询其他商品代替。"
    payload = json.loads(Path(BUSINESS_DATA_PATH).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("demo") is not True:
        raise ValueError("当前业务接口只支持已标明的演示数据。")
    if not isinstance(payload.get("products"), dict):
        raise ValueError("Invalid business product table")
    record = payload["products"].get(product_id)
    if not record:
        return None, None, "暂未查询到当前商品的业务数据。"
    if not isinstance(record, dict):
        raise ValueError("Invalid business product record")
    return catalog[product_id], record, None


def _result(kind, product_id, product_name, citation_start, variant=None):
    try:
        product, record, error = _lookup(product_id, product_name)
        if error:
            return error, []
        data = {"product_id": product.id, "product_name": product.name, "demo": True}
        variants = record["variants"]
        if not isinstance(variants, list) or any(
            not isinstance(item, dict)
            or any(
                not isinstance(item.get(key), str) or not item[key].strip()
                for key in ("sku", "specification", "color")
            )
            for item in variants
        ):
            raise ValueError("Invalid business variants")
        if kind == "商品信息":
            if (
                not isinstance(record["colors"], list)
                or not record["colors"]
                or any(
                    not isinstance(color, str) or not color.strip()
                    for color in record["colors"]
                )
                or not isinstance(record["specifications"], dict)
            ):
                raise ValueError("Invalid product information")
            data.update(
                category=product.category,
                colors=record["colors"],
                specifications=record["specifications"],
                variants=[
                    {k: v for k, v in item.items() if k != "price_fen"}
                    for item in variants
                ],
            )
        elif kind == "价格":
            if variant:

                def normalise(value):
                    return re.sub(
                        r"\s|RAM|ROM|GB", "", value.upper().replace("1TB", "1024GB")
                    )

                variants = [
                    item
                    for item in variants
                    if normalise(variant) == normalise(item["specification"])
                    or variant.strip() == item["sku"]
                    or normalise(variant) == normalise(item["color"])
                ]
            if not variants:
                return "暂未查询到指定规格的价格，不能用其他规格的价格代替。", []
            if any(
                type(item["price_fen"]) is not int or item["price_fen"] < 0
                for item in variants
            ):
                raise ValueError("Invalid demo price")
            data.update(currency="CNY", variants=variants)
        else:
            if (
                not isinstance(record["after_sales"], str)
                or not record["after_sales"].strip()
            ):
                raise ValueError("Invalid after-sales policy")
            data["policy"] = record["after_sales"]
        text = "演示业务接口结果（不代表真实在售商品或商业承诺）：\n" + json.dumps(
            data, ensure_ascii=False
        )
        annotated, citations = annotate_documents_for_citation(
            [
                Document(
                    page_content=text,
                    metadata={
                        "product_name": product.name,
                        "section": kind + "接口（演示）",
                        "source": "product_business.json",
                    },
                )
            ],
            start_number=citation_start,
        )
        return f"{annotated[0].metadata[CITATION_MARKER_KEY]}\n{text}", citations
    except (OSError, ValueError, KeyError, TypeError):
        return (
            "商品业务接口暂不可用，无法确认相关信息，请稍后重试；不要改用 RAG 猜测价格或售后。",
            [],
        )


@tool(response_format="content_and_artifact")
def product_info_tool(
    product_name: str | None = None,
    product_id: Annotated[str | None, InjectedToolArg] = None,
    citation_start: Annotated[int, InjectedToolArg] = 1,
):
    """查询商品颜色、可选配置、SKU 和基础参数表，包括容量、功率、材质、尺寸、内存与存储。

    “容量和功率是多少”等基础参数问题必须使用此工具，不走 RAG。
    不查询价格、售后、卖点或使用步骤。

    已选择商品时由服务端限定身份；未选择时，product_name 必须提供完整商品名或型号。
    """
    return _result("商品信息", product_id, product_name, citation_start)


@tool(response_format="content_and_artifact")
def price_tool(
    variant: str | None = None,
    product_name: str | None = None,
    product_id: Annotated[str | None, InjectedToolArg] = None,
    citation_start: Annotated[int, InjectedToolArg] = 1,
):
    """查询商品价格，所有价格、报价、多少钱问题必须调用本工具，不能从 RAG 或历史报价。

    variant 为用户指定的配置（例如 12+512、16GB+512GB）、颜色或 SKU；没有指定可省略。
    未选择当前商品时，product_name 必须提供完整商品名或型号。返回 price_fen，单位为人民币分，除以100为元。
    未提供的配置不报价；所有数据仅为演示价格。
    """
    return _result("价格", product_id, product_name, citation_start, variant)


@tool(response_format="content_and_artifact")
def after_sales_tool(
    product_name: str | None = None,
    product_id: Annotated[str | None, InjectedToolArg] = None,
    citation_start: Annotated[int, InjectedToolArg] = 1,
):
    """查询退货、换货、保修条件及售后流程，售后问题必须调用本工具，不从 RAG 推断。

    未选择当前商品时，product_name 必须提供完整商品名或型号。仅返回演示规则，不能承诺实际订单符合条件。
    """
    return _result("售后", product_id, product_name, citation_start)
