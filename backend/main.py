"""提供正式商品、聊天和存活检查接口，并保留原有问答入口。"""

from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Path, Query
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from backend.citation import Citation
from backend.products import Product, load_products

NonEmptyString = Annotated[
    str, StringConstraints(strict=True, strip_whitespace=True, min_length=1)
]


class QueryRequest(BaseModel):
    """JSON 问答输入；省略商品 ID 时继续使用原有全库问答模式。"""

    model_config = ConfigDict(extra="forbid")

    message: NonEmptyString = Field(description="本轮用户问题")
    session_id: NonEmptyString | None = Field(default=None, description="已有会话编号")
    product_id: NonEmptyString | None = Field(default=None, description="当前商品 ID")
    follow_conversation: bool = Field(
        default=False,
        strict=True,
        description="按会话承接话题和商品指代，例如‘那普通版呢’；默认保留显式商品范围",
    )


class QueryResponse(BaseModel):
    """返回回答、引用、会话编号和本轮接收的商品 ID。"""

    answer: str
    citations: list[Citation]
    session_id: str
    product_id: str | None


class ProductResponse(BaseModel):
    """商品选择和基础信息查询共用的响应，SKU 使用字符串保留前导零。"""

    model_config = ConfigDict(from_attributes=True)

    id: str = Field(description="商品 ID，提问时作为 product_id 传回")
    sku: str = Field(description="当前配置的 SKU 编号，保持字符串")
    name: str = Field(description="商品展示名称")
    category: str = Field(description="商品分类")


class HealthResponse(BaseModel):
    """HTTP 服务存活状态，不代表模型、Redis 或知识库已经就绪。"""

    status: Literal["ok"] = Field(description="当前 HTTP 服务可响应请求")
    service: Literal["flame"] = Field(description="服务名称")


app = FastAPI()


def _load_product_catalog() -> dict[str, Product]:
    """统一读取商品表，配置异常只向客户端返回通用说明。"""
    try:
        return load_products()
    except ValueError as error:
        raise HTTPException(status_code=500, detail="商品表配置错误。") from error


def _get_product(product_id: str) -> Product:
    """按 ID 获取商品，基础信息查询和问答校验共用不存在商品的处理。"""
    products = _load_product_catalog()
    if product_id not in products:
        raise HTTPException(status_code=404, detail="当前商品不存在。")
    return products[product_id]


def _invoke_agent(query, session_id, product_id):
    """输入校验通过后再加载 Agent，非法请求不加载 BGE 或调用聊天服务。"""
    from backend.retrieval import agent_invoke

    return agent_invoke(query=query, session_id=session_id, product_id=product_id)


def _answer_request(request: QueryRequest) -> QueryResponse:
    """正式聊天和兼容问答入口共用商品校验、会话、缓存及引用逻辑。"""
    if request.product_id is not None:
        _get_product(request.product_id)

    product_id = request.product_id
    if request.follow_conversation:
        from backend.retrieval import agent_invoke

        trace = {}
        answer, citations, session_id = agent_invoke(
            request.message,
            request.session_id,
            product_id,
            follow_conversation=True,
            trace=trace,
        )
        product_id = trace.get("product_id", product_id)
    else:
        answer, citations, session_id = _invoke_agent(
            request.message, request.session_id, product_id
        )
    return QueryResponse(
        answer=answer,
        citations=citations,
        session_id=session_id,
        product_id=product_id,
    )


@app.get(
    "/api/products",
    response_model=list[ProductResponse],
    summary="获取商品列表",
    tags=["商品"],
)
def list_products() -> list[Product]:
    """按商品表顺序返回基础信息，供前端选择商品，不加载问答模型。"""
    return list(_load_product_catalog().values())


@app.get(
    "/api/products/{product_id}",
    response_model=ProductResponse,
    summary="获取指定商品的基础信息",
    tags=["商品"],
    responses={
        404: {"description": "当前商品不存在"},
        500: {"description": "商品表配置错误"},
    },
)
def get_product(
    product_id: Annotated[NonEmptyString, Path(description="商品列表中返回的 id")],
) -> Product:
    """只读指定商品的 ID、SKU、名称和分类，不加载模型或检索知识库。"""
    return _get_product(product_id)


@app.post(
    "/api/chat",
    response_model=QueryResponse,
    summary="提交商品聊天消息",
    tags=["聊天"],
    responses={
        404: {"description": "当前商品不存在"},
        500: {"description": "商品表配置错误或问答服务异常"},
    },
)
def chat(request: QueryRequest) -> QueryResponse:
    """接收 JSON 消息并返回回答、引用和会话 ID，复用现有 Agent 链路。"""
    return _answer_request(request)


@app.get(
    "/api/health",
    response_model=HealthResponse,
    summary="检查 HTTP 服务是否存活",
    tags=["服务"],
)
def health() -> HealthResponse:
    """只报告进程能够响应 HTTP，不访问商品表或外部依赖。"""
    return HealthResponse(status="ok", service="flame")


@app.get("/query", response_model=QueryResponse)
def send_query(
    query: Annotated[NonEmptyString, Query(description="本轮用户问题")],
    session_id: Annotated[
        NonEmptyString | None, Query(description="已有会话编号")
    ] = None,
    product_id: Annotated[
        NonEmptyString | None, Query(description="当前商品 ID")
    ] = None,
):
    """保留 GET 问答入口，并支持可选的当前商品 ID。"""
    return _answer_request(
        QueryRequest(message=query, session_id=session_id, product_id=product_id)
    )


@app.post("/query", response_model=QueryResponse)
def send_message(request: QueryRequest):
    """接收 product_id、message、session_id 组成的 JSON 请求体。"""
    return _answer_request(request)
