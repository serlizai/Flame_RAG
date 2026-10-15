"""提供兼容 GET 参数和 JSON 请求体的 Flame 问答接口。"""

from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from backend.citation import Citation
from backend.products import load_products

NonEmptyString = Annotated[
    str, StringConstraints(strict=True, strip_whitespace=True, min_length=1)
]


class QueryRequest(BaseModel):
    """JSON 问答输入；省略商品 ID 时继续使用原有全库问答模式。"""

    model_config = ConfigDict(extra="forbid")

    message: NonEmptyString = Field(description="本轮用户问题")
    session_id: NonEmptyString | None = Field(default=None, description="已有会话编号")
    product_id: NonEmptyString | None = Field(default=None, description="当前商品 ID")


class QueryResponse(BaseModel):
    """返回回答、引用、会话编号和本轮接收的商品 ID。"""

    answer: str
    citations: list[Citation]
    session_id: str
    product_id: str | None


app = FastAPI()


def _invoke_agent(query, session_id, product_id):
    """输入校验通过后再加载 Agent，非法请求不加载 BGE 或调用聊天服务。"""
    from backend.retrieval import agent_invoke

    return agent_invoke(query=query, session_id=session_id, product_id=product_id)


def _answer_request(request: QueryRequest) -> QueryResponse:
    """两种请求格式共用商品校验和后端调用逻辑。"""
    if request.product_id is not None:
        try:
            products = load_products()
        except ValueError as error:
            raise HTTPException(status_code=500, detail="商品表配置错误。") from error
        if request.product_id not in products:
            raise HTTPException(status_code=404, detail="当前商品不存在。")

    answer, citations, session_id = _invoke_agent(
        request.message, request.session_id, request.product_id
    )
    return QueryResponse(
        answer=answer,
        citations=citations,
        session_id=session_id,
        product_id=request.product_id,
    )


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
