"""通过真实 FastAPI 检验第六周工具和第七周上下文，不依赖展示页面。"""

import os

from fastapi.testclient import TestClient
import pytest


@pytest.mark.skipif(
    os.getenv("RUN_AGENT_INTEGRATION") != "1",
    reason="设置 RUN_AGENT_INTEGRATION=1 后验证真实 API 业务工具和多轮对话。",
)
@pytest.mark.parametrize("suite", ["week6", "week7"])
def test_real_api_business_tools_and_independent_memory(suite):
    from backend.citation import Citation
    from backend.main import app
    from scripts.check_agent_acceptance import run_acceptance, runtime

    _, inspect, cleanup = runtime()
    with TestClient(app) as client:

        def invoke(
            question, session, product_id, *, trace, use_cache, follow_conversation
        ):
            from backend import retrieval

            original = retrieval.agent_invoke

            def instrument(*args, **kwargs):
                # API 还需要此对象来返回解析后的商品身份，不能替换它。
                actual_trace = kwargs.setdefault("trace", {})
                kwargs["use_cache"] = use_cache
                result = original(*args, **kwargs)
                trace.update(actual_trace)
                return result

            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(retrieval, "agent_invoke", instrument)
                response = client.post(
                    "/api/chat",
                    json={
                        "message": question,
                        "session_id": session,
                        "product_id": product_id,
                        "follow_conversation": follow_conversation,
                    },
                )
            response.raise_for_status()
            payload = response.json()
            assert set(payload) == {"answer", "session_id", "product_id", "citations"}
            assert payload["product_id"] == trace["product_id"]
            return (
                payload["answer"],
                [Citation(**c) for c in payload["citations"]],
                payload["session_id"],
            )

        report = run_acceptance(invoke, suite, inspect=inspect, cleanup=cleanup)
    assert report["passed"], [(r["name"], r["errors"]) for r in report["results"]]
