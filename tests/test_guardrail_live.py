"""通过正式 FastAPI 和真实模型验收第九周常见注入场景，默认按条件跳过。"""

import os

from fastapi.testclient import TestClient
import pytest


@pytest.mark.skipif(
    os.getenv("RUN_AGENT_INTEGRATION") != "1",
    reason="设置 RUN_AGENT_INTEGRATION=1 后执行真实模型注入验收。",
)
def test_real_api_guardrails_with_controlled_evidence():
    from backend import retrieval
    from backend.main import app
    from backend.citation import Citation
    from scripts.check_guardrail_acceptance import run_acceptance, runtime

    _, evidence, inspect, cleanup = runtime()
    original = retrieval.agent_invoke
    with TestClient(app) as client:

        def invoke(
            question,
            session,
            product_id,
            *,
            trace,
            use_cache,
            follow_conversation=False,
        ):
            def instrument(*args, **kwargs):
                actual = kwargs.setdefault("trace", {})
                kwargs["use_cache"] = use_cache
                result = original(*args, **kwargs)
                trace.update(actual)
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
            data = response.json()
            assert set(data) == {"answer", "citations", "session_id", "product_id"}
            assert data["product_id"] == trace["product_id"]
            return (
                data["answer"],
                [Citation(**c) for c in data["citations"]],
                data["session_id"],
            )

        report = run_acceptance(
            invoke, evidence=evidence, inspect=inspect, cleanup=cleanup
        )
    assert report["passed"], [(row["name"], row["errors"]) for row in report["results"]]
