"""旧版第九周完整真实API验收，默认不触发外部服务。"""

import os
import pytest


@pytest.mark.skipif(
    os.getenv("RUN_AGENT_INTEGRATION") != "1",
    reason="设置 RUN_AGENT_INTEGRATION=1 后运行第九周统一真实服务验收。",
)
def test_week9_formal_api_acceptance():
    from scripts.check_week9_acceptance import run_acceptance, runtime

    with runtime() as env:
        report = run_acceptance(env)
    assert report["passed"], [
        (row["name"], row["errors"]) for row in report["results"] if not row["passed"]
    ]
