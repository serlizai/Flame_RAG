"""验证检索脚本的直接执行和模块执行入口，防止项目配置导入失败。"""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("direct_file", [True, False])
def test_retrieval_entry_point_loads_project_settings(tmp_path, direct_file):
    source_root = Path(__file__).resolve().parents[1]
    missing_database = tmp_path / "missing-products"
    entry_point = (
        [str(source_root / "scripts" / "test_retrieval.py")]
        if direct_file
        else ["-m", "scripts.test_retrieval"]
    )
    environment = os.environ.copy()
    # 去掉外部导入路径，确保直接执行时由脚本自行定位项目配置。
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            *entry_point,
            "--persist-directory",
            str(missing_database),
        ],
        cwd=tmp_path if direct_file else source_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )

    # 到达数据库校验说明 settings 已导入，且没有加载模型或创建空库。
    assert result.returncode == 2
    assert "商品库不存在" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr
    assert not missing_database.exists()
