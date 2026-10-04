"""验证本地 embedding 模型目录与设备参数。"""

from types import SimpleNamespace
import sys

import pytest

from backend.embeddings import create_embeddings


def test_model_path_and_device_can_be_overridden_with_a_local_directory(
    tmp_path, monkeypatch
):
    model_directory = tmp_path / "model"
    model_directory.mkdir()
    monkeypatch.setenv("BGE_M3_PATH", str(tmp_path / "unused"))
    monkeypatch.setenv("EMBEDDING_DEVICE", "mps")
    options = {}
    result = object()

    def fake_embeddings(**kwargs):
        options.update(kwargs)
        return result

    monkeypatch.setitem(
        sys.modules,
        "langchain_huggingface",
        SimpleNamespace(HuggingFaceEmbeddings=fake_embeddings),
    )
    assert create_embeddings(model_path=model_directory, device="cpu") is result
    assert options["model_name"] == str(model_directory.resolve())
    assert options["model_kwargs"]["device"] == "cpu"
    assert options["encode_kwargs"]["normalize_embeddings"] is True


def test_missing_model_directory_is_rejected_before_embedding_initialization(tmp_path):
    with pytest.raises(ValueError, match="directory does not exist"):
        create_embeddings(model_path=tmp_path / "missing")
