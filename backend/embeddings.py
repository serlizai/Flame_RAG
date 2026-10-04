"""为建库、检索和评估创建一致的本地文本向量模型。"""

import os
from pathlib import Path


def create_embeddings(model_path: str | Path | None = None, device: str | None = None):
    model_path = model_path or os.getenv("BGE_M3_PATH")
    if not model_path:
        raise ValueError("Set BGE_M3_PATH to the local embedding model directory.")
    model_directory = Path(model_path).expanduser().resolve()
    if not model_directory.is_dir():
        raise ValueError(
            f"Local embedding model directory does not exist: {model_directory}"
        )

    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=str(model_directory),
        model_kwargs={"device": device or os.getenv("EMBEDDING_DEVICE", "cpu")},
        encode_kwargs={"normalize_embeddings": True},
    )
