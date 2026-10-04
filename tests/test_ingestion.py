"""验证 PDF 建库、来源引用和已有数据保留。"""

from pathlib import Path

import pymupdf
import pytest
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from backend import embeddings as embedding_config
from citations import annotate_documents_for_citation
from eval.chunk_sampling import load_all_chunk_texts
from scripts.build_bge_db import main
import settings


class OfflineEmbeddings(Embeddings):
    """用于本地存储验证的简单 embedding 替身，不用于评估检索质量。"""

    def embed_documents(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0, 0.0]


def write_pdf(path: Path, text: str):
    with pymupdf.open() as document:
        document.new_page().insert_text((72, 72), text)
        document.save(path)


def test_indexes_user_pdfs_with_citations_and_preserves_existing_data(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ANONYMIZED_TELEMETRY", "False")
    monkeypatch.setattr(settings, "PROJECT_ROOT", tmp_path)
    embeddings = OfflineEmbeddings()
    monkeypatch.setattr(embedding_config, "create_embeddings", lambda: embeddings)

    database_path = tmp_path / "index"
    database_path.mkdir()
    marker = database_path / "keep.txt"
    marker.write_text("existing data")
    first_pdf = tmp_path / "guide.pdf"
    second_pdf = tmp_path / "maintenance.pdf"
    write_pdf(first_pdf, "Connect the device to the power supply before starting it.")
    write_pdf(
        second_pdf, "Back up project documents before replacing the storage device."
    )

    working_directory = tmp_path / "notebooks"
    working_directory.mkdir()
    monkeypatch.chdir(working_directory)
    main(
        [
            str(first_pdf),
            "--source-name",
            "Product Guide",
            "--source-url",
            "https://example.com/guide.pdf",
            "--persist-directory",
            "index",
        ]
    )
    main([str(second_pdf), "--persist-directory", "index"])

    stored = Chroma(
        persist_directory=str(database_path), embedding_function=embeddings
    ).get()
    assert len(stored["documents"]) == 2
    by_title = dict(
        zip(
            [metadata["source_name"] for metadata in stored["metadatas"]],
            stored["metadatas"],
        )
    )
    assert by_title["Product Guide"]["source"] == "https://example.com/guide.pdf"
    assert by_title["maintenance"]["source"] == second_pdf.as_uri()
    assert marker.read_text() == "existing data"
    assert not (working_directory / "index").exists()
    assert set(load_all_chunk_texts(str(database_path))) == set(stored["documents"])

    from langchain_core.documents import Document

    _, citations = annotate_documents_for_citation(
        [
            Document(page_content=text, metadata=metadata)
            for text, metadata in zip(stored["documents"], stored["metadatas"])
        ]
    )
    assert len(citations) == 2
    assert {citation.label for citation in citations} == {
        "Product Guide, page 1",
        "maintenance, page 1",
    }


def test_missing_pdf_fails_before_loading_an_embedding_model(tmp_path, monkeypatch):
    def unexpected_model_load():
        raise AssertionError("A missing PDF must not load the model.")

    monkeypatch.setattr(embedding_config, "create_embeddings", unexpected_model_load)
    with pytest.raises(SystemExit) as error:
        main([str(tmp_path / "missing.pdf")])

    assert error.value.code == 2
