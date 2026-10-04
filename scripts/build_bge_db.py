"""根据用户指定的本地 PDF 构建向量库。

在项目根目录执行：
    python -m scripts.build_bge_db path/to/document.pdf
"""

import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf_path", type=Path, help="Local PDF to ingest")
    parser.add_argument("--source-name", help="Document title used in citations")
    parser.add_argument("--source-url", help="Optional source link for citations")
    parser.add_argument(
        "--persist-directory", type=Path, help="Vector database directory"
    )
    args = parser.parse_args(argv)

    pdf_path = args.pdf_path.expanduser().resolve()
    if not pdf_path.is_file():
        parser.error(f"PDF does not exist: {pdf_path}")

    from settings import CHROMA_PERSIST_DIRECTORY, PROJECT_ROOT
    from backend.embeddings import create_embeddings

    try:
        embeddings = create_embeddings()
    except ValueError as error:
        parser.error(str(error))

    from langchain_chroma import Chroma
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    database_path = (
        args.persist_directory.expanduser()
        if args.persist_directory is not None
        else Path(CHROMA_PERSIST_DIRECTORY)
    )
    if not database_path.is_absolute():
        database_path = PROJECT_ROOT / database_path
    database_path = database_path.resolve()

    documents = PyMuPDFLoader(str(pdf_path)).load()
    for document in documents:
        document.metadata.update(
            source_name=args.source_name or pdf_path.stem,
            source=args.source_url or pdf_path.as_uri(),
            db_owner="Flame",
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    if not chunks:
        parser.error("The PDF contains no extractable text.")

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(database_path),
    )
    print(f"Stored {len(chunks)} chunks in {database_path}")


if __name__ == "__main__":
    main()
