# Flame

中文开发记录见 [项目整理与商品知识库](Flame_Development_Notes.md)，包含已完成修改、配置、建库流程、验证结果和待办。

Flame is a document-based AI assistant. It retrieves relevant passages from a
local vector database, answers questions using those passages, and exposes
numbered citations with source text.

## Components

The Streamlit interface in `app.py` calls the LangGraph backend. The backend can
retrieve document chunks, generate an answer, and resolve its citation markers.
The same backend is available through FastAPI in `backend/main.py`.

`Flame` in `flame_main.py` provides a classic RAG interface. Both pipelines share
`prompts.py`, the citation model, and Redis cache utilities.

PDF ingestion uses PyMuPDF, overlapping text chunks, local BGE embeddings, and
Chroma. Source titles come from the PDF filename or `--source-name`. Page labels
and other document metadata can be used to locate supporting passages. No
external PDF is downloaded by the ingestion script.

## Setup and use

```bash
uv sync
cp .env.example .env
```

Configure the chat API and local embedding model in `.env`. Start Redis before
asking questions. See [the contribution guide](../CONTRIBUTING.md) for service
setup and environment details.

```bash
uv run python -m scripts.build_bge_db path/to/document.pdf --source-name "Product Guide"
uv run streamlit run app.py
```

The default vector database is `chroma_db/` at the project root. Set
`CHROMA_PERSIST_DIRECTORY` to select another corpus. Use the same setting for
indexing, the application, and evaluation. Index your own documents before
asking document questions.

For an optional source link:

```bash
uv run python -m scripts.build_bge_db path/to/document.pdf --source-url https://example.com/document.pdf
```

The script adds documents to the selected database without clearing existing
entries. Use a separate directory when building a new corpus.

## API

```bash
uv run uvicorn backend.main:app --reload
```

`GET /query` accepts `query` and an optional `session_id`. It returns `answer`,
`citations`, and `session_id`. Reuse the session ID for follow-up questions.

## Product metadata

The product Markdown files in `data/products/` have a first-level product title,
`SKU`, and `product_id` fields. Prepare their chunks with:

```bash
uv run python -m scripts.prepare_product_chunks
```

Every exported chunk includes `product_id`, `sku_id` (a string), `product_name`,
`category`, and the source filename. Section names and chunk IDs are also
preserved. The current defaults are 800 characters per chunk, 100 characters of
overlap, and category `smartphone`. Optional flags include `--products-directory`,
`--output`, `--category`, `--chunk-size`, and `--chunk-overlap`.

The output, `data/product_chunks.jsonl`, contains both text and metadata. It is a
local preparation artifact, not a vector database. See
[the metadata and progress notes](Week2_Metadata_and_Progress.md).

## Product database build

```bash
uv run python -m scripts.build_product_db --model-path /path/to/bge-m3
```

This reads `data/products/*.md`, reuses the product parser and text splitter,
generates embeddings with local BGE, and persists them to `chroma_db_products/`.
The collection name is `products`. `--model-path` overrides `BGE_M3_PATH`;
`--device` overrides `EMBEDDING_DEVICE`, which defaults to `cpu`.

For a preview requiring no embedding model:

```bash
uv run python -m scripts.build_product_db --dry-run
```

All product metadata is preserved. Stored chunks also carry an
`ingestion_pipeline` marker identifying records managed by the builder. Rebuilds
upsert reproducible IDs and remove obsolete managed records after the new write
succeeds. Other records, collections, and files are preserved.

The backend still uses its existing generic database configuration. In the next
plan step, configure `product_vector_store` with both the product database path
and `collection_name="products"`.

## Evaluation

`eval/` contains corpus sampling, candidate question drafting, retrieval metrics,
and citation metrics. The reference QA file starts empty; populate it with
reviewed questions and exact chunk text from the selected corpus. See
[the evaluation notes](../eval/RESULTS.md) for the reporting requirements.
