# Contributing to Flame

Flame answers questions using user-supplied documents. Contributions should keep
answers grounded in retrieved passages and make their sources easy to inspect.
See the Chinese [README](README.md) for installation and the complete service flow.

## Local setup

Use Python 3.11 or newer. From the project root:

```bash
uv sync --locked --python 3.11
```

Copy `.env.example` to `.env` on first setup; do not overwrite an existing `.env`.
Set `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `BGE_M3_PATH` in `.env` for your
configured chat service and local embedding model. The chat model is configured
in `backend/config.py`. `EMBEDDING_DEVICE` defaults to `cpu`; set it to a device
supported by your machine if needed.

Start Redis, then build the product database and launch the app:

```bash
docker run -d --name flame-redis -p 6379:6379 redis:7
uv run python -m scripts.build_product_db
uv run streamlit run app.py
```

The Streamlit and FastAPI agent uses `backend.config.product_vector_store`,
which opens `chroma_db_products/` at the project root, collection `products`.
Product evaluation uses the same product collection and top-5 retrieval as the
agent. The optional PDF ingestion script uses `CHROMA_PERSIST_DIRECTORY` from
`.env`, defaulting to `chroma_db/`. Relative PDF database paths resolve from the
project root. Vector databases,
generated data, model weights, credentials, and local environment files are
gitignored. The example Markdown files in `data/products/` can be versioned.

Flame keeps its Redis cache implementation in `backend/cache.py`, using its own
key namespace for cached answers and chat transcripts.

The ingestion script accepts an optional `--source-url` for citations and
`--persist-directory` for a specific database. It adds documents to that
database. Use a fresh directory to build a separate corpus.

To run the API:

```bash
uv run uvicorn backend.main:app --reload
curl "http://127.0.0.1:8000/query?query=What+documents+are+available%3F"
```

## Project layout

- `app.py`: Streamlit chat interface.
- `backend/`: LangGraph agent, retrieval tool, API, and service configuration.
- `backend/embeddings.py`: shared embedding factory for ingestion and retrieval.
- `backend/product_documents.py`: product Markdown parsing and chunk metadata.
- `settings.py`: shared environment loading and vector database path.
- `prompts.py`: canonical agent prompts.
- `config/prompts.yaml`: equivalent prompts for configuration-based integrations.
- `citations.py`: generic document labels, citation numbering, and cache payloads.
- `scripts/build_bge_db.py`: command-line PDF ingestion.
- `scripts/prepare_product_chunks.py`: export product chunks and metadata as JSONL.
- `scripts/build_product_db.py`: build and synchronize the product Chroma collection with local BGE.
- `scripts/test_retrieval.py`: run the three basic product retrieval checks with real local BGE.
- `src/pdf_ingestion.ipynb`: notebook example using the ingestion script.
- `eval/`: chunk sampling, question drafting, and retrieval/citation metrics.
- `tests/`: unit tests and an opt-in backend smoke test.

LangGraph is the only question-answering architecture. Classic RAG entry points,
their conversation notebook, the separate CrewAI example, and unused output
schemas have been removed. The remaining PDF notebook is an ingestion utility.

## Product document preparation

Product metadata is read from each Markdown file's title, `SKU`, and `product_id`
header. Every chunk keeps `product_id`, `sku_id`, `product_name`, `category`, and
`source`, plus its section, sequence number, and reproducible chunk ID. SKU values
remain strings. The default category is `smartphone`.

```bash
uv run python -m scripts.prepare_product_chunks
```

This writes `data/product_chunks.jsonl`, using a maximum chunk size of 800
characters and 100-character overlap where section length permits. Sections are
kept separate. This preparation step does not load BGE or write Chroma. See
[the metadata and progress notes](docs/Week2_Metadata_and_Progress.md).

## Product vector database

Set `BGE_M3_PATH` to an existing local model directory in `.env`, or pass the
model directory explicitly:

```bash
uv run python -m scripts.build_product_db --model-path /path/to/bge-m3
```

The script reads the Markdown files directly, retains their product metadata,
and stores their BGE embeddings in `chroma_db_products/`, collection `products`.
The defaults remain `chunk_size=800` and `chunk_overlap=100`. It does not require
a chat API key or Redis.

```bash
uv run python -m scripts.build_product_db --dry-run
uv run python -m scripts.build_product_db --device mps
```

`--dry-run` validates and previews without loading BGE or creating a database.
Additional flags include `--products-directory`, `--persist-directory`,
`--category`, `--chunk-size`, and `--chunk-overlap`. Relative data and explicit
model paths resolve from the project root.

Repeated builds upsert the same chunk IDs. After new chunks are successfully
written, obsolete entries from this script are removed, including entries for
product files removed from the input directory. Unmanaged records, other
collections, and unrelated files are preserved. An empty or invalid product
directory is rejected before database writes.

The backend now opens the same product database and collection. The existing
`retrieve_docs` tool queries `product_vector_store`; its name and the graph
workflow remain unchanged at this stage. Custom builder database paths must also
be applied in `backend/config.py`.

Run the week-two basic retrieval checks without a chat API or Redis:

```bash
uv run python -m scripts.test_retrieval --device cpu
```

The script checks the first result's product for single-product questions and
both product IDs in the top 5 for the comparison. It prints source details and
separate section diagnostics. All basic checks passing returns 0; any failure
returns 1. Missing data or invalid arguments returns 2. Current basic recall is
3/3; dedicated selling-point and specification sections still need improvement.
Optional flags are `--k`, `--persist-directory`, `--model-path`, and `--device`.
Add `--filter-products` to restrict each case to its known target products.
Single-product cases use metadata equality; the comparison uses `$in` for both
products. This exercises existing metadata without rebuilding the database.
The flag affects this test script; the backend still needs API/State integration
to receive the current product ID.
The script also supports `uv run python scripts/test_retrieval.py`. Direct file
execution resolves project imports from the script location, including when an
absolute script path is launched from another working directory.

## Verification

```bash
uv run pytest tests/ -q
uv run ruff check .
uv run ruff format --check .
```

Unit tests use generic document fixtures and run without Redis, a chat API key,
or a loaded embedding model. The backend smoke test runs only when
`RUN_AGENT_INTEGRATION=1`; it requires configured services and an indexed corpus.

For corpus evaluation, prepare reviewed question/chunk pairs from your actual
database. `eval/golden_qa.json` starts empty. Record the corpus, embedding model,
retriever settings, sample count, and metrics with each evaluation. Exact chunk
text determines the retrieval match, so rebuilding or rechunking a corpus
requires reviewing the reference set.

## Conversation state

The agent uses a process-local LangGraph checkpointer for working memory. Redis
stores chat transcripts and cached answers. The compiled graph is shared, and
`session_id` selects a conversation. Pass only the new message to the graph,
since the message reducer appends messages.

Current limitations include loss of agent memory on process restart, a backend
answer cache shared across sessions, and citation numbering that can collide
across retrieval calls. Describe effects on follow-up questions and restarts
when changing these paths.

## Changes and review

Keep changes focused. Describe the resulting behavior, affected pipeline, and
validation in a pull request. Retrieval or prompt changes should report corpus
evaluation results when an appropriate reviewed reference set is available.
Every inline citation must resolve to a retrieved passage. Keep credentials and
personal data out of code, notebook outputs, and committed fixtures.
