# Evaluation notes

No corpus evaluation has been recorded for the current document-based setup.
`golden_qa.json` is empty and should be populated with reviewed questions and
verbatim source chunks from the database being evaluated. Empty-set metrics do
not describe retrieval or answer quality.

## Preparing a reference set

1. Index the products with `scripts/build_product_db.py`.
2. Use `eval.chunk_sampling.load_all_chunk_texts()` to read the `products`
   collection in `chroma_db_products/`, then select passages for review.
   `sample_clean_chunks()` uses an English-prose heuristic; select Chinese
   Markdown passages manually until an appropriate sampling rule is added.
3. Draft candidate questions with `eval.qa_generation` and review each question
   for a direct answer in its paired passage.
4. Save accepted question/chunk pairs using `eval.golden_set`.

## Recording results

Record the corpus and ingestion settings, embedding model, selected pipeline,
retriever configuration, sample count, and evaluation date. The product
evaluation retriever uses the same `products` collection and `k=5` similarity
search as the live agent tool. Evaluation should identify any configuration
overrides and whether product filtering was applied.

`run_retrieval_eval()` reports recall, mean reciprocal rank, ranks, and a recall
confidence interval. `run_citation_eval()` reports marker validity, answer
citation coverage, and retrieval agreement with the paired reference chunk.
Those citation metrics do not establish factual correctness.

Matches use exact chunk text. Rechunking changes the reference text and requires
reviewing the set before comparing results. Small reference sets provide a
limited estimate, so include sample size and uncertainty with every report.
