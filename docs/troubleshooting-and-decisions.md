# RAG assistant troubleshooting and implementation notes

This note records the retrieval issues fixed in this project, the reasons for the changes, and the steps to follow when updating documents or diagnosing weak answers.

## Current retrieval flow

1. The API validates the question and creates a query embedding with the BGE query instruction.
2. PostgreSQL retrieves candidates using both pgvector cosine similarity and English full-text search.
3. Reciprocal Rank Fusion (RRF) combines the two rankings. The configured RRF constant is `60`.
4. The LLM receives retrieved chunks with stable source labels such as `[S1]` and is asked to answer from that context.
5. The API returns the answer and source metadata. The Streamlit UI displays the source chunk, similarity, and hybrid ranking details.

Relevant code: [retriever](../app/retriever.py), [API](../app/api.py), [LLM prompt](../app/llm.py), and [Streamlit UI](../ui/streamlit_app.py).

## Issues fixed

### Retriever failed while unpacking database rows

**Error**

```text
ValueError: too many values to unpack (expected 2, got 4)
```

**Cause:** The SQL query returned more columns (source, chunk index, content, and similarity) than the Python list comprehension expected. Python tried to unpack a four-column row into two variables.

**Fix:** The retriever was changed to consume the actual row shape. It now returns named metadata in dictionaries, which also made it easier to expose source details in the UI. When changing a SELECT list, keep the row mapping in `app/retriever.py` aligned with it.

### Relevant documents were retrieved, but the answer said “I don't know”

**Cause:** A hard minimum vector-similarity cutoff discarded lower-ranked passages even when they were among the best matches. The document’s explanation also crossed a chunk boundary, so a useful continuation could be missed by a restrictive cutoff.

**Fix:** The hard `MIN_SIMILARITY` filter was removed. Retrieval keeps the best-ranked candidates, and hybrid search now combines semantic and lexical rankings. The answer remains constrained to the supplied context; if that context does not contain the answer, the assistant should say so.

**When debugging this symptom:** Inspect the exact retrieved chunk text and ranking details first. Check whether the answer is in a neighboring chunk, whether the query uses the document’s terminology, and whether ingestion completed against the database the app is using.

### Answers and displayed sources were difficult to verify

**Cause:** The earlier prompt and UI did not make the relationship between an answer and its supporting passages clear enough.

**Fix:** The BGE query instruction is applied when embedding questions. Retrieved passages receive source labels in the prompt, and the UI shows source metadata and retrieval scores in expandable source panels. Check that claims in an answer are supported by those displayed chunks.

### Re-ingesting a PDF could leave duplicate chunks

**Cause:** A source path can be represented differently (for example, relative versus resolved/canonical path), so deleting only one representation may leave old rows behind.

**Fix:** The ingestion script removes existing rows for both the supplied path and its resolved path before inserting the new chunks. Re-ingesting the same PDF is intended to replace its indexed contents.

## Hybrid search and database index

Hybrid retrieval uses:

- **Semantic search:** pgvector similarity against stored document embeddings.
- **Keyword search:** PostgreSQL English full-text search using `websearch_to_tsquery` and `ts_rank_cd`.
- **Fusion:** Reciprocal Rank Fusion with `k = 60`, combining the two result lists.

The full-text search uses a GIN expression index on `to_tsvector('english', COALESCE(content, ''))`.

For a fresh database, the index definition is in [init.sql](../init.sql). For an existing database, apply the idempotent migration from the repository root:

```powershell
python -m scripts.migrate_hybrid_search
```

The migration was applied successfully to the development database during this work. Hybrid search does **not** require regenerating embeddings or re-ingesting existing documents: the new index is built from stored chunk text. Re-ingest only when document contents or chunking have changed.

## Updating the sample PDF or adding documents

After changing `data/sample.pdf` or adding another supported PDF, ingest the file from the repository root:

```powershell
python -m scripts.ingest_pdf data/sample.pdf
```

Use the path to the PDF you want to ingest. Re-ingestion replaces that source’s existing chunks. Ensure the command and running API use the same database configuration.

## Runtime troubleshooting checklist

1. Confirm PostgreSQL is running and the application’s database settings point to the intended database.
2. Confirm the database schema and hybrid full-text index are initialized; run the migration above for an existing database if needed.
3. If you changed Python code, restart the API and Streamlit processes so they load it.
4. If you changed or added source documents, run the ingestion command. A code restart alone does not refresh indexed document content.
5. Inspect retrieved passages and their ranks before changing prompts or similarity settings. This separates retrieval problems from answer-generation problems.
6. Check API and application logs for connection, embedding-provider, or LLM errors. The API reports upstream LLM failures as HTTP 502; blank questions are rejected during validation.

## Verification recorded

- `git diff --check` passed after the hybrid-search changes.
- `python -m scripts.migrate_hybrid_search` completed and reported that the hybrid full-text index was ready.
- End-to-end API/UI checks were not run as part of those changes.

## Privacy note

Document ingestion and vector/full-text retrieval use the configured PostgreSQL database. When Gemini answer generation is enabled, the question and retrieved context are sent to Google’s Gemini API. Do not put secrets in indexed documents or prompts.
