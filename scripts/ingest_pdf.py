from app.ingest import extract_pdf_text, chunk_text, embed_text
from app.db import engine
from app.vector_utils import to_pgvector_literal
from sqlalchemy import text
from pathlib import Path
import sys

if len(sys.argv) != 2:
    raise SystemExit("Usage: python -m scripts.ingest_pdf <pdf_path>")

provided_path = sys.argv[1]
pdf_path = str(Path(provided_path).expanduser().resolve())
text_data = extract_pdf_text(pdf_path)
chunks = chunk_text(text_data)

with engine.begin() as conn:
    # Replace this file's old chunks atomically so reruns stay idempotent.
    conn.execute(
        text("DELETE FROM documents WHERE source IN (:source, :provided_source)"),
        {"source": pdf_path, "provided_source": provided_path},
    )
    for i, chunk in enumerate(chunks):
        embedding = embed_text(chunk)
        conn.execute(
            text(
                "INSERT INTO documents (source, chunk_index, content, embedding) "
                "VALUES (:source, :chunk_index, :content, CAST(:embedding AS vector))"
            ),
            {
                "source": pdf_path,
                "chunk_index": i,
                "content": chunk,
                "embedding": to_pgvector_literal(embedding),
            },
        )

print(f"Inserted {len(chunks)} chunks")
