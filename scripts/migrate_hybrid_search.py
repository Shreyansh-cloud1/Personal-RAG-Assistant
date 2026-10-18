"""Create the PostgreSQL full-text index used by hybrid retrieval."""

from sqlalchemy import text

from app.db import engine


with engine.begin() as conn:
    conn.execute(
        text(
            """
            CREATE INDEX IF NOT EXISTS idx_documents_content_fts
            ON documents
            USING GIN (to_tsvector('english', COALESCE(content, '')))
            """
        )
    )

print("Hybrid search full-text index is ready.")
