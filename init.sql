CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    source TEXT,
    chunk_index INT,
    content TEXT,
    embedding VECTOR(384)
);

CREATE INDEX IF NOT EXISTS idx_documents_embedding
ON documents
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_documents_content_fts
ON documents
USING GIN (to_tsvector('english', COALESCE(content, '')));
