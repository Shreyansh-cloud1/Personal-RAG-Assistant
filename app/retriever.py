import os

import psycopg2
from dotenv import load_dotenv
from app.vector_utils import to_pgvector_literal

load_dotenv()


def search(query_embedding, top_k=5):
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT source, chunk_index, content,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM documents
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (to_pgvector_literal(query_embedding), to_pgvector_literal(query_embedding), top_k),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    MIN_SIMILARITY = 0.70
    return [
        (source, chunk_index, content, similarity)
        for source, chunk_index, content, similarity in rows
        if similarity >= MIN_SIMILARITY
    ]
