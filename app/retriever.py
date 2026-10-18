import os

import psycopg2
from dotenv import load_dotenv

from app.vector_utils import to_pgvector_literal

load_dotenv()

RRF_K = 60


def search(query_embedding, question, top_k=5):
    """Combine semantic and PostgreSQL full-text results with RRF."""
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )

    candidate_limit = max(top_k * 4, 20)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH semantic_candidates AS (
                    SELECT d.id,
                           1 - (d.embedding <=> %(query_embedding)s::vector) AS vector_similarity
                    FROM documents AS d
                    WHERE d.embedding IS NOT NULL
                    ORDER BY d.embedding <=> %(query_embedding)s::vector
                    LIMIT %(candidate_limit)s
                ),
                semantic_ranked AS (
                    SELECT id,
                           vector_similarity,
                           row_number() OVER (ORDER BY vector_similarity DESC) AS semantic_rank
                    FROM semantic_candidates
                ),
                keyword_candidates AS (
                    SELECT d.id,
                           ts_rank_cd(
                               to_tsvector('english', COALESCE(d.content, '')),
                               websearch_to_tsquery('english', %(question)s)
                           ) AS keyword_score
                    FROM documents AS d
                    WHERE to_tsvector('english', COALESCE(d.content, ''))
                          @@ websearch_to_tsquery('english', %(question)s)
                    ORDER BY keyword_score DESC
                    LIMIT %(candidate_limit)s
                ),
                keyword_ranked AS (
                    SELECT id,
                           row_number() OVER (ORDER BY keyword_score DESC) AS keyword_rank
                    FROM keyword_candidates
                ),
                ranked_candidates AS (
                    SELECT id,
                           vector_similarity,
                           semantic_rank,
                           NULL::bigint AS keyword_rank,
                           semantic_rank AS rrf_rank
                    FROM semantic_ranked
                    UNION ALL
                    SELECT id,
                           NULL::double precision AS vector_similarity,
                           NULL::bigint AS semantic_rank,
                           keyword_rank,
                           keyword_rank AS rrf_rank
                    FROM keyword_ranked
                ),
                fused AS (
                    SELECT id,
                           MAX(vector_similarity) AS vector_similarity,
                           MIN(semantic_rank) AS semantic_rank,
                           MIN(keyword_rank) AS keyword_rank,
                           SUM(1.0 / (%(rrf_k)s + rrf_rank)) AS hybrid_score
                    FROM ranked_candidates
                    GROUP BY id
                )
                SELECT d.source,
                       d.chunk_index,
                       d.content,
                       fused.vector_similarity,
                       fused.semantic_rank,
                       fused.keyword_rank,
                       fused.hybrid_score
                FROM fused
                JOIN documents AS d ON d.id = fused.id
                ORDER BY fused.hybrid_score DESC,
                         fused.vector_similarity DESC NULLS LAST
                LIMIT %(top_k)s;
                """,
                {
                    "query_embedding": to_pgvector_literal(query_embedding),
                    "question": question,
                    "candidate_limit": candidate_limit,
                    "rrf_k": RRF_K,
                    "top_k": top_k,
                },
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    return [
        {
            "source": source,
            "chunk_index": chunk_index,
            "content": content,
            "vector_similarity": float(vector_similarity) if vector_similarity is not None else None,
            "semantic_rank": semantic_rank,
            "keyword_rank": keyword_rank,
            "hybrid_score": float(hybrid_score),
        }
        for (
            source,
            chunk_index,
            content,
            vector_similarity,
            semantic_rank,
            keyword_rank,
            hybrid_score,
        ) in rows
    ]
