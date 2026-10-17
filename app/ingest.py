from sentence_transformers import SentenceTransformer

# model = SentenceTransformer("BAAI/bge-base-en-v1.5")
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def embed_text(text: str):
    return model.encode(text).tolist()


def embed_query(question: str):
    """Embed a search query using BGE's recommended retrieval instruction."""
    return model.encode(QUERY_INSTRUCTION + question).tolist()


from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(text, chunk_size=400, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)
    return splitter.split_text(text)


import fitz

def extract_pdf_text(path):
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)


from sqlalchemy import text
from app.db import engine
from app.vector_utils import to_pgvector_literal

def insert_chunk(source, chunk_index, content, embedding):
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO documents (
                    source,
                    chunk_index,
                    content,
                    embedding
                )
                VALUES (
                    :source,
                    :chunk_index,
                    :content,
                    CAST(:embedding AS vector)
                )
            """),
            {
                "source": source,
                "chunk_index": chunk_index,
                "content": content,
                "embedding": to_pgvector_literal(embedding),
            },
        )
