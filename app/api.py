from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.ingest import embed_query
from app.retriever import search
from app.llm import generate_answer

app = FastAPI()


class ChatRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be empty.")

    query_embedding = embed_query(question)
    results = search(query_embedding, question, top_k=5)
    context = "\n\n".join(
        f"[S{index}] Source: {result['source']} (chunk {result['chunk_index']})\n{result['content']}"
        for index, result in enumerate(results, start=1)
    )
    if not context:
        answer = "I don't know based on the provided documents."
    else:
        try:
            answer = generate_answer(context, question)
        except Exception as exc:
            raise HTTPException(status_code=502, detail="Answer generation failed.") from exc

    return {
        "question": question,
        "answer": answer,
        "sources": [
            {
                "id": f"S{index}",
                "source": result["source"],
                "chunk_index": result["chunk_index"],
                "hybrid_score": round(result["hybrid_score"], 5),
                "vector_similarity": (
                    round(result["vector_similarity"], 3)
                    if result["vector_similarity"] is not None
                    else None
                ),
                "semantic_rank": result["semantic_rank"],
                "keyword_rank": result["keyword_rank"],
                "content": result["content"],
            }
            for index, result in enumerate(results, start=1)
        ],
    }
