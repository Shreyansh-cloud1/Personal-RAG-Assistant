from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.ingest import embed_text
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

    query_embedding = embed_text(question)
    results = search(query_embedding, top_k=5)
    context = "\n\n".join([row[0] for row in results])
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
        "sources": [row[0][:200] for row in results],
    }
