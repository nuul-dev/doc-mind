import os
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from qdrant_client import QdrantClient

from app.eval_agent import evaluate
from app.generation import generate_answer
from app.ingest import ingest_pdf
from app.retrieval import retrieve
from app.types import Source

app = FastAPI(title="DocMind", version="0.1.0")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
_qdrant: QdrantClient | None = None


def get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(url=QDRANT_URL)
    return _qdrant


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    evaluate: bool = False


@app.post("/ingest")
async def ingest_endpoint(file: UploadFile = File(...)):
    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    result = ingest_pdf(get_qdrant(), file.filename, data)
    return JSONResponse(content={"status": "ok", "filename": result["filename"], "chunks": result["chunks"]})


@app.post("/query")
async def query_endpoint(req: QueryRequest) -> dict[str, Any]:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    chunks = retrieve(get_qdrant(), req.question, top_k=req.top_k)
    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant documents found")

    answer = generate_answer(req.question, chunks)

    sources: list[Source] = [
        Source(filename=c["filename"], chunk_index=c["chunk_index"], score=c["score"])
        for c in chunks
    ]

    if req.evaluate:
        return {
            "answer": answer,
            "sources": sources,
            "eval": evaluate(req.question, chunks, answer),
        }

    return {"answer": answer, "sources": sources}


@app.get("/health")
async def health():
    return {"status": "ok"}
