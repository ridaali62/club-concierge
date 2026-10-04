"""FastAPI app: POST /ask, GET /health and a simple chat page at /."""
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

import rag

load_dotenv()
app = FastAPI(title="Club Concierge")
index = rag.Index(rag.load_chunks(), rag.sentence_transformer_embedder())
llm = rag.claude_llm()


class Question(BaseModel):
    question: str


@app.post("/ask")
def ask(q: Question):
    return rag.answer(q.question, index, llm)


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(index.chunks)}


@app.get("/")
def chat_page():
    return FileResponse(Path(__file__).parent / "static" / "index.html")
