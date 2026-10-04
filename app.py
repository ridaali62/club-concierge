"""FastAPI app: POST /ask, GET /health and a simple chat page at /."""
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

import rag

load_dotenv()
app = FastAPI(title="Club Concierge")
index = rag.Index(rag.load_chunks(), rag.sentence_transformer_embedder())
llm = rag.gemini_llm()


class Question(BaseModel):
    question: str
    member_id: int | None = None  # demo field; a real app would take this from the signed-in session


@app.post("/ask")
def ask(q: Question):
    tools = rag.tee_time_tools(q.member_id)  # visitors: look up only; members: look up and book
    return rag.answer(q.question, index, llm, member=q.member_id is not None, tools=tools,
                      today=date.today().isoformat())


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(index.chunks)}


@app.get("/")
def chat_page():
    return FileResponse(Path(__file__).parent / "static" / "index.html")
