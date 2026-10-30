"""FastAPI service: upload documents, ask questions, get cited answers."""
from __future__ import annotations

import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .ingest import load_documents
from .llm import NOT_FOUND, generate_answer, provider
from .retriever import Retriever

DOCS_DIR = Path(os.getenv("DOCS_DIR", "sample_docs"))
ALLOWED = {".pdf", ".txt", ".md", ".markdown"}

_state: dict = {}


def rebuild_index() -> int:
    files = sorted(p for p in DOCS_DIR.glob("*") if p.suffix.lower() in ALLOWED)
    chunks = load_documents(files)
    _state["retriever"] = Retriever(chunks)
    _state["files"] = [f.name for f in files]
    return len(chunks)


@asynccontextmanager
async def lifespan(_: FastAPI):
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    rebuild_index()
    yield


app = FastAPI(title="RAG Document Assistant", version="1.0.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str
    top_k: int = 4


class Source(BaseModel):
    ref: int
    source: str
    page: int | None
    snippet: str


class AskResponse(BaseModel):
    answer: str
    found: bool
    sources: list[Source]
    provider: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "documents": _state.get("files", []), "provider": provider()}


@app.post("/documents")
async def upload(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(400, f"Only {sorted(ALLOWED)} files are supported")
    dest = DOCS_DIR / Path(file.filename).name
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    n = rebuild_index()
    return {"uploaded": dest.name, "chunks_indexed": n}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    if not req.question.strip():
        raise HTTPException(400, "Question is empty")
    hits = _state["retriever"].search(req.question, k=req.top_k)
    answer = generate_answer(req.question, hits)
    sources = [
        Source(ref=i, source=c.source, page=c.page, snippet=c.text[:240])
        for i, (c, _) in enumerate(hits, start=1)
    ]
    return AskResponse(answer=answer, found=answer != NOT_FOUND, sources=sources, provider=provider())


@app.get("/", response_class=HTMLResponse)
def ui() -> str:
    return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")
