from pathlib import Path

import pytest

from app.ingest import load_documents, split_text
from app.llm import NOT_FOUND, extractive_answer
from app.retriever import Retriever

DOCS = Path(__file__).resolve().parent.parent / "sample_docs"


def test_split_respects_chunk_size():
    text = "\n\n".join(["word " * 60] * 10)
    chunks = split_text(text, chunk_size=400, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 460 for c in chunks)


def test_retrieves_right_document():
    chunks = load_documents(sorted(DOCS.glob("*.md")))
    r = Retriever(chunks, use_embeddings=False)
    top, _ = r.search("How long does express shipping take?", k=1)[0]
    assert top.source == "shipping-faq.md"


def test_extractive_answer_cites_source():
    chunks = load_documents(sorted(DOCS.glob("*.md")))
    hits = Retriever(chunks, use_embeddings=False).search("refund within how many days", k=2)
    answer = extractive_answer("refund within how many days", hits)
    assert "30 days" in answer and "[1]" in answer


def test_unknown_question_says_not_found():
    chunks = load_documents(sorted(DOCS.glob("*.md")))
    hits = Retriever(chunks, use_embeddings=False).search("zebra migration patterns", k=2)
    assert extractive_answer("zebra migration patterns", hits) == NOT_FOUND


def test_api_ask(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        resp = client.post("/ask", json={"question": "Do you ship to Canada?"})
        body = resp.json()
        assert resp.status_code == 200
        assert body["found"] is True
        assert body["sources"][0]["source"] == "shipping-faq.md"
