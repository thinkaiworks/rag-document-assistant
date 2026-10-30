"""Hybrid retriever: BM25 keyword search, plus OpenAI embeddings when a key is set."""
from __future__ import annotations

import math
import os
import re
from collections import Counter

from .ingest import Chunk

STOPWORDS = set(
    "a an and are as at be by for from has have how i in is it its of on or our that the this to was we what when where which who why will with you your do does can".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.avgdl = sum(len(d) for d in docs) / max(len(docs), 1)
        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in docs]

    def score(self, query: list[str], idx: int) -> float:
        tf, dl = self.tf[idx], len(self.docs[idx])
        s = 0.0
        for q in query:
            if q not in tf:
                continue
            f = tf[q]
            s += self.idf.get(q, 0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class Retriever:
    """Keyword (BM25) retrieval always on; vector retrieval added when OPENAI_API_KEY is present.

    Scores from both are normalised and blended (reciprocal-rank fusion) so you get the
    precision of keyword search on names/codes and the recall of semantic search.
    """

    def __init__(self, chunks: list[Chunk], use_embeddings: bool | None = None):
        self.chunks = chunks
        self.bm25 = BM25([tokenize(c.text) for c in chunks])
        if use_embeddings is None:
            use_embeddings = bool(os.getenv("OPENAI_API_KEY"))
        self.vectors: list[list[float]] | None = None
        if use_embeddings and chunks:
            self.vectors = self._embed([c.text for c in chunks])

    @staticmethod
    def _embed(texts: list[str]) -> list[list[float]]:
        from openai import OpenAI

        client = OpenAI()
        model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        out: list[list[float]] = []
        for i in range(0, len(texts), 96):
            resp = client.embeddings.create(model=model, input=texts[i : i + 96])
            out.extend(d.embedding for d in resp.data)
        return out

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        q = tokenize(query)
        kw = sorted(range(len(self.chunks)), key=lambda i: self.bm25.score(q, i), reverse=True)
        kw_scores = {i: self.bm25.score(q, i) for i in kw}
        fused: dict[int, float] = {}
        for rank, i in enumerate(kw):
            if kw_scores[i] > 0:
                fused[i] = fused.get(i, 0) + 1 / (60 + rank)
        if self.vectors is not None:
            qv = self._embed([query])[0]
            sem = sorted(range(len(self.chunks)), key=lambda i: _cosine(qv, self.vectors[i]), reverse=True)
            for rank, i in enumerate(sem[: k * 5]):
                fused[i] = fused.get(i, 0) + 1 / (60 + rank)
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]
        return [(self.chunks[i], s) for i, s in ranked]
