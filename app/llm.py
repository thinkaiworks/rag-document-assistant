"""Answer generation. Uses Claude or OpenAI when a key is set, otherwise an extractive fallback.

The prompt forces the model to answer only from the retrieved sources and to cite them,
and to say "I don't know" instead of guessing.
"""
from __future__ import annotations

import os
import re

from .ingest import Chunk
from .retriever import tokenize

SYSTEM_PROMPT = """You are a support assistant that answers ONLY from the company documents provided.
Rules:
- Use only facts found in the SOURCES. Never use outside knowledge.
- Cite every fact with its source tag, e.g. [1] or [2].
- If the sources do not contain the answer, reply exactly: "I couldn't find that in our documents."
- Be concise: 2-5 sentences unless the user asks for detail."""

NOT_FOUND = "I couldn't find that in our documents."


def build_context(hits: list[tuple[Chunk, float]]) -> str:
    parts = []
    for n, (chunk, _) in enumerate(hits, start=1):
        loc = f"{chunk.source}" + (f", page {chunk.page}" if chunk.page else "")
        parts.append(f"[{n}] ({loc})\n{chunk.text}")
    return "\n\n".join(parts)


def provider() -> str:
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "extractive"


def generate_answer(question: str, hits: list[tuple[Chunk, float]]) -> str:
    if not hits:
        return NOT_FOUND
    context = build_context(hits)
    user = f"SOURCES:\n{context}\n\nQUESTION: {question}"
    p = provider()
    if p == "anthropic":
        import anthropic

        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
            max_tokens=600,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
    if p == "openai":
        from openai import OpenAI

        client = OpenAI()
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content.strip()
    return extractive_answer(question, hits)


def extractive_answer(question: str, hits: list[tuple[Chunk, float]]) -> str:
    """No-API fallback: return the best-matching sentences from the top sources, with citations."""
    q = set(tokenize(question))
    scored: list[tuple[int, int, str]] = []
    for n, (chunk, _) in enumerate(hits, start=1):
        for sent in re.split(r"(?<=[.!?])\s+|\n+", chunk.text):
            sent = sent.strip(" -#*")
            if len(sent) < 20:
                continue
            overlap = len(q & set(tokenize(sent)))
            if overlap:
                scored.append((overlap, n, sent))
    if not scored:
        return NOT_FOUND
    scored.sort(key=lambda x: (-x[0], x[1]))
    picked = scored[:2]
    return " ".join(f"{s} [{n}]" for _, n, s in picked)
