"""Load documents (PDF, TXT, MD) and split them into overlapping chunks."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page: int | None = None
    meta: dict = field(default_factory=dict)


def read_file(path: Path) -> list[tuple[str, int | None]]:
    """Return a list of (text, page_number) tuples for a file."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return [(page.extract_text() or "", i + 1) for i, page in enumerate(reader.pages)]
    if suffix in {".txt", ".md", ".markdown"}:
        return [(path.read_text(encoding="utf-8", errors="ignore"), None)]
    raise ValueError(f"Unsupported file type: {suffix}")


def split_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Split on paragraph boundaries, then pack paragraphs into chunks of ~chunk_size chars."""
    text = re.sub(r"\r\n?", "\n", text)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        # Hard-split paragraphs that are bigger than a whole chunk.
        while len(para) > chunk_size:
            head, para = para[:chunk_size], para[chunk_size - overlap :]
            if current:
                chunks.append(current)
                current = ""
            chunks.append(head)
        if len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            tail = current[-overlap:] if current and overlap else ""
            current = f"{tail}\n\n{para}" if tail else para
    if current:
        chunks.append(current)
    return chunks


def load_documents(paths: list[Path], chunk_size: int = 800, overlap: int = 150) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in paths:
        for text, page in read_file(path):
            for i, piece in enumerate(split_text(text, chunk_size, overlap)):
                cid = f"{path.name}:{page or 0}:{i}"
                chunks.append(Chunk(id=cid, text=piece, source=path.name, page=page))
    return chunks
