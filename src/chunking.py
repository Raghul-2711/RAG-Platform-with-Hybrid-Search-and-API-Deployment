"""
Document ingestion + chunking.

Splits raw text into overlapping word-based chunks. Word-based chunking is
simple, dependency-free, and works well enough for BM25 + embedding hybrid
retrieval. Swap this out for a token-based splitter later if you need exact
LLM context-window control.
"""

from dataclasses import dataclass, field
from typing import List
import os
import uuid


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


def chunk_text(text: str, source: str, chunk_size: int, overlap: int) -> List[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    idx = 0
    step = chunk_size - overlap

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_str = " ".join(chunk_words)
        chunks.append(
            Chunk(
                id=str(uuid.uuid4()),
                text=chunk_str,
                source=source,
                chunk_index=idx,
                metadata={"word_count": len(chunk_words)},
            )
        )
        idx += 1
        if end == len(words):
            break
        start += step

    return chunks


def load_documents(docs_dir: str) -> List[tuple]:
    """Load all .txt/.md files from a directory. Returns list of (filename, text)."""
    docs = []
    if not os.path.isdir(docs_dir):
        return docs
    for fname in sorted(os.listdir(docs_dir)):
        if fname.lower().endswith((".txt", ".md")):
            path = os.path.join(docs_dir, fname)
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                docs.append((fname, f.read()))
    return docs


def build_chunks_from_dir(docs_dir: str, chunk_size: int, overlap: int) -> List[Chunk]:
    all_chunks = []
    for fname, text in load_documents(docs_dir):
        all_chunks.extend(chunk_text(text, fname, chunk_size, overlap))
    return all_chunks
