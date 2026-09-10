"""
Document loading + chunking.

Supports TXT, Markdown, PDF, DOCX, and PPTX documents through
the dedicated loader layer in src/loaders/.
"""

from dataclasses import dataclass, field
from typing import List
import hashlib
import os

from .loaders import (
    TextLoader,
    PDFLoader,
    DocxLoader,
    PptxLoader,
)


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


def _create_stable_chunk_id(
    source: str,
    chunk_index: int,
    metadata: dict,
) -> str:
    """
    Create a deterministic chunk ID.

    The same document/page/chunk position produces the same ID
    across repeated index rebuilds.
    """
    page_number = metadata.get("page_number", "")
    slide_number = metadata.get("slide_number", "")

    identity = (
        f"{source}|"
        f"page={page_number}|"
        f"slide={slide_number}|"
        f"chunk={chunk_index}"
    )

    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def chunk_text(
    text: str,
    source: str,
    chunk_size: int,
    overlap: int,
    metadata: dict = None,
) -> List[Chunk]:
    """
    Split text into overlapping word-based chunks.

    Chunk IDs are deterministic so evaluation and retrieval remain
    reproducible across index rebuilds.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    if overlap < 0:
        raise ValueError("overlap must be >= 0")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0
    idx = 0
    step = chunk_size - overlap

    base_metadata = dict(metadata or {})

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_str = " ".join(chunk_words)

        chunk_metadata = dict(base_metadata)
        chunk_metadata["word_count"] = len(chunk_words)
        chunk_metadata["chunk_index"] = idx

        chunk_id = _create_stable_chunk_id(
            source=source,
            chunk_index=idx,
            metadata=chunk_metadata,
        )

        chunks.append(
            Chunk(
                id=chunk_id,
                text=chunk_str,
                source=source,
                chunk_index=idx,
                metadata=chunk_metadata,
            )
        )

        idx += 1

        if end == len(words):
            break

        start += step

    return chunks


# Reuse loader instances instead of creating them for every file.
_DOCUMENT_LOADERS = [
    TextLoader(),
    PDFLoader(),
    DocxLoader(),
    PptxLoader(),
]


def get_document_loaders():
    """Return all supported document loaders."""
    return _DOCUMENT_LOADERS


def get_loader(file_path: str):
    """Return the appropriate loader for a file."""
    for loader in get_document_loaders():
        if loader.can_load(file_path):
            return loader

    return None


def load_documents(docs_dir: str):
    """
    Load all supported documents from a directory.

    Returns a list of LoadedDocument objects.
    """
    documents = []

    if not os.path.isdir(docs_dir):
        return documents

    for fname in sorted(os.listdir(docs_dir)):
        path = os.path.join(docs_dir, fname)

        if not os.path.isfile(path):
            continue

        loader = get_loader(path)

        if loader is None:
            continue

        documents.extend(loader.load(path))

    return documents


def build_chunks_from_dir(
    docs_dir: str,
    chunk_size: int,
    overlap: int,
) -> List[Chunk]:
    """
    Load supported documents and convert them into chunks.

    Loader metadata such as PDF page numbers and PPTX slide numbers
    is preserved in every generated chunk.
    """
    all_chunks = []

    for document in load_documents(docs_dir):
        chunks = chunk_text(
            document.text,
            document.source,
            chunk_size,
            overlap,
            metadata=document.metadata,
        )

        total_chunks = len(chunks)

        for chunk in chunks:
            chunk.metadata.update(
                {
                    "document_name": document.source,
                    "total_chunks": total_chunks,
                }
            )

        all_chunks.extend(chunks)

    return all_chunks