"""
Docstore: a plain chunk_id -> Chunk lookup, persisted alongside the indexes.
"""

import pickle
from typing import Dict, List

from .chunking import Chunk


class DocStore:
    def __init__(self):
        self.chunks: Dict[str, Chunk] = {}

    def add(self, chunks: List[Chunk]):
        for c in chunks:
            self.chunks[c.id] = c

    def get(self, chunk_id: str) -> Chunk:
        return self.chunks.get(chunk_id)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.chunks, f)

    @classmethod
    def load(cls, path: str) -> "DocStore":
        obj = cls()
        with open(path, "rb") as f:
            obj.chunks = pickle.load(f)
        return obj
