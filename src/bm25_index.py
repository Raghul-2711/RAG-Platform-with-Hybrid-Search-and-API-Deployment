"""
BM25 sparse keyword index, built on top of rank_bm25.
"""

import pickle
import re
from typing import List, Tuple

from rank_bm25 import BM25Okapi


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Index:
    def __init__(self):
        self.bm25: BM25Okapi = None
        self.chunk_ids: List[str] = []

    def build(self, chunk_ids: List[str], texts: List[str]):
        if len(chunk_ids) != len(texts):
            raise ValueError("chunk_ids and texts must be the same length")
        tokenized = [tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(tokenized)
        self.chunk_ids = list(chunk_ids)

    def search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self.chunk_ids, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunk_ids": self.chunk_ids}, f)

    @classmethod
    def load(cls, path: str) -> "BM25Index":
        obj = cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj.bm25 = data["bm25"]
        obj.chunk_ids = data["chunk_ids"]
        return obj
