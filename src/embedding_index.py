"""
Dense embedding index using sentence-transformers + cosine similarity (numpy).

No external vector DB is required — cosine similarity over a numpy matrix is
plenty fast for tens of thousands of chunks and keeps the project dependency
list short. Swap in FAISS/Chroma/Pinecone later by replacing this module's
internals; the public interface (build/search/save/load) stays the same.
"""

import pickle
from typing import List, Tuple, Callable, Optional

import numpy as np

_model_cache = {}


def get_embedder(model_name: str):
    """
    Lazily loads and caches a sentence-transformers model.
    Imported inside the function so the rest of the codebase (and its tests)
    don't require the heavy sentence-transformers/torch dependency to import.
    """
    if model_name in _model_cache:
        return _model_cache[model_name]
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    _model_cache[model_name] = model
    return model


def default_encode_fn(model_name: str) -> Callable[[List[str]], np.ndarray]:
    model = get_embedder(model_name)

    def _encode(texts: List[str]) -> np.ndarray:
        vecs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return _normalize(np.asarray(vecs, dtype=np.float32))

    return _encode


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return matrix / norms


class EmbeddingIndex:
    """
    encode_fn: Callable[[List[str]], np.ndarray] — injected so tests (or
    alternative embedding providers) can avoid loading a real transformer model.
    """

    def __init__(self, encode_fn: Optional[Callable[[List[str]], np.ndarray]] = None):
        self.encode_fn = encode_fn
        self.chunk_ids: List[str] = []
        self.matrix: np.ndarray = None

    def build(self, chunk_ids: List[str], texts: List[str]):
        if len(chunk_ids) != len(texts):
            raise ValueError("chunk_ids and texts must be the same length")
        if self.encode_fn is None:
            raise ValueError("encode_fn was not provided to EmbeddingIndex")
        self.matrix = self.encode_fn(texts)
        self.chunk_ids = list(chunk_ids)

    def search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        if self.matrix is None or len(self.chunk_ids) == 0:
            return []
        q_vec = self.encode_fn([query])[0]
        sims = self.matrix @ q_vec  # cosine similarity, since rows are normalized
        ranked_idx = np.argsort(-sims)[:top_k]
        return [(self.chunk_ids[i], float(sims[i])) for i in ranked_idx]

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"chunk_ids": self.chunk_ids, "matrix": self.matrix}, f)

    @classmethod
    def load(cls, path: str, encode_fn=None) -> "EmbeddingIndex":
        obj = cls(encode_fn=encode_fn)
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj.chunk_ids = data["chunk_ids"]
        obj.matrix = data["matrix"]
        return obj
