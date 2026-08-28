"""
Hybrid retrieval: merges BM25 (sparse/keyword) and dense (embedding) results
using min-max normalized weighted score fusion.
"""

from typing import Dict, List

from .bm25_index import BM25Index
from .embedding_index import EmbeddingIndex


def _minmax_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


class HybridRetriever:
    def __init__(self, bm25_index: BM25Index, embedding_index: EmbeddingIndex, alpha: float = 0.5):
        """
        alpha: weight given to the dense/embedding score.
               final = alpha * dense_norm + (1 - alpha) * bm25_norm
        """
        self.bm25_index = bm25_index
        self.embedding_index = embedding_index
        self.alpha = alpha

    def search(self, query: str, top_k: int = 5, candidate_pool: int = 25) -> List[Dict]:
        bm25_hits = dict(self.bm25_index.search(query, candidate_pool))
        dense_hits = dict(self.embedding_index.search(query, candidate_pool))

        bm25_norm = _minmax_normalize(bm25_hits)
        dense_norm = _minmax_normalize(dense_hits)

        all_ids = set(bm25_norm) | set(dense_norm)
        fused = []
        for cid in all_ids:
            b = bm25_norm.get(cid, 0.0)
            d = dense_norm.get(cid, 0.0)
            score = self.alpha * d + (1 - self.alpha) * b
            fused.append({
                "chunk_id": cid,
                "score": score,
                "bm25_score": bm25_hits.get(cid, 0.0),
                "dense_score": dense_hits.get(cid, 0.0),
            })

        fused.sort(key=lambda x: x["score"], reverse=True)
        return fused[:top_k]
