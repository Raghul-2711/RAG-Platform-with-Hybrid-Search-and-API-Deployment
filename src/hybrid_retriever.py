"""
Hybrid retrieval for RAG.

Supports two fusion strategies:

1. Weighted min-max fusion
   final = alpha * dense_score + (1 - alpha) * bm25_score

2. Reciprocal Rank Fusion (RRF)
   final = sum(1 / (rrf_k + rank))

RRF is the default because it combines retrieval rankings without
depending on the raw score scales of BM25 and dense retrieval.
"""

from typing import Dict, List

from .bm25_index import BM25Index
from .embedding_index import EmbeddingIndex


def _minmax_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize scores to the range [0, 1].

    If all scores are identical, every item receives 1.0.
    """
    if not scores:
        return {}

    values = list(scores.values())
    lo = min(values)
    hi = max(values)

    if hi - lo < 1e-12:
        return {key: 1.0 for key in scores}

    return {
        key: (value - lo) / (hi - lo)
        for key, value in scores.items()
    }


class HybridRetriever:
    """
    Combines BM25 and dense retrieval.

    Parameters
    ----------
    bm25_index:
        BM25 retrieval index.

    embedding_index:
        Dense embedding retrieval index.

    alpha:
        Weight assigned to dense retrieval when using weighted fusion.
        BM25 receives (1 - alpha).

    fusion_method:
        "rrf" for Reciprocal Rank Fusion.
        "weighted" for the existing min-max weighted fusion.

    rrf_k:
        RRF smoothing constant. The standard value of 60 is used by
        default.

    Example
    -------
    RRF:

        retriever = HybridRetriever(
            bm25_index=bm25,
            embedding_index=embedding_index,
            fusion_method="rrf",
            rrf_k=60,
        )

    Weighted:

        retriever = HybridRetriever(
            bm25_index=bm25,
            embedding_index=embedding_index,
            alpha=0.5,
            fusion_method="weighted",
        )
    """

    VALID_FUSION_METHODS = {"rrf", "weighted"}

    def __init__(
        self,
        bm25_index: BM25Index,
        embedding_index: EmbeddingIndex,
        alpha: float = 0.5,
        fusion_method: str = "rrf",
        rrf_k: int = 60,
    ):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(
                f"alpha must be between 0.0 and 1.0, got {alpha}"
            )

        if fusion_method not in self.VALID_FUSION_METHODS:
            raise ValueError(
                f"fusion_method must be one of "
                f"{sorted(self.VALID_FUSION_METHODS)}, "
                f"got '{fusion_method}'"
            )

        if not isinstance(rrf_k, int) or rrf_k <= 0:
            raise ValueError(
                f"rrf_k must be a positive integer, got {rrf_k}"
            )

        self.bm25_index = bm25_index
        self.embedding_index = embedding_index
        self.alpha = alpha
        self.fusion_method = fusion_method
        self.rrf_k = rrf_k

    def _weighted_fusion(
        self,
        bm25_hits: Dict[str, float],
        dense_hits: Dict[str, float],
    ) -> List[Dict]:
        """
        Fuse BM25 and dense scores using min-max normalization.
        """
        bm25_norm = _minmax_normalize(bm25_hits)
        dense_norm = _minmax_normalize(dense_hits)

        all_ids = set(bm25_norm) | set(dense_norm)

        fused = []

        for chunk_id in all_ids:
            bm25_normalized = bm25_norm.get(chunk_id, 0.0)
            dense_normalized = dense_norm.get(chunk_id, 0.0)

            score = (
                self.alpha * dense_normalized
                + (1.0 - self.alpha) * bm25_normalized
            )

            fused.append(
                {
                    "chunk_id": chunk_id,
                    "score": score,
                    "bm25_score": bm25_hits.get(chunk_id, 0.0),
                    "dense_score": dense_hits.get(chunk_id, 0.0),
                }
            )

        fused.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return fused

    def _rrf_fusion(
        self,
        bm25_hits: Dict[str, float],
        dense_hits: Dict[str, float],
    ) -> List[Dict]:
        """
        Fuse BM25 and dense rankings using Reciprocal Rank Fusion.

        RRF score:

            1 / (rrf_k + rank)

        Rank starts at 1.
        """
        rrf_scores: Dict[str, float] = {}

        # BM25 ranking
        bm25_ranked = sorted(
            bm25_hits.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        for rank, (chunk_id, _) in enumerate(bm25_ranked, start=1):
            rrf_scores[chunk_id] = (
                rrf_scores.get(chunk_id, 0.0)
                + 1.0 / (self.rrf_k + rank)
            )

        # Dense ranking
        dense_ranked = sorted(
            dense_hits.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        for rank, (chunk_id, _) in enumerate(dense_ranked, start=1):
            rrf_scores[chunk_id] = (
                rrf_scores.get(chunk_id, 0.0)
                + 1.0 / (self.rrf_k + rank)
            )

        fused = []

        for chunk_id, score in rrf_scores.items():
            fused.append(
                {
                    "chunk_id": chunk_id,
                    "score": score,
                    "bm25_score": bm25_hits.get(chunk_id, 0.0),
                    "dense_score": dense_hits.get(chunk_id, 0.0),
                }
            )

        fused.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return fused

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_pool: int = 50,
    ) -> List[Dict]:
        """
        Retrieve and fuse BM25 + dense results.

        Parameters
        ----------
        query:
            User's search query.

        top_k:
            Number of final results to return.

        candidate_pool:
            Number of candidates retrieved independently from BM25
            and dense retrieval before fusion.

        Returns
        -------
        List[Dict]
            Each result contains:

            {
                "chunk_id": str,
                "score": float,
                "bm25_score": float,
                "dense_score": float
            }
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(
                f"top_k must be a positive integer, got {top_k}"
            )

        if not isinstance(candidate_pool, int) or candidate_pool <= 0:
            raise ValueError(
                f"candidate_pool must be a positive integer, "
                f"got {candidate_pool}"
            )

        if top_k > candidate_pool:
            raise ValueError(
                f"top_k ({top_k}) cannot be greater than "
                f"candidate_pool ({candidate_pool})"
            )

        bm25_results = self.bm25_index.search(
            query,
            candidate_pool,
        )

        dense_results = self.embedding_index.search(
            query,
            candidate_pool,
        )

        bm25_hits = dict(bm25_results)
        dense_hits = dict(dense_results)

        if self.fusion_method == "rrf":
            fused = self._rrf_fusion(
                bm25_hits,
                dense_hits,
            )
        else:
            fused = self._weighted_fusion(
                bm25_hits,
                dense_hits,
            )

        return fused[:top_k]