"""
Production-grade hybrid retrieval for RAG.

Supports:
- BM25 lexical retrieval
- Dense semantic retrieval
- Weighted min-max fusion
- Reciprocal Rank Fusion (RRF)
- Candidate pools
- Minimum score filtering
- Content-aware retrieval adjustment
- Deterministic ranking
- Retrieval diagnostics
"""

from typing import Dict, List, Optional

from .bm25_index import BM25Index
from .embedding_index import EmbeddingIndex


def _minmax_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    """Normalize scores to [0, 1]."""
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

    fusion_method:
        "rrf"      -> Reciprocal Rank Fusion
        "weighted" -> min-max weighted fusion

    alpha:
        Dense weight for weighted fusion.
        BM25 weight = 1 - alpha.

    docstore:
        Optional DocStore used for content-aware retrieval.
    """

    VALID_FUSION_METHODS = {"rrf", "weighted"}

    # Bibliography is useful for explicit reference questions,
    # but normally creates noise for factual/explanatory questions.
    BIBLIOGRAPHY_SCORE_FACTOR = 0.25

    REFERENCE_QUERY_TERMS = {
        "reference",
        "references",
        "bibliography",
        "citation",
        "citations",
        "cited",
        "author",
        "authors",
        "arxiv",
        "doi",
    }

    def __init__(
        self,
        bm25_index: BM25Index,
        embedding_index: EmbeddingIndex,
        alpha: float = 0.5,
        fusion_method: str = "rrf",
        rrf_k: int = 60,
        docstore=None,
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
        self.docstore = docstore

    def _weighted_fusion(
        self,
        bm25_hits: Dict[str, float],
        dense_hits: Dict[str, float],
    ) -> List[Dict]:
        """Fuse BM25 and dense scores using min-max normalization."""
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
                    "score": float(score),
                    "bm25_score": float(
                        bm25_hits.get(chunk_id, 0.0)
                    ),
                    "dense_score": float(
                        dense_hits.get(chunk_id, 0.0)
                    ),
                    "retrieved_by_bm25": chunk_id in bm25_hits,
                    "retrieved_by_dense": chunk_id in dense_hits,
                }
            )

        fused.sort(
            key=lambda item: (
                -item["score"],
                -item["dense_score"],
                -item["bm25_score"],
                item["chunk_id"],
            )
        )

        return fused

    def _rrf_fusion(
        self,
        bm25_results: List,
        dense_results: List,
    ) -> List[Dict]:
        """
        Fuse rankings using Reciprocal Rank Fusion.

        RRF contribution:
            1 / (rrf_k + rank)

        Rank starts at 1.
        """
        rrf_scores: Dict[str, float] = {}

        bm25_hits = dict(bm25_results)
        dense_hits = dict(dense_results)

        for rank, (chunk_id, _) in enumerate(
            bm25_results,
            start=1,
        ):
            rrf_scores[chunk_id] = (
                rrf_scores.get(chunk_id, 0.0)
                + 1.0 / (self.rrf_k + rank)
            )

        for rank, (chunk_id, _) in enumerate(
            dense_results,
            start=1,
        ):
            rrf_scores[chunk_id] = (
                rrf_scores.get(chunk_id, 0.0)
                + 1.0 / (self.rrf_k + rank)
            )

        fused = []

        for chunk_id, score in rrf_scores.items():
            fused.append(
                {
                    "chunk_id": chunk_id,
                    "score": float(score),
                    "bm25_score": float(
                        bm25_hits.get(chunk_id, 0.0)
                    ),
                    "dense_score": float(
                        dense_hits.get(chunk_id, 0.0)
                    ),
                    "retrieved_by_bm25": chunk_id in bm25_hits,
                    "retrieved_by_dense": chunk_id in dense_hits,
                }
            )

        fused.sort(
            key=lambda item: (
                -item["score"],
                -item["dense_score"],
                -item["bm25_score"],
                item["chunk_id"],
            )
        )

        return fused

    @classmethod
    def _is_reference_query(cls, query: str) -> bool:
        """Return True when the query explicitly asks about references."""
        query_terms = set(query.lower().split())
        return bool(query_terms & cls.REFERENCE_QUERY_TERMS)

    def _get_content_type(self, chunk_id: str) -> str:
        """
        Return the content type for a chunk.

        Missing metadata is treated as normal content so that
        existing indexes remain fully compatible.
        """
        if self.docstore is None:
            return "content"

        chunks = getattr(self.docstore, "chunks", None)

        if not isinstance(chunks, dict):
            return "content"

        chunk = chunks.get(chunk_id)

        if chunk is None:
            return "content"

        metadata = getattr(chunk, "metadata", None)

        if not isinstance(metadata, dict):
            return "content"

        return metadata.get("content_type", "content")

    def _apply_content_type_adjustment(
        self,
        query: str,
        results: List[Dict],
    ) -> List[Dict]:
        """
        Reduce bibliography noise for normal questions.

        Bibliography chunks remain fully retrievable when the user
        explicitly asks about references, citations, authors, etc.
        """
        if not results:
            return results

        # Reference-oriented questions should retain bibliography
        # results without penalty.
        if self._is_reference_query(query):
            return results

        adjusted = []

        for result in results:
            updated = dict(result)

            content_type = self._get_content_type(
                result["chunk_id"]
            )

            updated["content_type"] = content_type

            if content_type == "bibliography":
                updated["score"] = float(
                    result["score"]
                    * self.BIBLIOGRAPHY_SCORE_FACTOR
                )

            adjusted.append(updated)

        adjusted.sort(
            key=lambda item: (
                -item["score"],
                -item["dense_score"],
                -item["bm25_score"],
                item["chunk_id"],
            )
        )

        return adjusted

    @staticmethod
    def _validate_parameters(
        query: str,
        top_k: int,
        candidate_pool: int,
        min_score: Optional[float],
    ) -> None:
        """Validate search parameters."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(
                f"top_k must be a positive integer, got {top_k}"
            )

        if not isinstance(candidate_pool, int) or candidate_pool <= 0:
            raise ValueError(
                "candidate_pool must be a positive integer, "
                f"got {candidate_pool}"
            )

        if top_k > candidate_pool:
            raise ValueError(
                f"top_k ({top_k}) cannot be greater than "
                f"candidate_pool ({candidate_pool})"
            )

        if min_score is not None:
            if not isinstance(min_score, (int, float)):
                raise ValueError(
                    "min_score must be a number or None"
                )

            if min_score < 0.0:
                raise ValueError(
                    f"min_score must be >= 0.0, got {min_score}"
                )

    def _fuse(
        self,
        bm25_results: List,
        dense_results: List,
    ) -> List[Dict]:
        """Fuse BM25 and dense rankings using the configured method."""
        if self.fusion_method == "rrf":
            return self._rrf_fusion(
                bm25_results,
                dense_results,
            )

        return self._weighted_fusion(
            dict(bm25_results),
            dict(dense_results),
        )

    def _retrieve_and_fuse(
        self,
        query: str,
        candidate_pool: int,
    ):
        """Retrieve candidates and perform fusion."""
        bm25_results = self.bm25_index.search(
            query,
            candidate_pool,
        )

        dense_results = self.embedding_index.search(
            query,
            candidate_pool,
        )

        fused = self._fuse(
            bm25_results,
            dense_results,
        )

        return bm25_results, dense_results, fused

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_pool: int = 50,
        min_score: Optional[float] = None,
    ) -> List[Dict]:
        """
        Retrieve and fuse BM25 + dense results.
        """
        self._validate_parameters(
            query=query,
            top_k=top_k,
            candidate_pool=candidate_pool,
            min_score=min_score,
        )

        bm25_results, dense_results, fused = (
            self._retrieve_and_fuse(
                query,
                candidate_pool,
            )
        )

        if not bm25_results and not dense_results:
            return []

        # Apply content-aware adjustment BEFORE final top-k.
        # This allows a strong non-bibliography candidate outside
        # the original top-k to move into the final result set.
        fused = self._apply_content_type_adjustment(
            query,
            fused,
        )

        if min_score is not None:
            fused = [
                result
                for result in fused
                if result["score"] >= min_score
            ]

        return fused[:top_k]

    def search_with_diagnostics(
        self,
        query: str,
        top_k: int = 5,
        candidate_pool: int = 50,
        min_score: Optional[float] = None,
    ) -> Dict:
        """
        Execute retrieval and return results plus diagnostics.
        """
        self._validate_parameters(
            query=query,
            top_k=top_k,
            candidate_pool=candidate_pool,
            min_score=min_score,
        )

        bm25_results, dense_results, fused = (
            self._retrieve_and_fuse(
                query,
                candidate_pool,
            )
        )

        pre_adjustment_count = len(fused)

        fused = self._apply_content_type_adjustment(
            query,
            fused,
        )

        if min_score is not None:
            fused = [
                result
                for result in fused
                if result["score"] >= min_score
            ]

        final_results = fused[:top_k]

        bibliography_candidates = sum(
            1
            for result in final_results
            if result.get("content_type") == "bibliography"
        )

        return {
            "query": query,
            "fusion_method": self.fusion_method,
            "alpha": self.alpha,
            "rrf_k": self.rrf_k,
            "candidate_pool": candidate_pool,
            "requested_top_k": top_k,
            "min_score": min_score,
            "bm25_candidates": len(bm25_results),
            "dense_candidates": len(dense_results),
            "fused_candidates": pre_adjustment_count,
            "results_after_threshold": len(fused),
            "final_results": len(final_results),
            "bibliography_results": bibliography_candidates,
            "reference_query": self._is_reference_query(query),
            "results": final_results,
        }