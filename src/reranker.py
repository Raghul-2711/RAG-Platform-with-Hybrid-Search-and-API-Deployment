"""
Query-aware reranking for Enterprise RAG.

This module performs a lightweight deterministic reranking pass after
hybrid RRF retrieval.

It does not require an additional ML model or external service.
"""

import re
from typing import Callable, Dict, List


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def _tokenize(text: str) -> List[str]:
    """
    Tokenize text into lowercase alphanumeric terms.
    """

    if not text:
        return []

    return re.findall(
        r"[a-z0-9]+",
        str(text).lower(),
    )


def _query_terms(query: str) -> List[str]:
    """
    Return meaningful query terms with common stopwords removed.
    """

    return [
        token
        for token in _tokenize(query)
        if token not in _STOPWORDS
        and len(token) > 1
    ]


def _normalize_overlap(
    query_terms: List[str],
    text: str,
) -> float:
    """
    Calculate normalized query-term overlap.

    Returns a value between 0.0 and 1.0.
    """

    unique_terms = set(query_terms)

    if not unique_terms:
        return 0.0

    text_terms = set(_tokenize(text))

    matched = sum(
        1
        for term in unique_terms
        if term in text_terms
    )

    return matched / len(unique_terms)


def _phrase_overlap(
    query: str,
    text: str,
) -> float:
    """
    Measure meaningful consecutive query phrase overlap.

    Uses bigrams and trigrams after removing stopwords.

    Returns a value between 0.0 and 1.0.
    """

    query_terms = _query_terms(query)

    if len(query_terms) < 2:
        return 0.0

    text_terms = _tokenize(text)

    if not text_terms:
        return 0.0

    text_normalized = " ".join(text_terms)

    phrases = []

    for size in (3, 2):
        if len(query_terms) < size:
            continue

        for index in range(
            len(query_terms) - size + 1
        ):
            phrase = " ".join(
                query_terms[
                    index:index + size
                ]
            )

            phrases.append(phrase)

    if not phrases:
        return 0.0

    matched = sum(
        1
        for phrase in phrases
        if phrase in text_normalized
    )

    return matched / len(phrases)


def _safe_float(
    value,
    default: float = 0.0,
) -> float:
    """
    Safely convert a value to float.
    """

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bounded_score(
    value: float,
) -> float:
    """
    Clamp a numeric score to the range [0.0, 1.0].
    """

    return max(
        0.0,
        min(
            value,
            1.0,
        ),
    )


def _score_result(
    result: Dict,
    query: str,
    text: str,
) -> Dict:
    """
    Calculate deterministic reranking features.
    """

    terms = _query_terms(query)

    overlap = _normalize_overlap(
        terms,
        text,
    )

    phrase = _phrase_overlap(
        query,
        text,
    )

    original_score = _safe_float(
        result.get("score", 0.0)
    )

    bm25_score = _safe_float(
        result.get("bm25_score", 0.0)
    )

    dense_score = _safe_float(
        result.get("dense_score", 0.0)
    )

    # RRF scores are normally small values around 0.01-0.04.
    # Convert the RRF signal into a bounded 0-1 range.
    rrf_component = _bounded_score(
        original_score * 25.0
    )

    # BM25 values are document/query dependent.
    # Convert them into a bounded saturation score instead of
    # allowing unusually large BM25 values to dominate.
    bm25_component = (
        bm25_score
        / (bm25_score + 10.0)
        if bm25_score > 0.0
        else 0.0
    )

    bm25_component = _bounded_score(
        bm25_component
    )

    # Dense cosine similarity is expected to be approximately
    # within the [-1, 1] range. For this retrieval pipeline,
    # positive similarity is the useful signal.
    dense_component = _bounded_score(
        dense_score
    )

    # Deterministic lightweight reranking:
    #
    # RRF       -> preserves hybrid retrieval ranking
    # BM25      -> lexical retrieval signal
    # Dense     -> semantic retrieval signal
    # Overlap   -> direct query-term relevance
    # Phrase    -> direct multi-word phrase relevance
    #
    # Query overlap is deliberately strong enough to move generic
    # semantic matches below passages containing the actual concepts
    # requested by the user.
    rerank_score = (
        0.30 * rrf_component
        + 0.15 * bm25_component
        + 0.15 * dense_component
        + 0.30 * overlap
        + 0.10 * phrase
    )

    return {
        **result,
        "rerank_score": float(
            rerank_score
        ),
        "query_term_overlap": round(
            overlap,
            6,
        ),
        "phrase_overlap": round(
            phrase,
            6,
        ),
    }


def rerank(
    query: str,
    results: List[Dict],
    text_lookup: Callable[[str], str],
) -> List[Dict]:
    """
    Rerank retrieved candidates.

    Parameters
    ----------
    query:
        User query.

    results:
        RRF/content-filtered retrieval candidates.

    text_lookup:
        Callable receiving chunk_id and returning chunk text.

    Returns
    -------
    List[Dict]
        Reranked results with deterministic reranking metadata.
    """

    if not results:
        return []

    reranked: List[Dict] = []

    for result in results:
        chunk_id = result.get("chunk_id")

        if not chunk_id:
            continue

        try:
            text = text_lookup(chunk_id)
        except Exception:
            text = ""

        if text is None:
            text = ""

        reranked.append(
            _score_result(
                result,
                query,
                str(text),
            )
        )

    reranked.sort(
        key=lambda item: (
            -item["rerank_score"],
            -_safe_float(item.get("score", 0.0)),
            -_safe_float(item.get("dense_score", 0.0)),
            -_safe_float(item.get("bm25_score", 0.0)),
            item.get("chunk_id", ""),
        )
    )

    return reranked