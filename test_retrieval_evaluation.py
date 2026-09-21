"""
Retrieval Evaluation for the RAG System

Evaluates:
1. BM25 retrieval
2. Dense embedding retrieval
3. Weighted Hybrid retrieval
4. RRF Hybrid retrieval

Metrics:
- Hit@K
- Recall@K
- Precision@K
- MRR
"""

import os
import time
from typing import Dict, List, Set

from src.loaders import PDFLoader
from src.chunking import chunk_text
from src.bm25_index import BM25Index
from src.embedding_index import EmbeddingIndex, default_encode_fn
from src.hybrid_retriever import HybridRetriever
from src.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL_NAME,
    HYBRID_ALPHA,
)


# ============================================================
# CONFIGURATION
# ============================================================

PDF_PATH = (
    r"D:\projects\sample book files for rag"
    r"\foundations-of-large-language-models-tong-xiao-and-jingbo-zhu-835.pdf"
)

TOP_K = 10
CANDIDATE_POOL = 50


# ============================================================
# EVALUATION QUESTIONS
# ============================================================

EVALUATION_QUESTIONS = [
    {
        "question": "What is Retrieval-Augmented Generation (RAG)?",
        "relevant_terms": [
            "retrieval-augmented generation",
            "retrieval augmented generation",
            "RAG",
        ],
    },
    {
        "question": "How does RAG work with large language models?",
        "relevant_terms": [
            "retrieval-augmented generation",
            "retrieval augmented generation",
            "RAG",
            "large language models",
            "LLM",
        ],
    },
    {
        "question": (
            "How does Retrieval-Augmented Generation work "
            "with large language models?"
        ),
        "relevant_terms": [
            "retrieval-augmented generation",
            "retrieval augmented generation",
            "RAG",
            "large language models",
            "LLM",
        ],
    },
    {
        "question": (
            "RAG retrieval augmented generation LLM retrieve "
            "relevant documents context prompt"
        ),
        "relevant_terms": [
            "retrieval-augmented generation",
            "retrieval augmented generation",
            "RAG",
            "retrieve",
            "relevant",
            "context",
            "prompt",
        ],
    },
]


# ============================================================
# RETRIEVAL METHODS
# ============================================================

METHODS = [
    "BM25",
    "Dense",
    "Weighted Hybrid",
    "RRF Hybrid",
]


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize text for keyword matching.
    """

    return (
        text.lower()
        .replace("-", " ")
        .replace("_", " ")
        .replace("(", " ")
        .replace(")", " ")
        .replace(",", " ")
        .replace(".", " ")
        .replace(":", " ")
        .replace(";", " ")
        .replace("/", " ")
    )


# ============================================================
# RELEVANCE CHECK
# ============================================================

def is_relevant(
    text: str,
    relevant_terms: List[str],
) -> bool:
    """
    Determine whether a chunk is relevant using keyword matching.

    NOTE:
    This is a preliminary evaluation strategy.
    """

    normalized_text = normalize_text(text)

    for term in relevant_terms:

        normalized_term = normalize_text(term)

        if normalized_term in normalized_text:
            return True

    return False


# ============================================================
# FIND RELEVANT CHUNK IDS
# ============================================================

def get_relevant_ids(
    chunks,
    relevant_terms: List[str],
) -> Set[str]:
    """
    Return IDs of chunks considered relevant.
    """

    relevant_ids = set()

    for chunk in chunks:

        if is_relevant(
            chunk.text,
            relevant_terms,
        ):
            relevant_ids.add(chunk.id)

    return relevant_ids


# ============================================================
# RETRIEVE IDS
# ============================================================

def retrieve_ids(
    method: str,
    question: str,
    bm25: BM25Index,
    embedding_index: EmbeddingIndex,
    weighted_retriever: HybridRetriever,
    rrf_retriever: HybridRetriever,
) -> List[str]:
    """
    Run one retrieval method and return ranked chunk IDs.

    BM25/Dense return:
        List[Tuple[str, float]]

    Hybrid returns:
        List[Dict]
    """

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    if method == "BM25":

        results = bm25.search(
            question,
            TOP_K,
        )

        return [
            chunk_id
            for chunk_id, _score in results
        ]

    # --------------------------------------------------------
    # Dense
    # --------------------------------------------------------

    elif method == "Dense":

        results = embedding_index.search(
            question,
            TOP_K,
        )

        return [
            chunk_id
            for chunk_id, _score in results
        ]

    # --------------------------------------------------------
    # Weighted Hybrid
    # --------------------------------------------------------

    elif method == "Weighted Hybrid":

        results = weighted_retriever.search(
            question,
            TOP_K,
            candidate_pool=CANDIDATE_POOL,
        )

        return [
            result["chunk_id"]
            for result in results
        ]

    # --------------------------------------------------------
    # RRF Hybrid
    # --------------------------------------------------------

    elif method == "RRF Hybrid":

        results = rrf_retriever.search(
            question,
            TOP_K,
            candidate_pool=CANDIDATE_POOL,
        )

        return [
            result["chunk_id"]
            for result in results
        ]

    # --------------------------------------------------------
    # Invalid method
    # --------------------------------------------------------

    else:

        raise ValueError(
            f"Unknown retrieval method: {method}"
        )


# ============================================================
# HIT@K
# ============================================================

def hit_at_k(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
) -> int:
    """
    Hit@K = 1 when at least one relevant chunk
    appears in the retrieved results.
    """

    return int(
        any(
            chunk_id in relevant_ids
            for chunk_id in retrieved_ids
        )
    )


# ============================================================
# RECALL@K
# ============================================================

def recall_at_k(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
) -> float:
    """
    Recall@K:

        number of relevant retrieved chunks
        ----------------------------------
        total number of relevant chunks
    """

    if not relevant_ids:
        return 0.0

    retrieved_relevant = sum(
        chunk_id in relevant_ids
        for chunk_id in retrieved_ids
    )

    return (
        retrieved_relevant
        / len(relevant_ids)
    )


# ============================================================
# PRECISION@K
# ============================================================

def precision_at_k(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
) -> float:
    """
    Precision@K:

        number of relevant retrieved chunks
        ----------------------------------
        number of retrieved chunks
    """

    if not retrieved_ids:
        return 0.0

    retrieved_relevant = sum(
        chunk_id in relevant_ids
        for chunk_id in retrieved_ids
    )

    return (
        retrieved_relevant
        / len(retrieved_ids)
    )


# ============================================================
# MRR
# ============================================================

def reciprocal_rank(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
) -> float:
    """
    Calculate reciprocal rank.

    First relevant result:
        Rank 1 -> 1.0
        Rank 2 -> 0.5
        Rank 3 -> 0.3333
    """

    for rank, chunk_id in enumerate(
        retrieved_ids,
        start=1,
    ):

        if chunk_id in relevant_ids:

            return 1.0 / rank

    return 0.0


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RETRIEVAL EVALUATION")
    print("=" * 70)

    # ========================================================
    # VALIDATE PDF
    # ========================================================

    if not os.path.exists(PDF_PATH):

        raise FileNotFoundError(
            f"PDF file not found:\n{PDF_PATH}"
        )

    # ========================================================
    # LOAD PDF
    # ========================================================

    print()
    print("[1] Loading PDF...")

    start_time = time.perf_counter()

    loader = PDFLoader()

    documents = loader.load(
        PDF_PATH
    )

    load_time = (
        time.perf_counter()
        - start_time
    )

    print(
        f"Pages extracted : {len(documents)}"
    )

    print(
        f"Load time       : {load_time:.2f}s"
    )

    # ========================================================
    # BUILD CHUNKS
    # ========================================================

    print()
    print("[2] Building chunks...")

    start_time = time.perf_counter()

    chunks = []

    for document in documents:

        document_chunks = chunk_text(
            document.text,
            document.source,
            CHUNK_SIZE,
            CHUNK_OVERLAP,
            metadata=document.metadata,
        )

        chunks.extend(
            document_chunks
        )

    chunk_time = (
        time.perf_counter()
        - start_time
    )

    print(
        f"Chunks created  : {len(chunks)}"
    )

    print(
        f"Chunk time      : {chunk_time:.2f}s"
    )

    # ========================================================
    # BUILD BM25
    # ========================================================

    print()
    print("[3] Building BM25 index...")

    start_time = time.perf_counter()

    bm25 = BM25Index()

    bm25.build(
        [chunk.id for chunk in chunks],
        [chunk.text for chunk in chunks],
    )

    bm25_time = (
        time.perf_counter()
        - start_time
    )

    print(
        f"BM25 build time : {bm25_time:.2f}s"
    )

    # ========================================================
    # BUILD DENSE INDEX
    # ========================================================

    print()
    print("[4] Building dense embedding index...")

    print(
        f"Embedding model : {EMBEDDING_MODEL_NAME}"
    )

    start_time = time.perf_counter()

    encode_fn = default_encode_fn(
        EMBEDDING_MODEL_NAME
    )

    embedding_index = EmbeddingIndex(
        encode_fn=encode_fn
    )

    embedding_index.build(
        [chunk.id for chunk in chunks],
        [chunk.text for chunk in chunks],
    )

    embedding_time = (
        time.perf_counter()
        - start_time
    )

    print(
        f"Dense build time: {embedding_time:.2f}s"
    )

    # ========================================================
    # WEIGHTED HYBRID
    # ========================================================

    print()
    print("[5] Creating weighted hybrid retriever...")

    weighted_retriever = HybridRetriever(
        bm25_index=bm25,
        embedding_index=embedding_index,
        alpha=HYBRID_ALPHA,
        fusion_method="weighted",
    )

    print(
        f"Hybrid alpha    : {HYBRID_ALPHA}"
    )

    print(
        "Fusion method   : weighted"
    )

    # ========================================================
    # RRF HYBRID
    # ========================================================

    print()
    print("[6] Creating RRF hybrid retriever...")

    rrf_retriever = HybridRetriever(
        bm25_index=bm25,
        embedding_index=embedding_index,
        alpha=HYBRID_ALPHA,
        fusion_method="rrf",
        rrf_k=60,
    )

    print(
        "Fusion method   : RRF"
    )

    print(
        "RRF k           : 60"
    )

    # ========================================================
    # RESULTS STORAGE
    # ========================================================

    results: Dict[
        str,
        Dict[str, List[float]]
    ] = {}

    for method in METHODS:

        results[method] = {
            "hit": [],
            "recall": [],
            "precision": [],
            "mrr": [],
        }

    # ========================================================
    # RUN EVALUATION
    # ========================================================

    print()
    print("=" * 70)
    print("RUNNING EVALUATION QUESTIONS")
    print("=" * 70)

    for question_number, evaluation_item in enumerate(
        EVALUATION_QUESTIONS,
        start=1,
    ):

        question = evaluation_item[
            "question"
        ]

        relevant_terms = evaluation_item[
            "relevant_terms"
        ]

        relevant_ids = get_relevant_ids(
            chunks,
            relevant_terms,
        )

        print()
        print("-" * 70)
        print(
            f"QUESTION {question_number}"
        )
        print("-" * 70)

        print(
            f"Question: {question}"
        )

        print()
        print(
            f"Relevant chunks identified: "
            f"{len(relevant_ids)}"
        )

        # ----------------------------------------------------
        # Evaluate all retrieval methods
        # ----------------------------------------------------

        for method in METHODS:

            retrieved_ids = retrieve_ids(
                method=method,
                question=question,
                bm25=bm25,
                embedding_index=embedding_index,
                weighted_retriever=weighted_retriever,
                rrf_retriever=rrf_retriever,
            )

            hit = hit_at_k(
                retrieved_ids,
                relevant_ids,
            )

            recall = recall_at_k(
                retrieved_ids,
                relevant_ids,
            )

            precision = precision_at_k(
                retrieved_ids,
                relevant_ids,
            )

            mrr = reciprocal_rank(
                retrieved_ids,
                relevant_ids,
            )

            results[method]["hit"].append(
                float(hit)
            )

            results[method]["recall"].append(
                recall
            )

            results[method]["precision"].append(
                precision
            )

            results[method]["mrr"].append(
                mrr
            )

            print()
            print(
                f"{method}"
            )

            print(
                f"  Retrieved chunks : "
                f"{len(retrieved_ids)}"
            )

            print(
                f"  Hit@{TOP_K}       : "
                f"{hit}"
            )

            print(
                f"  Recall@{TOP_K}    : "
                f"{recall:.4f}"
            )

            print(
                f"  Precision@{TOP_K} : "
                f"{precision:.4f}"
            )

            print(
                f"  MRR              : "
                f"{mrr:.4f}"
            )

    # ========================================================
    # FINAL METRICS
    # ========================================================

    print()
    print()
    print("=" * 70)
    print("FINAL EVALUATION RESULTS")
    print("=" * 70)

    print()

    print(
        f"{'Method':<20}"
        f"{'Hit@K':<12}"
        f"{'Recall@K':<12}"
        f"{'Precision@K':<15}"
        f"{'MRR':<10}"
    )

    print("-" * 70)

    final_metrics = {}

    for method in METHODS:

        hit_average = (
            sum(results[method]["hit"])
            / len(results[method]["hit"])
        )

        recall_average = (
            sum(results[method]["recall"])
            / len(results[method]["recall"])
        )

        precision_average = (
            sum(results[method]["precision"])
            / len(results[method]["precision"])
        )

        mrr_average = (
            sum(results[method]["mrr"])
            / len(results[method]["mrr"])
        )

        final_metrics[method] = {
            "hit": hit_average,
            "recall": recall_average,
            "precision": precision_average,
            "mrr": mrr_average,
        }

        print(
            f"{method:<20}"
            f"{hit_average:<12.4f}"
            f"{recall_average:<12.4f}"
            f"{precision_average:<15.4f}"
            f"{mrr_average:<10.4f}"
        )

    # ========================================================
    # BEST METHOD BY MRR
    # ========================================================

    print()
    print("=" * 70)
    print("BEST METHODS")
    print("=" * 70)

    best_mrr_method = max(
        METHODS,
        key=lambda method: final_metrics[method]["mrr"],
    )

    print()
    print(
        f"Best method by MRR: "
        f"{best_mrr_method}"
    )

    print(
        f"MRR score         : "
        f"{final_metrics[best_mrr_method]['mrr']:.4f}"
    )

    # ========================================================
    # BEST METHOD BY HIT@K
    # ========================================================

    best_hit_method = max(
        METHODS,
        key=lambda method: final_metrics[method]["hit"],
    )

    print()
    print(
        f"Best method by Hit@{TOP_K}: "
        f"{best_hit_method}"
    )

    print(
        f"Hit@{TOP_K} score: "
        f"{final_metrics[best_hit_method]['hit']:.4f}"
    )

    # ========================================================
    # BEST METHOD BY RECALL
    # ========================================================

    best_recall_method = max(
        METHODS,
        key=lambda method: final_metrics[method]["recall"],
    )

    print()
    print(
        f"Best method by Recall@{TOP_K}: "
        f"{best_recall_method}"
    )

    print(
        f"Recall@{TOP_K} score: "
        f"{final_metrics[best_recall_method]['recall']:.4f}"
    )

    # ========================================================
    # BEST METHOD BY PRECISION
    # ========================================================

    best_precision_method = max(
        METHODS,
        key=lambda method: final_metrics[method]["precision"],
    )

    print()
    print(
        f"Best method by Precision@{TOP_K}: "
        f"{best_precision_method}"
    )

    print(
        f"Precision@{TOP_K} score: "
        f"{final_metrics[best_precision_method]['precision']:.4f}"
    )

    # ========================================================
    # IMPORTANT NOTE
    # ========================================================

    print()
    print("=" * 70)
    print("IMPORTANT EVALUATION NOTE")
    print("=" * 70)

    print()
    print(
        "The current evaluator identifies relevant chunks "
        "using keyword matching."
    )

    print(
        "This is useful for an initial retrieval check, "
        "but it is not a perfect semantic ground-truth dataset."
    )

    print()
    print(
        "For a production-quality benchmark, we should later "
        "create stable ground-truth relevance labels using "
        "document metadata such as page number and chunk index."
    )

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()