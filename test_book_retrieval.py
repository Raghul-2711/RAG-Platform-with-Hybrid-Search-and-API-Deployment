import time

import torch

from src import config
from src.loaders import PDFLoader
from src.chunking import chunk_text
from src.bm25_index import BM25Index
from src.embedding_index import EmbeddingIndex, default_encode_fn
from src.hybrid_retriever import HybridRetriever


PDF_PATH = r"D:\projects\sample book files for rag\foundations-of-large-language-models-tong-xiao-and-jingbo-zhu-835.pdf"

TOP_K = 10
CANDIDATE_POOL = 50


def find_chunk(chunks, chunk_id):
    """Find a chunk by its ID."""
    for chunk in chunks:
        if chunk.id == chunk_id:
            return chunk

    return None


def print_chunk_preview(chunk, max_chars=700):
    """Print a readable preview of a retrieved chunk."""
    if chunk is None:
        print("Chunk not found.")
        return

    text = chunk.text.replace("\n", " ").strip()

    if len(text) > max_chars:
        text = text[:max_chars] + "..."

    print(f"Page : {chunk.metadata.get('page', 'N/A')}")
    print(f"Text : {text}")


def run_bm25_diagnostic(bm25, chunks, question):
    """Show the top BM25 results."""
    print("\n" + "=" * 70)
    print("BM25 TOP 10")
    print("=" * 70)

    results = bm25.search(
        question,
        CANDIDATE_POOL,
    )

    for rank, (chunk_id, score) in enumerate(
        results[:TOP_K],
        start=1,
    ):
        chunk = find_chunk(chunks, chunk_id)

        print(
            f"\nBM25 #{rank}"
            f" | score={score:.4f}"
            f" | chunk={chunk_id}"
        )

        print_chunk_preview(chunk)


def run_dense_diagnostic(embedding_index, chunks, question):
    """Show the top dense retrieval results."""
    print("\n" + "=" * 70)
    print("DENSE TOP 10")
    print("=" * 70)

    results = embedding_index.search(
        question,
        CANDIDATE_POOL,
    )

    for rank, (chunk_id, score) in enumerate(
        results[:TOP_K],
        start=1,
    ):
        chunk = find_chunk(chunks, chunk_id)

        print(
            f"\nDense #{rank}"
            f" | score={score:.4f}"
            f" | chunk={chunk_id}"
        )

        print_chunk_preview(chunk)


def run_hybrid_diagnostic(
    retriever,
    chunks,
    question,
    title,
):
    """Show the top hybrid retrieval results."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    results = retriever.search(
        question,
        top_k=TOP_K,
        candidate_pool=CANDIDATE_POOL,
    )

    for rank, result in enumerate(
        results,
        start=1,
    ):
        chunk = find_chunk(
            chunks,
            result["chunk_id"],
        )

        print(
            f"\nHybrid #{rank}"
            f" | score={result['score']:.6f}"
            f" | BM25={result['bm25_score']:.4f}"
            f" | Dense={result['dense_score']:.4f}"
            f" | chunk={result['chunk_id']}"
        )

        print_chunk_preview(chunk)


def main():
    print("=" * 70)
    print("REAL BOOK HYBRID RETRIEVAL - RRF UPGRADE TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Load PDF
    # ---------------------------------------------------------
    print("\n[1] Loading PDF...")

    start = time.perf_counter()

    loader = PDFLoader()
    documents = loader.load(PDF_PATH)

    load_time = time.perf_counter() - start

    print(f"Pages extracted: {len(documents)}")
    print(f"Load time       : {load_time:.2f}s")

    # ---------------------------------------------------------
    # 2. Create chunks
    # ---------------------------------------------------------
    print("\n[2] Creating chunks...")

    start = time.perf_counter()

    chunks = []

    for document in documents:
        document_chunks = chunk_text(
        document.text,
        document.source,
        config.CHUNK_SIZE,
        config.CHUNK_OVERLAP,
        metadata=document.metadata,
        )
        

        chunks.extend(document_chunks)

    chunk_time = time.perf_counter() - start

    print(f"Total chunks    : {len(chunks)}")
    print(f"Chunk time      : {chunk_time:.2f}s")

    # ---------------------------------------------------------
    # 3. BM25 index
    # ---------------------------------------------------------
    print("\n[3] Building BM25 index...")

    start = time.perf_counter()

    bm25 = BM25Index()
    bm25.build(
        [chunk.id for chunk in chunks],
        [chunk.text for chunk in chunks],
        )

    bm25_time = time.perf_counter() - start

    print(f"BM25 build time : {bm25_time:.2f}s")

    # ---------------------------------------------------------
    # 4. Dense embedding index
    # ---------------------------------------------------------
    print("\n[4] Building embedding index...")

    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU           : {torch.cuda.get_device_name(0)}")
        print(f"CUDA version  : {torch.version.cuda}")

    start = time.perf_counter()

    embedding_index = EmbeddingIndex(
        encode_fn=default_encode_fn(config.EMBEDDING_MODEL_NAME),
        )
    embedding_index.build(
        [chunk.id for chunk in chunks],
        [chunk.text for chunk in chunks],
        )

    embedding_time = time.perf_counter() - start

    print(f"Embedding build time: {embedding_time:.2f}s")

    # ---------------------------------------------------------
    # 5. Create RRF hybrid retriever
    # ---------------------------------------------------------
    print("\n[5] Creating RRF hybrid retriever...")

    rrf_retriever = HybridRetriever(
        bm25_index=bm25,
        embedding_index=embedding_index,
        alpha=config.HYBRID_ALPHA,
        fusion_method="rrf",
        rrf_k=60,
    )

    print("RRF retriever created successfully.")
    print(f"Fusion method : {rrf_retriever.fusion_method}")
    print(f"RRF k         : {rrf_retriever.rrf_k}")
    print(f"Candidate pool: {CANDIDATE_POOL}")

    # ---------------------------------------------------------
    # 6. Create weighted hybrid retriever
    # ---------------------------------------------------------
    print("\n[6] Creating weighted baseline retriever...")

    weighted_retriever = HybridRetriever(
        bm25_index=bm25,
        embedding_index=embedding_index,
        alpha=config.HYBRID_ALPHA,
        fusion_method="weighted",
        rrf_k=60,
    )

    print("Weighted retriever created successfully.")
    print(f"Fusion method : {weighted_retriever.fusion_method}")
    print(f"Alpha         : {weighted_retriever.alpha}")

    # ---------------------------------------------------------
    # 7. Test questions
    # ---------------------------------------------------------
    questions = [
        "What is Retrieval-Augmented Generation (RAG)?",
        "How does RAG work with large language models?",
        "How does Retrieval-Augmented Generation work with large language models?",
        "RAG retrieval augmented generation LLM retrieve relevant documents context prompt",
    ]

    # ---------------------------------------------------------
    # 8. Run diagnostics
    # ---------------------------------------------------------
    for question_number, question in enumerate(
        questions,
        start=1,
    ):
        print("\n\n" + "#" * 70)
        print(f"QUESTION {question_number}")
        print("#" * 70)
        print(f"\nQuery: {question}")

        run_bm25_diagnostic(
            bm25,
            chunks,
            question,
        )

        run_dense_diagnostic(
            embedding_index,
            chunks,
            question,
        )

        run_hybrid_diagnostic(
            weighted_retriever,
            chunks,
            question,
            "WEIGHTED HYBRID TOP 10",
        )

        run_hybrid_diagnostic(
            rrf_retriever,
            chunks,
            question,
            "RRF HYBRID TOP 10",
        )

    print("\n\n" + "=" * 70)
    print("RRF RETRIEVAL TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()