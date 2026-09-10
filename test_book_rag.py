import time
import torch

from src.loaders import PDFLoader
from src.chunking import chunk_text
from src.bm25_index import BM25Index
from src.embedding_index import EmbeddingIndex, default_encode_fn
from src.hybrid_retriever import HybridRetriever
from src.llm_client import generate_answer
from src import config


PDF_PATH = r"D:\projects\sample book files for rag\foundations-of-large-language-models-tong-xiao-and-jingbo-zhu-835.pdf"


def main():
    print("=" * 70)
    print("REAL PDF -> HYBRID RETRIEVAL -> GEMINI RAG TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Load PDF
    # ---------------------------------------------------------
    print("\n[1] Loading PDF...")

    loader = PDFLoader()

    start = time.perf_counter()
    documents = loader.load(PDF_PATH)
    load_time = time.perf_counter() - start

    print(f"Pages extracted : {len(documents)}")
    print(f"Load time       : {load_time:.2f}s")

    # ---------------------------------------------------------
    # 2. Chunk every PDF page
    # ---------------------------------------------------------
    print("\n[2] Building chunks...")

    start = time.perf_counter()

    chunks = []

    for document in documents:
        page_chunks = chunk_text(
            text=document.text,
            source=document.source,
            chunk_size=config.CHUNK_SIZE,
            overlap=config.CHUNK_OVERLAP,
            metadata=document.metadata,
        )

        total_chunks = len(page_chunks)

        for chunk in page_chunks:
            chunk.metadata.update(
                {
                    "document_name": document.source,
                    "total_chunks": total_chunks,
                }
            )

        chunks.extend(page_chunks)

    chunk_time = time.perf_counter() - start

    print(f"Chunks created  : {len(chunks)}")
    print(f"Chunk time      : {chunk_time:.2f}s")

    # ---------------------------------------------------------
    # 3. Build BM25
    # ---------------------------------------------------------
    print("\n[3] Building BM25 index...")

    chunk_ids = [c.id for c in chunks]
    texts = [c.text for c in chunks]

    start = time.perf_counter()

    bm25 = BM25Index()
    bm25.build(chunk_ids, texts)

    bm25_time = time.perf_counter() - start

    print(f"BM25 time       : {bm25_time:.2f}s")

    # ---------------------------------------------------------
    # 4. Build dense embeddings
    # ---------------------------------------------------------
    print("\n[4] Building embedding index...")

    print(f"CUDA available  : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU             : {torch.cuda.get_device_name(0)}")

    encode_fn = default_encode_fn(config.EMBEDDING_MODEL_NAME)

    start = time.perf_counter()

    embed = EmbeddingIndex(encode_fn=encode_fn)
    embed.build(chunk_ids, texts)

    embed_time = time.perf_counter() - start

    print(f"Embedding time  : {embed_time:.2f}s")

    # ---------------------------------------------------------
    # 5. Create hybrid retriever
    # ---------------------------------------------------------
    print("\n[5] Creating hybrid retriever...")

    retriever = HybridRetriever(
        bm25_index=bm25,
        embedding_index=embed,
        alpha=config.HYBRID_ALPHA,
    )

    print(f"Hybrid alpha    : {config.HYBRID_ALPHA}")

    # ---------------------------------------------------------
    # 6. Question
    # ---------------------------------------------------------
    question = "How does RAG work with large language models?"

    print("\n" + "=" * 70)
    print("QUESTION")
    print("=" * 70)
    print(question)

    # ---------------------------------------------------------
    # 7. Hybrid retrieval
    # ---------------------------------------------------------
    start = time.perf_counter()

    results = retriever.search(
        question,
        top_k=5,
        candidate_pool=25,
    )

    retrieval_time = time.perf_counter() - start

    print(f"\nRetrieval time  : {retrieval_time:.4f}s")
    print(f"Results         : {len(results)}")

    # ---------------------------------------------------------
    # 8. Map retrieved IDs back to chunks
    # ---------------------------------------------------------
    chunk_map = {c.id: c for c in chunks}

    hits = []

    print("\n" + "=" * 70)
    print("RETRIEVED CONTEXT")
    print("=" * 70)

    for i, result in enumerate(results, start=1):
        chunk = chunk_map[result["chunk_id"]]

        page = chunk.metadata.get("page_number")
        source = chunk.source

        print(f"\n--- Result #{i} ---")
        print(f"Source       : {source}")
        print(f"Page         : {page}")
        print(f"Fused score  : {result['score']:.4f}")
        print(f"BM25 score   : {result['bm25_score']:.4f}")
        print(f"Dense score  : {result['dense_score']:.4f}")

        print("\nText:")
        print(chunk.text[:1000])

        hits.append(chunk)

    # ---------------------------------------------------------
    # 9. Generate grounded answer using Gemini
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("GENERATING ANSWER WITH GEMINI")
    print("=" * 70)

    contexts = [chunk.text for chunk in hits if chunk.text]

    start = time.perf_counter()

    answer = generate_answer(
        question,
        contexts,
    )

    generation_time = time.perf_counter() - start

    # ---------------------------------------------------------
    # 10. Final answer
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("GEMINI ANSWER")
    print("=" * 70)

    print(answer)

    # ---------------------------------------------------------
    # 11. Timing summary
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("TIMING")
    print("=" * 70)

    print(f"PDF loading     : {load_time:.2f}s")
    print(f"Chunking        : {chunk_time:.2f}s")
    print(f"BM25            : {bm25_time:.2f}s")
    print(f"Embeddings      : {embed_time:.2f}s")
    print(f"Retrieval       : {retrieval_time:.4f}s")
    print(f"Gemini          : {generation_time:.2f}s")

    print("\n" + "=" * 70)
    print("END-TO-END TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()