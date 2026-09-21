"""
Production ingestion entry point.

Supports:
    TXT
    MD
    PDF
    DOCX
    PPTX

The ingestion pipeline:
    documents
        -> chunks
        -> BM25 index
        -> dense embedding index
        -> docstore

Usage:
    python ingest.py
"""

import sys
import time
from collections import Counter

from src import config
from src.chunking import build_chunks_from_dir
from src.bm25_index import BM25Index
from src.embedding_index import EmbeddingIndex, default_encode_fn
from src.docstore import DocStore


def main():
    start_time = time.perf_counter()

    print("=" * 70)
    print("ENTERPRISE RAG - MULTI-DOCUMENT INGESTION")
    print("=" * 70)

    print(f"\nDocument directory : {config.DOCS_DIR}")

    # ---------------------------------------------------------
    # 1. Load + chunk all supported documents
    # ---------------------------------------------------------
    print("\n[1] Loading and chunking documents...")

    chunks = build_chunks_from_dir(
        config.DOCS_DIR,
        config.CHUNK_SIZE,
        config.CHUNK_OVERLAP,
    )

    if not chunks:
        print(
            f"\nERROR: No supported documents found in "
            f"{config.DOCS_DIR}"
        )
        print("Supported formats: TXT, MD, PDF, DOCX, PPTX")
        sys.exit(1)

    print(f"Chunks created     : {len(chunks)}")

    # ---------------------------------------------------------
    # 2. Document statistics
    # ---------------------------------------------------------
    document_counter = Counter(
        chunk.source
        for chunk in chunks
    )

    extension_counter = Counter()

    for chunk in chunks:
        source = chunk.source.lower()

        if "." in source:
            extension = source.rsplit(".", 1)[-1]
            extension_counter[extension] += 1

    print(f"Documents indexed  : {len(document_counter)}")

    print("\nDocuments:")
    for source, count in sorted(document_counter.items()):
        print(f"  - {source}: {count} chunks")

    print("\nFormats:")
    for extension, count in sorted(extension_counter.items()):
        print(f"  - .{extension}: {count} chunks")

    # ---------------------------------------------------------
    # 3. Prepare index data
    # ---------------------------------------------------------
    chunk_ids = [chunk.id for chunk in chunks]
    texts = [chunk.text for chunk in chunks]

    # Safety check: chunk IDs must be unique.
    if len(chunk_ids) != len(set(chunk_ids)):
        print("\nERROR: Duplicate chunk IDs detected.")
        sys.exit(1)

    # ---------------------------------------------------------
    # 4. Build BM25
    # ---------------------------------------------------------
    print("\n[2] Building BM25 index...")

    bm25_start = time.perf_counter()

    bm25 = BM25Index()
    bm25.build(
        chunk_ids,
        texts,
    )
    bm25.save(
        config.BM25_INDEX_PATH
    )

    bm25_time = time.perf_counter() - bm25_start

    print(
        f"BM25 complete     : "
        f"{bm25_time:.3f}s"
    )

    # ---------------------------------------------------------
    # 5. Build dense embedding index
    # ---------------------------------------------------------
    print("\n[3] Loading embedding model...")

    encode_fn = default_encode_fn(
        config.EMBEDDING_MODEL_NAME
    )

    print(
        f"Embedding model   : "
        f"{config.EMBEDDING_MODEL_NAME}"
    )

    print("\n[4] Building dense embedding index...")

    embedding_start = time.perf_counter()

    embedding_index = EmbeddingIndex(
        encode_fn=encode_fn
    )

    embedding_index.build(
        chunk_ids,
        texts,
    )

    embedding_index.save(
        config.EMBED_INDEX_PATH
    )

    embedding_time = (
        time.perf_counter()
        - embedding_start
    )

    print(
        f"Dense index done  : "
        f"{embedding_time:.3f}s"
    )

    # ---------------------------------------------------------
    # 6. Save document store
    # ---------------------------------------------------------
    print("\n[5] Saving document store...")

    store = DocStore()

    store.add(chunks)

    store.save(
        config.DOCSTORE_PATH
    )

    # ---------------------------------------------------------
    # 7. Final statistics
    # ---------------------------------------------------------
    total_time = (
        time.perf_counter()
        - start_time
    )

    print("\n" + "=" * 70)
    print("INGESTION COMPLETE")
    print("=" * 70)

    print(f"Documents         : {len(document_counter)}")
    print(f"Chunks            : {len(chunks)}")
    print(f"BM25 time         : {bm25_time:.3f}s")
    print(f"Dense time        : {embedding_time:.3f}s")
    print(f"Total time        : {total_time:.3f}s")
    print(f"Index directory   : {config.INDEX_DIR}")

    print("\nIndexed documents:")
    for source in sorted(document_counter):
        print(f"  ✓ {source}")

    print("\nReady for multi-document RAG.")


if __name__ == "__main__":
    main()
