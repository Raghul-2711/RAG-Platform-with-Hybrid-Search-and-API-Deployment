"""
Ingestion entry point.

Usage:
    python ingest.py

Reads every supported document from sample_docs/, chunks it, builds a BM25 index
and a dense embedding index, and persists all three (+ the docstore) to
storage/ so app.py can load them at request time without re-embedding.
"""

import sys

from src import config
from src.chunking import build_chunks_from_dir
from src.bm25_index import BM25Index
from src.embedding_index import EmbeddingIndex, default_encode_fn
from src.docstore import DocStore


def main():
    print(f"Loading documents from {config.DOCS_DIR} ...")
    chunks = build_chunks_from_dir(config.DOCS_DIR, config.CHUNK_SIZE, config.CHUNK_OVERLAP)

    if not chunks:
        print(
            f"No supported documents found in {config.DOCS_DIR}. "
            "Add some documents and re-run."
        )
        sys.exit(1)

    print(f"Built {len(chunks)} chunks from the source documents.")

    chunk_ids = [c.id for c in chunks]
    texts = [c.text for c in chunks]

    print("Building BM25 index...")
    bm25 = BM25Index()
    bm25.build(chunk_ids, texts)
    bm25.save(config.BM25_INDEX_PATH)

    print(f"Loading embedding model '{config.EMBEDDING_MODEL_NAME}' (downloads on first run)...")
    encode_fn = default_encode_fn(config.EMBEDDING_MODEL_NAME)

    print("Building embedding index...")
    emb_index = EmbeddingIndex(encode_fn=encode_fn)
    emb_index.build(chunk_ids, texts)
    emb_index.save(config.EMBED_INDEX_PATH)

    print("Saving docstore...")
    store = DocStore()
    store.add(chunks)
    store.save(config.DOCSTORE_PATH)

    print("Done. Indexes written to:", config.INDEX_DIR)


if __name__ == "__main__":
    main()
