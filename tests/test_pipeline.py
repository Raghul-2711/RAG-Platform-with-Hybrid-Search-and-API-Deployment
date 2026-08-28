"""
End-to-end smoke test for the RAG pipeline using a deterministic mock embedder,
so it runs without downloading any real transformer model (useful for CI /
offline environments). Run with: python tests/test_pipeline.py
"""

import sys
import os
import hashlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chunking import chunk_text, build_chunks_from_dir
from src.bm25_index import BM25Index
from src.embedding_index import EmbeddingIndex, _normalize
from src.hybrid_retriever import HybridRetriever
from src.docstore import DocStore
from src import config


def mock_encode_fn(texts):
    """Deterministic bag-of-words hashing embedding — good enough to prove the
    plumbing works without needing a real sentence-transformers model."""
    dim = 64
    vectors = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for word in t.lower().split():
            h = int(hashlib.md5(word.encode()).hexdigest(), 16) % dim
            vectors[i, h] += 1.0
    return _normalize(vectors)


def test_chunking():
    text = "word " * 1000
    chunks = chunk_text(text.strip(), source="test.txt", chunk_size=300, overlap=50)
    assert len(chunks) > 1, "expected multiple chunks"
    assert all(c.source == "test.txt" for c in chunks)
    print(f"[OK] chunking: {len(chunks)} chunks produced")


def test_full_pipeline():
    chunks = build_chunks_from_dir(config.DOCS_DIR, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    assert len(chunks) > 0, "no chunks built from sample_docs"

    chunk_ids = [c.id for c in chunks]
    texts = [c.text for c in chunks]

    bm25 = BM25Index()
    bm25.build(chunk_ids, texts)

    embed = EmbeddingIndex(encode_fn=mock_encode_fn)
    embed.build(chunk_ids, texts)

    store = DocStore()
    store.add(chunks)

    retriever = HybridRetriever(bm25, embed, alpha=0.5)
    results = retriever.search("What is BM25 used for in search?", top_k=3)

    assert len(results) > 0, "hybrid search returned no results"
    top_chunk = store.get(results[0]["chunk_id"])
    assert top_chunk is not None
    assert "bm25" in top_chunk.text.lower(), "expected the BM25 chunk to rank highly for a BM25 question"
    print(f"[OK] hybrid retrieval: top hit from '{top_chunk.source}' (score={results[0]['score']:.3f})")

    # save/load roundtrip
    bm25.save(os.path.join(config.INDEX_DIR, "test_bm25.pkl"))
    bm25_loaded = BM25Index.load(os.path.join(config.INDEX_DIR, "test_bm25.pkl"))
    assert bm25_loaded.search("BM25 ranking function", 3), "reloaded bm25 index returned nothing"
    print("[OK] bm25 save/load roundtrip")

    embed.save(os.path.join(config.INDEX_DIR, "test_embed.pkl"))
    embed_loaded = EmbeddingIndex.load(os.path.join(config.INDEX_DIR, "test_embed.pkl"), encode_fn=mock_encode_fn)
    assert embed_loaded.search("hybrid search embeddings", 3), "reloaded embedding index returned nothing"
    print("[OK] embedding index save/load roundtrip")


def test_flask_api():
    import app as flask_app_module

    flask_app_module._state["bm25"] = BM25Index()
    chunks = build_chunks_from_dir(config.DOCS_DIR, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    chunk_ids = [c.id for c in chunks]
    texts = [c.text for c in chunks]
    flask_app_module._state["bm25"].build(chunk_ids, texts)

    embed = EmbeddingIndex(encode_fn=mock_encode_fn)
    embed.build(chunk_ids, texts)
    flask_app_module._state["embed"] = embed

    store = DocStore()
    store.add(chunks)
    flask_app_module._state["store"] = store

    flask_app_module._state["retriever"] = HybridRetriever(
        flask_app_module._state["bm25"], embed, alpha=0.5
    )

    client = flask_app_module.app.test_client()

    r = client.get("/health")
    assert r.status_code == 200 and r.get_json()["indexes_loaded"] is True
    print("[OK] /health endpoint")

    r = client.post("/query", json={"question": "What does Flask do?", "top_k": 2})
    assert r.status_code == 200
    data = r.get_json()
    assert len(data["results"]) > 0
    print(f"[OK] /query endpoint: {len(data['results'])} results returned")

    r = client.post("/query", json={})
    assert r.status_code == 400
    print("[OK] /query correctly rejects missing 'question'")


if __name__ == "__main__":
    test_chunking()
    test_full_pipeline()
    test_flask_api()
    print("\nAll tests passed.")
