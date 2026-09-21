"""
End-to-end smoke test for the RAG pipeline using a deterministic mock embedder,
so it runs without downloading any real transformer model.

Run with:
    python tests/test_pipeline.py
"""

import sys
import os
import hashlib

import numpy as np

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

from src.chunking import chunk_text, build_chunks_from_dir
from src.bm25_index import BM25Index
from src.embedding_index import EmbeddingIndex, _normalize
from src.hybrid_retriever import HybridRetriever
from src.docstore import DocStore
from src import config


# ============================================================
# Deterministic mock embedding
# ============================================================

def mock_encode_fn(texts):
    """
    Deterministic bag-of-words hashing embedding.

    Used only for tests so the test suite does not download
    or load a real transformer model.
    """
    dim = 64

    vectors = np.zeros(
        (len(texts), dim),
        dtype=np.float32,
    )

    for i, text in enumerate(texts):
        for word in text.lower().split():
            h = int(
                hashlib.md5(word.encode()).hexdigest(),
                16,
            ) % dim

            vectors[i, h] += 1.0

    return _normalize(vectors)


# ============================================================
# Test 1: Chunking
# ============================================================

def test_chunking():
    text = "word " * 1000

    chunks = chunk_text(
        text.strip(),
        source="test.txt",
        chunk_size=300,
        overlap=50,
    )

    assert len(chunks) > 1, (
        "expected multiple chunks"
    )

    assert all(
        chunk.source == "test.txt"
        for chunk in chunks
    ), "all chunks should preserve the source"

    print(
        f"[OK] chunking: {len(chunks)} chunks produced"
    )


# ============================================================
# Test 2: Full retrieval pipeline
# ============================================================

def test_full_pipeline():
    chunks = build_chunks_from_dir(
        config.DOCS_DIR,
        config.CHUNK_SIZE,
        config.CHUNK_OVERLAP,
    )

    assert len(chunks) > 0, (
        "no chunks built from sample_docs"
    )

    chunk_ids = [chunk.id for chunk in chunks]
    texts = [chunk.text for chunk in chunks]

    # --------------------------------------------------------
    # Build BM25
    # --------------------------------------------------------

    bm25 = BM25Index()

    bm25.build(
        chunk_ids,
        texts,
    )

    # --------------------------------------------------------
    # Build deterministic embedding index
    # --------------------------------------------------------

    embed = EmbeddingIndex(
        encode_fn=mock_encode_fn,
    )

    embed.build(
        chunk_ids,
        texts,
    )

    # --------------------------------------------------------
    # Build document store
    # --------------------------------------------------------

    store = DocStore()

    store.add(chunks)

    # ========================================================
    # BM25 lexical retrieval
    # ========================================================

    bm25_results = bm25.search(
        "What is BM25 used for in search?",
        3,
    )

    assert bm25_results, (
        "BM25 search returned no results"
    )

    bm25_top_chunk = store.get(
        bm25_results[0][0]
    )

    assert bm25_top_chunk is not None, (
        "BM25 returned a chunk ID that is missing "
        "from the document store"
    )

    assert "bm25" in bm25_top_chunk.text.lower(), (
        "BM25 lexical retrieval failed to rank "
        "the BM25 chunk highly"
    )

    print(
        "[OK] BM25 retrieval: "
        f"top hit from '{bm25_top_chunk.source}' "
        f"(score={bm25_results[0][1]:.3f})"
    )

    # ========================================================
    # Hybrid retrieval
    # ========================================================

    retriever = HybridRetriever(
        bm25_index=bm25,
        embedding_index=embed,
        alpha=0.5,
        fusion_method="rrf",
        rrf_k=60,
    )

    results = retriever.search(
        "What is BM25 used for in search?",
        top_k=3,
        candidate_pool=10,
    )

    assert len(results) > 0, (
        "hybrid search returned no results"
    )

    top_chunk = store.get(
        results[0]["chunk_id"]
    )

    assert top_chunk is not None, (
        "hybrid retrieval returned a chunk ID "
        "that is missing from the document store"
    )

    print(
        "[OK] hybrid retrieval: "
        f"top hit from '{top_chunk.source}' "
        f"(score={results[0]['score']:.3f})"
    )

    # ========================================================
    # BM25 save/load roundtrip
    # ========================================================

    test_bm25_path = os.path.join(
        config.INDEX_DIR,
        "test_bm25.pkl",
    )

    bm25.save(test_bm25_path)

    bm25_loaded = BM25Index.load(
        test_bm25_path,
    )

    assert bm25_loaded.search(
        "BM25 ranking function",
        3,
    ), "reloaded BM25 index returned nothing"

    print(
        "[OK] bm25 save/load roundtrip"
    )

    # ========================================================
    # Embedding index save/load roundtrip
    # ========================================================

    test_embed_path = os.path.join(
        config.INDEX_DIR,
        "test_embed.pkl",
    )

    embed.save(test_embed_path)

    embed_loaded = EmbeddingIndex.load(
        test_embed_path,
        encode_fn=mock_encode_fn,
    )

    assert embed_loaded.search(
        "hybrid search embeddings",
        3,
    ), "reloaded embedding index returned nothing"

    print(
        "[OK] embedding index save/load roundtrip"
    )


# ============================================================
# Test 3: Flask API
# ============================================================

def test_flask_api():
    import app as flask_app_module

    # --------------------------------------------------------
    # Build deterministic in-memory test indexes
    # --------------------------------------------------------

    chunks = build_chunks_from_dir(
        config.DOCS_DIR,
        config.CHUNK_SIZE,
        config.CHUNK_OVERLAP,
    )

    chunk_ids = [chunk.id for chunk in chunks]
    texts = [chunk.text for chunk in chunks]

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    bm25 = BM25Index()

    bm25.build(
        chunk_ids,
        texts,
    )

    flask_app_module._state["bm25"] = bm25

    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------

    embed = EmbeddingIndex(
        encode_fn=mock_encode_fn,
    )

    embed.build(
        chunk_ids,
        texts,
    )

    flask_app_module._state["embed"] = embed

    # --------------------------------------------------------
    # Document store
    # --------------------------------------------------------

    store = DocStore()

    store.add(chunks)

    flask_app_module._state["store"] = store

    # --------------------------------------------------------
    # Hybrid retriever
    # --------------------------------------------------------

    flask_app_module._state["retriever"] = HybridRetriever(
        bm25_index=bm25,
        embedding_index=embed,
        alpha=0.5,
        fusion_method="rrf",
        rrf_k=60,
    )

    # --------------------------------------------------------
    # Flask test client
    # --------------------------------------------------------

    client = flask_app_module.app.test_client()

    # ========================================================
    # /health
    # ========================================================

    response = client.get("/health")

    assert response.status_code == 200, (
        f"/health expected 200, got {response.status_code}"
    )

    health = response.get_json()

    assert health is not None, (
        "/health did not return JSON"
    )

    assert health["status"] == "ok", (
        "/health status should be 'ok'"
    )

    assert "service" in health, (
        "/health response missing 'service'"
    )

    assert "request_id" in health, (
        "/health response missing 'request_id'"
    )

    assert response.headers.get("X-Request-ID"), (
        "/health response missing X-Request-ID header"
    )

    print(
        "[OK] /health endpoint"
    )

    # ========================================================
    # /ready
    # ========================================================

    response = client.get("/ready")

    assert response.status_code == 200, (
        f"/ready expected 200, got {response.status_code}"
    )

    ready = response.get_json()

    assert ready is not None, (
        "/ready did not return JSON"
    )

    assert ready["status"] == "ready", (
        "/ready status should be 'ready'"
    )

    assert ready["indexes_loaded"] is True, (
        "/ready should report indexes_loaded=true"
    )

    assert "request_id" in ready, (
        "/ready response missing 'request_id'"
    )

    print(
        "[OK] /ready endpoint"
    )

    # ========================================================
    # /status
    # ========================================================

    response = client.get("/status")

    assert response.status_code == 200, (
        f"/status expected 200, got {response.status_code}"
    )

    status = response.get_json()

    assert status is not None, (
        "/status did not return JSON"
    )

    assert isinstance(status, dict), (
        "/status response should be a JSON object"
    )

    print(
        "[OK] /status endpoint"
    )

    # ========================================================
    # /query - retrieval only
    # ========================================================

    response = client.post(
        "/query",
        json={
            "question": "What does Flask do?",
            "top_k": 2,
            "generate": False,
        },
    )

    assert response.status_code == 200, (
        f"/query expected 200, got {response.status_code}"
    )

    data = response.get_json()

    assert data is not None, (
        "/query did not return JSON"
    )

    assert data["question"] == "What does Flask do?", (
        "/query did not preserve the question"
    )

    assert len(data["results"]) > 0, (
        "/query returned no retrieval results"
    )

    assert "retrieval" in data, (
        "/query response missing retrieval diagnostics"
    )

    assert "sources" in data, (
        "/query response missing sources"
    )

    assert isinstance(data["sources"], list), (
        "sources should be a list"
    )

    assert "request_id" in data, (
        "/query response missing request_id"
    )

    assert response.headers.get("X-Request-ID"), (
        "/query response missing X-Request-ID header"
    )

    print(
        "[OK] /query endpoint: "
        f"{len(data['results'])} results returned"
    )

    # ========================================================
    # /query - missing question
    # ========================================================

    response = client.post(
        "/query",
        json={},
    )

    assert response.status_code == 400, (
        "missing question should return HTTP 400"
    )

    error = response.get_json()

    assert error is not None, (
        "missing-question response should be JSON"
    )

    assert "error" in error, (
        "error response missing 'error'"
    )

    print(
        "[OK] /query correctly rejects "
        "missing 'question'"
    )

    # ========================================================
    # /query - empty question
    # ========================================================

    response = client.post(
        "/query",
        json={
            "question": "   ",
        },
    )

    assert response.status_code == 400, (
        "empty question should return HTTP 400"
    )

    print(
        "[OK] /query correctly rejects "
        "empty 'question'"
    )

    # ========================================================
    # /query - invalid top_k
    # ========================================================

    response = client.post(
        "/query",
        json={
            "question": "What is RAG?",
            "top_k": "invalid",
        },
    )

    assert response.status_code == 400, (
        "invalid top_k should return HTTP 400"
    )

    print(
        "[OK] /query correctly rejects "
        "invalid 'top_k'"
    )

    # ========================================================
    # /query - invalid generate
    # ========================================================

    response = client.post(
        "/query",
        json={
            "question": "What is RAG?",
            "generate": "yes",
        },
    )

    assert response.status_code == 400, (
        "non-boolean generate should return HTTP 400"
    )

    print(
        "[OK] /query correctly rejects "
        "invalid 'generate'"
    )

    # ========================================================
    # /query - invalid min_score
    # ========================================================

    response = client.post(
        "/query",
        json={
            "question": "What is RAG?",
            "min_score": "invalid",
        },
    )

    assert response.status_code == 400, (
        "invalid min_score should return HTTP 400"
    )

    print(
        "[OK] /query correctly rejects "
        "invalid 'min_score'"
    )

    # ========================================================
    # /query - top_k > candidate_pool
    # ========================================================

    response = client.post(
        "/query",
        json={
            "question": "What is RAG?",
            "top_k": 10,
            "candidate_pool": 5,
        },
    )

    assert response.status_code == 400, (
        "top_k > candidate_pool should return HTTP 400"
    )

    print(
        "[OK] /query correctly rejects "
        "top_k > candidate_pool"
    )

    # ========================================================
    # Non-JSON request
    # ========================================================

    response = client.post(
        "/query",
        data="not json",
        content_type="text/plain",
    )

    assert response.status_code == 400, (
        "non-JSON request should return HTTP 400"
    )

    print(
        "[OK] /query correctly rejects "
        "non-JSON request"
    )

    # ========================================================
    # 404 handling
    # ========================================================

    response = client.get(
        "/does-not-exist"
    )

    assert response.status_code == 404, (
        "unknown route should return HTTP 404"
    )

    error = response.get_json()

    assert error is not None, (
        "404 response should be JSON"
    )

    assert "error" in error, (
        "404 response missing 'error'"
    )

    print(
        "[OK] 404 error handling"
    )

    # ========================================================
    # 405 handling
    # ========================================================

    response = client.get(
        "/query"
    )

    assert response.status_code == 405, (
        "GET /query should return HTTP 405"
    )

    print(
        "[OK] 405 method handling"
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("RAG PIPELINE END-TO-END SMOKE TEST")
    print("=" * 70)

    test_chunking()
    test_full_pipeline()
    test_flask_api()

    print()
    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)