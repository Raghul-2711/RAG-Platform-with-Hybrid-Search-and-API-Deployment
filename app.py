"""
Flask REST API for the RAG Platform.

Endpoints:
    GET  /health              -> liveness check
    POST /ingest               -> (re)build indexes from sample_docs/
    POST /query                -> hybrid search, optional LLM generation

Run:
    python app.py
"""

import os
import traceback

from flask import Flask, request, jsonify

from src import config
from src.bm25_index import BM25Index
from src.embedding_index import EmbeddingIndex, default_encode_fn
from src.docstore import DocStore
from src.hybrid_retriever import HybridRetriever
from src.chunking import build_chunks_from_dir
from src.llm_client import generate_answer

app = Flask(__name__)

# Lazily-populated global state. Loaded on first request or via /ingest.
_state = {"bm25": None, "embed": None, "store": None, "retriever": None}


def _indexes_exist() -> bool:
    return (
        os.path.exists(config.BM25_INDEX_PATH)
        and os.path.exists(config.EMBED_INDEX_PATH)
        and os.path.exists(config.DOCSTORE_PATH)
    )


def _load_state():
    encode_fn = default_encode_fn(config.EMBEDDING_MODEL_NAME)
    bm25 = BM25Index.load(config.BM25_INDEX_PATH)
    embed = EmbeddingIndex.load(config.EMBED_INDEX_PATH, encode_fn=encode_fn)
    store = DocStore.load(config.DOCSTORE_PATH)
    retriever = HybridRetriever(bm25, embed, alpha=config.HYBRID_ALPHA)
    _state.update({"bm25": bm25, "embed": embed, "store": store, "retriever": retriever})


def _ensure_loaded():
    if _state["retriever"] is None:
        if not _indexes_exist():
            raise RuntimeError(
                "No indexes found. Run `python ingest.py` first, or POST /ingest."
            )
        _load_state()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "indexes_loaded": _state["retriever"] is not None})


@app.route("/ingest", methods=["POST"])
def ingest():
    """Rebuilds the BM25 + embedding indexes from sample_docs/ (or a custom dir)."""
    try:
        docs_dir = request.json.get("docs_dir", config.DOCS_DIR) if request.is_json else config.DOCS_DIR
        chunks = build_chunks_from_dir(docs_dir, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        if not chunks:
            return jsonify({"error": f"No .txt/.md files found in {docs_dir}"}), 400

        chunk_ids = [c.id for c in chunks]
        texts = [c.text for c in chunks]

        bm25 = BM25Index()
        bm25.build(chunk_ids, texts)
        bm25.save(config.BM25_INDEX_PATH)

        encode_fn = default_encode_fn(config.EMBEDDING_MODEL_NAME)
        embed = EmbeddingIndex(encode_fn=encode_fn)
        embed.build(chunk_ids, texts)
        embed.save(config.EMBED_INDEX_PATH)

        store = DocStore()
        store.add(chunks)
        store.save(config.DOCSTORE_PATH)

        retriever = HybridRetriever(bm25, embed, alpha=config.HYBRID_ALPHA)
        _state.update({"bm25": bm25, "embed": embed, "store": store, "retriever": retriever})

        return jsonify({"status": "ok", "chunks_indexed": len(chunks)})
    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/query", methods=["POST"])
def query():
    try:
        if not request.is_json:
            return jsonify({"error": "Request body must be JSON"}), 400

        body = request.get_json()
        question = body.get("question")
        if not question:
            return jsonify({"error": "'question' is required"}), 400

        top_k = int(body.get("top_k", config.TOP_K))
        generate = bool(body.get("generate", False))

        _ensure_loaded()

        results = _state["retriever"].search(question, top_k=top_k)
        store = _state["store"]

        hits = []
        for r in results:
            chunk = store.get(r["chunk_id"])
            hits.append({
                "chunk_id": r["chunk_id"],
                "source": chunk.source if chunk else None,
                "text": chunk.text if chunk else None,
                "score": round(r["score"], 4),
                "bm25_score": round(r["bm25_score"], 4),
                "dense_score": round(r["dense_score"], 4),
            })

        response = {"question": question, "results": hits}

        if generate:
            contexts = [h["text"] for h in hits if h["text"]]
            response["answer"] = generate_answer(question, contexts)

        return jsonify(response)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG", "0") == "1")
