"""
Flask REST API for the Enterprise RAG Platform.

Endpoints:
    GET  /health
    POST /ingest
    POST /query

The /query endpoint supports:
    - hybrid retrieval
    - configurable top-k
    - candidate pools
    - minimum score threshold
    - retrieval diagnostics
    - grounded LLM generation
    - source/page/slide metadata
"""

import os
import time
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


_state = {
    "bm25": None,
    "embed": None,
    "store": None,
    "retriever": None,
}


def _indexes_exist() -> bool:
    """Return True when all persisted indexes exist."""

    return (
        os.path.exists(config.BM25_INDEX_PATH)
        and os.path.exists(config.EMBED_INDEX_PATH)
        and os.path.exists(config.DOCSTORE_PATH)
    )


def _create_retriever(
    bm25,
    embed,
):
    """Create the application's configured hybrid retriever."""

    return HybridRetriever(
        bm25_index=bm25,
        embedding_index=embed,
        alpha=config.HYBRID_ALPHA,
        fusion_method="weighted",
    )


def _load_state():
    """Load persisted indexes into memory."""

    encode_fn = default_encode_fn(
        config.EMBEDDING_MODEL_NAME
    )

    bm25 = BM25Index.load(
        config.BM25_INDEX_PATH
    )

    embed = EmbeddingIndex.load(
        config.EMBED_INDEX_PATH,
        encode_fn=encode_fn,
    )

    store = DocStore.load(
        config.DOCSTORE_PATH
    )

    retriever = _create_retriever(
        bm25,
        embed,
    )

    _state.update(
        {
            "bm25": bm25,
            "embed": embed,
            "store": store,
            "retriever": retriever,
        }
    )


def _ensure_loaded():
    """Load indexes when necessary."""

    if _state["retriever"] is not None:
        return

    if not _indexes_exist():
        raise RuntimeError(
            "No indexes found. Run `python ingest.py` first, "
            "or POST /ingest."
        )

    _load_state()


def _safe_int(
    value,
    default,
    minimum=1,
):
    """Safely parse an integer request parameter."""

    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default

    return max(parsed, minimum)


def _safe_float_or_none(value):
    """Parse an optional float."""

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(
            "min_score must be a valid number"
        )


def _serialize_hit(result, store):
    """Convert an internal retrieval result into API output."""

    chunk_id = result["chunk_id"]
    chunk = store.get(chunk_id)

    if chunk is None:
        return {
            "chunk_id": chunk_id,
            "source": None,
            "text": None,
            "metadata": {},
            "score": round(result["score"], 6),
            "bm25_score": round(
                result["bm25_score"],
                6,
            ),
            "dense_score": round(
                result["dense_score"],
                6,
            ),
            "retrieved_by_bm25": result.get(
                "retrieved_by_bm25",
                False,
            ),
            "retrieved_by_dense": result.get(
                "retrieved_by_dense",
                False,
            ),
        }

    return {
        "chunk_id": chunk.id,
        "source": chunk.source,
        "text": chunk.text,
        "metadata": dict(chunk.metadata or {}),
        "score": round(result["score"], 6),
        "bm25_score": round(
            result["bm25_score"],
            6,
        ),
        "dense_score": round(
            result["dense_score"],
            6,
        ),
        "retrieved_by_bm25": result.get(
            "retrieved_by_bm25",
            False,
        ),
        "retrieved_by_dense": result.get(
            "retrieved_by_dense",
            False,
        ),
    }


@app.route("/health", methods=["GET"])
def health():
    """Application health check."""

    return jsonify(
        {
            "status": "ok",
            "indexes_loaded": (
                _state["retriever"] is not None
            ),
            "llm_provider": config.LLM_PROVIDER,
            "embedding_model": config.EMBEDDING_MODEL_NAME,
        }
    )


@app.route("/ingest", methods=["POST"])
def ingest():
    """Build BM25, dense embedding, and document-store indexes."""

    start_time = time.perf_counter()

    try:
        if request.is_json:
            body = request.get_json(silent=True) or {}
            docs_dir = body.get(
                "docs_dir",
                config.DOCS_DIR,
            )
        else:
            docs_dir = config.DOCS_DIR

        if not isinstance(docs_dir, str) or not docs_dir.strip():
            return jsonify(
                {"error": "docs_dir must be a non-empty string"}
            ), 400

        if not os.path.isdir(docs_dir):
            return jsonify(
                {
                    "error": (
                        f"Document directory does not exist: "
                        f"{docs_dir}"
                    )
                }
            ), 400

        chunks = build_chunks_from_dir(
            docs_dir,
            config.CHUNK_SIZE,
            config.CHUNK_OVERLAP,
        )

        if not chunks:
            return jsonify(
                {
                    "error": (
                        f"No supported documents found in {docs_dir}. "
                        "Supported formats: TXT, MD, PDF, DOCX, PPTX."
                    )
                }
            ), 400

        chunk_ids = [
            chunk.id
            for chunk in chunks
        ]

        texts = [
            chunk.text
            for chunk in chunks
        ]

        # BM25
        bm25 = BM25Index()

        bm25.build(
            chunk_ids,
            texts,
        )

        bm25.save(
            config.BM25_INDEX_PATH
        )

        # Dense embeddings
        encode_fn = default_encode_fn(
            config.EMBEDDING_MODEL_NAME
        )

        embed = EmbeddingIndex(
            encode_fn=encode_fn
        )

        embed.build(
            chunk_ids,
            texts,
        )

        embed.save(
            config.EMBED_INDEX_PATH
        )

        # Document store
        store = DocStore()

        store.add(chunks)

        store.save(
            config.DOCSTORE_PATH
        )

        # Runtime retriever
        retriever = _create_retriever(
            bm25,
            embed,
        )

        _state.update(
            {
                "bm25": bm25,
                "embed": embed,
                "store": store,
                "retriever": retriever,
            }
        )

        elapsed = time.perf_counter() - start_time

        return jsonify(
            {
                "status": "ok",
                "chunks_indexed": len(chunks),
                "documents_directory": docs_dir,
                "embedding_model": (
                    config.EMBEDDING_MODEL_NAME
                ),
                "fusion_method": "weighted",
                "elapsed_seconds": round(
                    elapsed,
                    3,
                ),
            }
        )

    except Exception as e:
        app.logger.error(
            traceback.format_exc()
        )

        return jsonify(
            {"error": str(e)}
        ), 500


@app.route("/query", methods=["POST"])
def query():
    """Retrieve context and optionally generate a grounded answer."""

    request_start = time.perf_counter()

    try:
        if not request.is_json:
            return jsonify(
                {
                    "error": "Request body must be JSON"
                }
            ), 400

        body = request.get_json(
            silent=True
        )

        if not isinstance(body, dict):
            return jsonify(
                {
                    "error": "Request body must be a JSON object"
                }
            ), 400

        question = body.get("question")

        if (
            not isinstance(question, str)
            or not question.strip()
        ):
            return jsonify(
                {
                    "error": "'question' is required"
                }
            ), 400

        top_k = _safe_int(
            body.get(
                "top_k",
                config.TOP_K,
            ),
            config.TOP_K,
            minimum=1,
        )

        candidate_pool = _safe_int(
            body.get(
                "candidate_pool",
                max(config.TOP_K, 50),
            ),
            max(config.TOP_K, 50),
            minimum=1,
        )

        if top_k > candidate_pool:
            candidate_pool = top_k

        min_score = _safe_float_or_none(
            body.get("min_score")
        )

        generate = bool(
            body.get(
                "generate",
                False,
            )
        )

        _ensure_loaded()

        retrieval_start = time.perf_counter()

        diagnostics = (
            _state["retriever"]
            .search_with_diagnostics(
                question,
                top_k=top_k,
                candidate_pool=candidate_pool,
                min_score=min_score,
            )
        )

        retrieval_time = (
            time.perf_counter()
            - retrieval_start
        )

        store = _state["store"]

        hits = [
            _serialize_hit(
                result,
                store,
            )
            for result in diagnostics["results"]
        ]

        response = {
            "question": question,
            "results": hits,
            "retrieval": {
                "fusion_method": (
                    diagnostics["fusion_method"]
                ),
                "alpha": diagnostics["alpha"],
                "rrf_k": diagnostics["rrf_k"],
                "candidate_pool": (
                    diagnostics["candidate_pool"]
                ),
                "top_k": diagnostics["requested_top_k"],
                "min_score": (
                    diagnostics["min_score"]
                ),
                "bm25_candidates": (
                    diagnostics["bm25_candidates"]
                ),
                "dense_candidates": (
                    diagnostics["dense_candidates"]
                ),
                "fused_candidates": (
                    diagnostics["fused_candidates"]
                ),
                "results_after_threshold": (
                    diagnostics[
                        "results_after_threshold"
                    ]
                ),
                "final_results": (
                    diagnostics["final_results"]
                ),
                "retrieval_time_seconds": round(
                    retrieval_time,
                    4,
                ),
            },
        }

        if generate:
            generation_start = time.perf_counter()

            # Only send useful retrieved context to the LLM.
            contexts = [
                {
                    "text": hit["text"],
                    "source": hit["source"],
                    "metadata": hit["metadata"],
                }
                for hit in hits
                if hit.get("text")
            ]

            answer = generate_answer(
                question,
                contexts,
            )

            generation_time = (
                time.perf_counter()
                - generation_start
            )

            response["answer"] = answer
            response["generation"] = {
                "provider": config.LLM_PROVIDER,
                "model": config.LLM_MODEL,
                "context_count": len(contexts),
                "generation_time_seconds": round(
                    generation_time,
                    4,
                ),
            }

        response["total_time_seconds"] = round(
            time.perf_counter()
            - request_start,
            4,
        )

        return jsonify(response)

    except ValueError as e:
        return jsonify(
            {"error": str(e)}
        ), 400

    except RuntimeError as e:
        return jsonify(
            {"error": str(e)}
        ), 400

    except Exception as e:
        app.logger.error(
            traceback.format_exc()
        )

        return jsonify(
            {"error": str(e)}
        ), 500


if __name__ == "__main__":
    port = int(
        os.environ.get(
            "PORT",
            5000,
        )
    )

    debug = (
        os.environ.get(
            "FLASK_DEBUG",
            "0",
        )
        == "1"
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=debug,
    )