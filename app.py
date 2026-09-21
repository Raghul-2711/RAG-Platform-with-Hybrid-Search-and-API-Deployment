"""
Enterprise RAG Platform - Production API.

Endpoints
---------
GET  /health
GET  /ready
GET  /status
POST /ingest
POST /query

The API provides:
- deterministic request IDs
- structured errors
- request validation
- liveness/readiness checks
- ingestion status
- hybrid retrieval
- RRF retrieval
- content-aware bibliography filtering
- query-aware reranking
- retrieval diagnostics
- grounded LLM generation
- clean source citations
- CORS support for the frontend
"""

import os
import re
import time
import traceback
import uuid
from pathlib import Path

from flask import Flask, jsonify, request

from src import config
from src.bm25_index import BM25Index
from src.embedding_index import EmbeddingIndex, default_encode_fn
from src.docstore import DocStore
from src.hybrid_retriever import HybridRetriever
from src.chunking import build_chunks_from_dir
from src.llm_client import generate_answer
from src.reranker import rerank


app = Flask(__name__)


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

_state = {
    "bm25": None,
    "embed": None,
    "store": None,
    "retriever": None,
}

_ingestion_state = {
    "status": "idle",
    "started_at": None,
    "completed_at": None,
    "documents_directory": None,
    "chunks_indexed": 0,
    "elapsed_seconds": None,
    "error": None,
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_TOP_K = 20
MAX_CANDIDATE_POOL = 200
MAX_QUESTION_LENGTH = 4000
MAX_DOCS_DIR_LENGTH = 1000

RRF_K = 60

CONTENT_AWARE_EXPANSION = 3

BIBLIOGRAPHY_QUERY_TERMS = {
    "reference",
    "references",
    "bibliography",
    "citation",
    "citations",
    "cited",
    "cite",
    "author",
    "authors",
    "paper",
    "papers",
    "arxiv",
    "doi",
    "publication",
    "publications",
}

FRONTEND_ORIGIN = os.environ.get(
    "FRONTEND_ORIGIN",
    "http://localhost:5173",
)


# ---------------------------------------------------------------------------
# Request ID / CORS
# ---------------------------------------------------------------------------

@app.before_request
def assign_request_id():
    """Assign a request ID to every request."""

    incoming = request.headers.get("X-Request-ID", "").strip()

    if incoming and len(incoming) <= 100:
        request.request_id = incoming
    else:
        request.request_id = uuid.uuid4().hex


@app.after_request
def add_response_headers(response):
    """Add request ID and frontend CORS headers."""

    response.headers["X-Request-ID"] = getattr(
        request,
        "request_id",
        uuid.uuid4().hex,
    )

    response.headers["Access-Control-Allow-Origin"] = FRONTEND_ORIGIN
    response.headers["Access-Control-Allow-Headers"] = (
        "Content-Type, Authorization, X-Request-ID"
    )
    response.headers["Access-Control-Allow-Methods"] = (
        "GET, POST, OPTIONS"
    )

    return response


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def _error_response(
    code,
    message,
    status_code,
    details=None,
):
    """Return the standard API error schema."""

    payload = {
        "error": {
            "code": code,
            "message": message,
        },
        "request_id": getattr(
            request,
            "request_id",
            None,
        ),
    }

    if details is not None:
        payload["error"]["details"] = details

    return jsonify(payload), status_code


@app.errorhandler(404)
def handle_not_found(error):
    return _error_response(
        "NOT_FOUND",
        "The requested endpoint does not exist.",
        404,
    )


@app.errorhandler(405)
def handle_method_not_allowed(error):
    return _error_response(
        "METHOD_NOT_ALLOWED",
        "The HTTP method is not allowed for this endpoint.",
        405,
    )


@app.errorhandler(413)
def handle_request_too_large(error):
    return _error_response(
        "REQUEST_TOO_LARGE",
        "The request payload is too large.",
        413,
    )


@app.errorhandler(Exception)
def handle_unexpected_error(error):
    """Final safety net for unexpected application errors."""

    app.logger.error(
        "Unhandled exception request_id=%s\n%s",
        getattr(request, "request_id", None),
        traceback.format_exc(),
    )

    return _error_response(
        "INTERNAL_SERVER_ERROR",
        "An unexpected server error occurred.",
        500,
    )


# ---------------------------------------------------------------------------
# Index / state helpers
# ---------------------------------------------------------------------------

def _indexes_exist():
    """Return True when all persisted indexes exist."""

    return (
        os.path.exists(config.BM25_INDEX_PATH)
        and os.path.exists(config.EMBED_INDEX_PATH)
        and os.path.exists(config.DOCSTORE_PATH)
    )


def _indexes_loaded():
    """Return True when all runtime components are loaded."""

    return all(
        _state[key] is not None
        for key in (
            "bm25",
            "embed",
            "store",
            "retriever",
        )
    )


def _create_retriever(bm25, embed):
    """Create the production hybrid retriever using RRF."""

    return HybridRetriever(
        bm25_index=bm25,
        embedding_index=embed,
        alpha=config.HYBRID_ALPHA,
        fusion_method="rrf",
        rrf_k=RRF_K,
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

    if _indexes_loaded():
        return

    if not _indexes_exist():
        raise RuntimeError(
            "No indexes found. Run `python ingest.py` first "
            "or POST /ingest."
        )

    _load_state()


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _parse_json_body():
    """Parse a JSON object from the request."""

    if not request.is_json:
        raise ValueError(
            "Request body must use Content-Type: application/json."
        )

    body = request.get_json(silent=True)

    if not isinstance(body, dict):
        raise ValueError(
            "Request body must be a JSON object."
        )

    return body


def _parse_positive_int(
    body,
    field,
    default,
    maximum,
):
    """Parse a strictly typed bounded positive integer."""

    if field not in body:
        return default

    value = body[field]

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"'{field}' must be an integer."
        )

    if value < 1:
        raise ValueError(
            f"'{field}' must be >= 1."
        )

    if value > maximum:
        raise ValueError(
            f"'{field}' must be <= {maximum}."
        )

    return value


def _parse_min_score(body):
    """Parse an optional retrieval threshold."""

    if "min_score" not in body:
        return None

    value = body["min_score"]

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise ValueError(
            "'min_score' must be a number."
        )

    if value < 0.0 or value > 1.0:
        raise ValueError(
            "'min_score' must be between 0.0 and 1.0."
        )

    return float(value)


def _parse_generate(body):
    """Parse the generation flag strictly."""

    if "generate" not in body:
        return False

    value = body["generate"]

    if not isinstance(value, bool):
        raise ValueError(
            "'generate' must be a boolean."
        )

    return value


def _validate_question(body):
    """Validate the user question."""

    question = body.get("question")

    if not isinstance(question, str):
        raise ValueError(
            "'question' is required and must be a string."
        )

    question = question.strip()

    if not question:
        raise ValueError(
            "'question' must not be empty."
        )

    if len(question) > MAX_QUESTION_LENGTH:
        raise ValueError(
            f"'question' must be <= {MAX_QUESTION_LENGTH} characters."
        )

    return question


def _validate_docs_dir(body):
    """Validate an ingestion directory."""

    docs_dir = body.get(
        "docs_dir",
        config.DOCS_DIR,
    )

    if not isinstance(docs_dir, str):
        raise ValueError(
            "'docs_dir' must be a string."
        )

    docs_dir = docs_dir.strip()

    if not docs_dir:
        raise ValueError(
            "'docs_dir' must not be empty."
        )

    if len(docs_dir) > MAX_DOCS_DIR_LENGTH:
        raise ValueError(
            f"'docs_dir' must be <= {MAX_DOCS_DIR_LENGTH} characters."
        )

    path = Path(docs_dir).expanduser().resolve()
    base = Path(config.BASE_DIR).resolve()

    try:
        path.relative_to(base)
    except ValueError:
        raise ValueError(
            "'docs_dir' must be inside the project directory."
        )

    if not path.is_dir():
        raise ValueError(
            f"Document directory does not exist: {path}"
        )

    return str(path)


# ---------------------------------------------------------------------------
# Content-aware retrieval
# ---------------------------------------------------------------------------

def _is_reference_query(question):
    """Return True when the query explicitly asks about references."""

    tokens = set(
        re.findall(
            r"[a-z0-9]+",
            question.lower(),
        )
    )

    return bool(
        tokens.intersection(BIBLIOGRAPHY_QUERY_TERMS)
    )


def _select_content_aware_results(
    results,
    store,
    top_k,
    question,
):
    """
    Suppress bibliography chunks for normal questions.

    Bibliography remains fully eligible for reference-oriented queries.

    Each result is enriched with its DocStore metadata so downstream
    reranking can make content-aware decisions.
    """

    reference_query = _is_reference_query(question)

    bibliography_results = []
    content_results = []

    for result in results:
        chunk = store.get(
            result["chunk_id"]
        )

        metadata = {}

        if chunk is not None:
            metadata = dict(
                chunk.metadata or {}
            )

        enriched_result = {
            **result,
            "metadata": metadata,
        }

        if metadata.get("content_type") == "bibliography":
            bibliography_results.append(
                enriched_result
            )
        else:
            content_results.append(
                enriched_result
            )

    if reference_query:
        selected = (
            [
                {
                    **result,
                    "metadata": (
                        dict(
                            store.get(
                                result["chunk_id"]
                            ).metadata or {}
                        )
                        if store.get(
                            result["chunk_id"]
                        ) is not None
                        else {}
                    ),
                }
                for result in results
            ]
        )[:top_k]
    else:
        selected = (
            content_results + bibliography_results
        )[:top_k]

    return selected, {
        "enabled": True,
        "reference_query": reference_query,
        "raw_candidates": len(results),
        "bibliography_candidates": len(
            bibliography_results
        ),
        "bibliography_suppressed": (
            0
            if reference_query
            else max(
                0,
                len(bibliography_results)
                - max(
                    0,
                    top_k - len(content_results),
                ),
            )
        ),
        "selected_results": len(selected),
    }


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

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
            "rerank_score": round(
                result.get("rerank_score", 0.0),
                6,
            ),
            "query_term_overlap": round(
                result.get("query_term_overlap", 0.0),
                6,
            ),
            "phrase_overlap": round(
                result.get("phrase_overlap", 0.0),
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
        "rerank_score": round(
            result.get("rerank_score", 0.0),
            6,
        ),
        "query_term_overlap": round(
            result.get("query_term_overlap", 0.0),
            6,
        ),
        "phrase_overlap": round(
            result.get("phrase_overlap", 0.0),
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


def _build_sources(hits):
    """Build a frontend-friendly citation list."""

    sources = []
    seen = set()

    for rank, hit in enumerate(hits, start=1):
        source = hit.get("source")

        if not source:
            continue

        metadata = hit.get("metadata") or {}

        page = metadata.get("page_number")
        slide = metadata.get("slide_number")

        if page is not None:
            location = f"Page {page}"
        elif slide is not None:
            location = f"Slide {slide}"
        else:
            location = None

        key = (
            source,
            page,
            slide,
        )

        if key in seen:
            continue

        seen.add(key)

        citation = {
            "rank": rank,
            "source": source,
            "location": location,
            "score": hit.get(
                "rerank_score",
                hit.get("score"),
            ),
            "chunk_id": hit.get("chunk_id"),
        }

        if page is not None:
            citation["page"] = page

        if slide is not None:
            citation["slide"] = slide

        sources.append(citation)

    return sources


# ---------------------------------------------------------------------------
# Health / readiness / status
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    """Liveness endpoint."""

    return jsonify(
        {
            "status": "ok",
            "service": "enterprise-rag-api",
            "request_id": request.request_id,
        }
    ), 200


@app.route("/ready", methods=["GET"])
def ready():
    """Readiness endpoint."""

    try:
        _ensure_loaded()

        return jsonify(
            {
                "status": "ready",
                "service": "enterprise-rag-api",
                "indexes_loaded": True,
                "indexes_exist": _indexes_exist(),
                "document_count": len(
                    _state["store"].chunks
                ),
                "embedding_model": (
                    config.EMBEDDING_MODEL_NAME
                ),
                "llm_provider": config.LLM_PROVIDER,
                "request_id": request.request_id,
            }
        ), 200

    except Exception as exc:
        app.logger.error(
            "Readiness check failed request_id=%s error=%s",
            request.request_id,
            exc,
        )

        return _error_response(
            "SERVICE_NOT_READY",
            "The RAG service is not ready.",
            503,
            {
                "indexes_exist": _indexes_exist(),
                "indexes_loaded": _indexes_loaded(),
            },
        )


@app.route("/status", methods=["GET"])
def status():
    """Return detailed runtime and ingestion status."""

    return jsonify(
        {
            "status": "ok",
            "service": "enterprise-rag-api",
            "runtime": {
                "indexes_exist": _indexes_exist(),
                "indexes_loaded": _indexes_loaded(),
                "documents_loaded": (
                    len(_state["store"].chunks)
                    if _state["store"] is not None
                    else 0
                ),
                "embedding_model": (
                    config.EMBEDDING_MODEL_NAME
                ),
                "fusion_method": "rrf",
                "hybrid_alpha": config.HYBRID_ALPHA,
                "rrf_k": RRF_K,
                "top_k_default": config.TOP_K,
                "llm_provider": config.LLM_PROVIDER,
                "llm_model": config.LLM_MODEL,
                "content_aware_filtering": {
                    "enabled": True,
                    "bibliography_suppression": True,
                    "reference_query_detection": True,
                },
                "reranking": {
                    "enabled": True,
                    "method": "query-aware-deterministic",
                },
            },
            "ingestion": dict(_ingestion_state),
            "request_id": request.request_id,
        }
    ), 200


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

@app.route("/ingest", methods=["POST"])
def ingest():
    """Build BM25, dense embeddings, and document-store indexes."""

    start_time = time.perf_counter()

    try:
        body = _parse_json_body()
        docs_dir = _validate_docs_dir(body)

        if _ingestion_state["status"] == "running":
            return _error_response(
                "INGESTION_IN_PROGRESS",
                "An ingestion operation is already running.",
                409,
            )

        _ingestion_state.update(
            {
                "status": "running",
                "started_at": time.time(),
                "completed_at": None,
                "documents_directory": docs_dir,
                "chunks_indexed": 0,
                "elapsed_seconds": None,
                "error": None,
            }
        )

        chunks = build_chunks_from_dir(
            docs_dir,
            config.CHUNK_SIZE,
            config.CHUNK_OVERLAP,
        )

        if not chunks:
            raise ValueError(
                "No supported documents found. "
                "Supported formats: TXT, MD, PDF, DOCX, PPTX."
            )

        chunk_ids = [
            chunk.id
            for chunk in chunks
        ]

        texts = [
            chunk.text
            for chunk in chunks
        ]

        if len(chunk_ids) != len(set(chunk_ids)):
            raise RuntimeError(
                "Duplicate chunk IDs detected during ingestion."
            )

        bm25 = BM25Index()

        bm25.build(
            chunk_ids,
            texts,
        )

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

        store = DocStore()

        store.add(chunks)

        bm25.save(
            config.BM25_INDEX_PATH
        )

        embed.save(
            config.EMBED_INDEX_PATH
        )

        store.save(
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

        elapsed = (
            time.perf_counter()
            - start_time
        )

        _ingestion_state.update(
            {
                "status": "completed",
                "completed_at": time.time(),
                "chunks_indexed": len(chunks),
                "elapsed_seconds": round(
                    elapsed,
                    3,
                ),
                "error": None,
            }
        )

        return jsonify(
            {
                "status": "ok",
                "request_id": request.request_id,
                "chunks_indexed": len(chunks),
                "documents_indexed": len(
                    set(
                        chunk.source
                        for chunk in chunks
                    )
                ),
                "documents_directory": docs_dir,
                "embedding_model": (
                    config.EMBEDDING_MODEL_NAME
                ),
                "fusion_method": "rrf",
                "rrf_k": RRF_K,
                "content_aware_filtering": True,
                "reranking": {
                    "enabled": True,
                    "method": "query-aware-deterministic",
                },
                "elapsed_seconds": round(
                    elapsed,
                    3,
                ),
            }
        ), 200

    except ValueError as exc:
        _ingestion_state.update(
            {
                "status": "failed",
                "completed_at": time.time(),
                "error": str(exc),
            }
        )

        return _error_response(
            "INVALID_INGESTION_REQUEST",
            str(exc),
            400,
        )

    except Exception as exc:
        _ingestion_state.update(
            {
                "status": "failed",
                "completed_at": time.time(),
                "error": str(exc),
            }
        )

        app.logger.error(
            "Ingestion failed request_id=%s\n%s",
            request.request_id,
            traceback.format_exc(),
        )

        return _error_response(
            "INGESTION_FAILED",
            "Document ingestion failed.",
            500,
        )


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

@app.route("/query", methods=["POST"])
def query():
    """Retrieve context and optionally generate a grounded answer."""

    request_start = time.perf_counter()

    try:
        body = _parse_json_body()

        question = _validate_question(body)

        top_k = _parse_positive_int(
            body,
            "top_k",
            config.TOP_K,
            MAX_TOP_K,
        )

        candidate_pool = _parse_positive_int(
            body,
            "candidate_pool",
            max(config.TOP_K, 50),
            MAX_CANDIDATE_POOL,
        )

        if top_k > candidate_pool:
            raise ValueError(
                "'top_k' cannot be greater than 'candidate_pool'."
            )

        min_score = _parse_min_score(body)
        generate = _parse_generate(body)

        _ensure_loaded()

        retrieval_top_k = min(
            candidate_pool,
            max(
                top_k * CONTENT_AWARE_EXPANSION,
                top_k,
            ),
        )

        retrieval_start = time.perf_counter()

        diagnostics = _state[
            "retriever"
        ].search_with_diagnostics(
            question,
            top_k=retrieval_top_k,
            candidate_pool=candidate_pool,
            min_score=min_score,
        )

        retrieval_time = (
            time.perf_counter()
            - retrieval_start
        )

        store = _state["store"]

        selected_results, content_filter = (
            _select_content_aware_results(
                diagnostics["results"],
                store,
                retrieval_top_k,
                question,
            )
        )

        reranked_results = rerank(
            question,
            selected_results,
            lambda chunk_id: (
                store.get(chunk_id).text
                if store.get(chunk_id) is not None
                else None
            ),
        )

        selected_results = (
            reranked_results[:top_k]
        )

        hits = [
            _serialize_hit(
                result,
                store,
            )
            for result in selected_results
        ]

        response = {
            "status": "ok",
            "request_id": request.request_id,
            "question": question,
            "answer": None,
            "generation": {
                "requested": generate,
                "provider": (
                    config.LLM_PROVIDER
                    if generate
                    else None
                ),
                "model": (
                    config.LLM_MODEL
                    if generate
                    else None
                ),
                "context_count": 0,
                "generation_time_seconds": 0.0,
            },
            "results": hits,
            "sources": _build_sources(hits),
            "retrieval": {
                "fusion_method": diagnostics[
                    "fusion_method"
                ],
                "alpha": diagnostics["alpha"],
                "rrf_k": diagnostics["rrf_k"],
                "candidate_pool": diagnostics[
                    "candidate_pool"
                ],
                "top_k": top_k,
                "raw_top_k": retrieval_top_k,
                "min_score": diagnostics[
                    "min_score"
                ],
                "bm25_candidates": diagnostics[
                    "bm25_candidates"
                ],
                "dense_candidates": diagnostics[
                    "dense_candidates"
                ],
                "fused_candidates": diagnostics[
                    "fused_candidates"
                ],
                "raw_results_after_threshold": (
                    diagnostics[
                        "results_after_threshold"
                    ]
                ),
                "raw_final_results": diagnostics[
                    "final_results"
                ],
                "content_aware_filtering": (
                    content_filter
                ),
                "reranking": {
                    "enabled": True,
                    "method": (
                        "query-aware-deterministic"
                    ),
                    "candidate_count": len(
                        reranked_results
                    ),
                },
                "final_results": len(hits),
                "retrieval_time_seconds": round(
                    retrieval_time,
                    4,
                ),
            },
        }

        if generate:
            generation_start = (
                time.perf_counter()
            )

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

            response["generation"].update(
                {
                    "context_count": len(
                        contexts
                    ),
                    "generation_time_seconds": round(
                        generation_time,
                        4,
                    ),
                }
            )

        response["total_time_seconds"] = round(
            time.perf_counter()
            - request_start,
            4,
        )

        return jsonify(response), 200

    except ValueError as exc:
        return _error_response(
            "INVALID_QUERY_REQUEST",
            str(exc),
            400,
        )

    except RuntimeError as exc:
        return _error_response(
            "SERVICE_NOT_READY",
            str(exc),
            503,
        )

    except Exception as exc:
        app.logger.error(
            "Query failed request_id=%s\n%s",
            request.request_id,
            traceback.format_exc(),
        )

        return _error_response(
            "QUERY_FAILED",
            "The query could not be completed.",
            500,
        )


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

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