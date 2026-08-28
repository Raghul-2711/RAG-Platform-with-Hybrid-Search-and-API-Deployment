# RAG Platform with Hybrid Search & API Deployment

A clean, tested rebuild of the hybrid-search RAG project: BM25 (sparse) + sentence-transformer
embeddings (dense), fused into one ranking, served over a Flask REST API, with optional LLM
answer generation.

## Architecture

```
sample_docs/*.txt  --chunking-->  Chunk objects
                                       |
                       -------------------------------
                       |                             |
                  BM25Index                    EmbeddingIndex
               (rank_bm25, keyword)      (sentence-transformers, cosine sim)
                       |                             |
                       -------------------------------
                                       |
                              HybridRetriever
                     (min-max normalize + weighted fusion)
                                       |
                                  Flask app.py
                          /query  ->  top-k chunks (+ optional LLM answer)
```

Each piece is a separate, independently testable module under `src/` — that modularity is what
was missing in the original repo and is what caused most of its errors (tightly coupled globals,
no separation between index-building and serving).

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # optional — only needed for LLM generation
```

## 2. Add your documents

Drop `.txt` or `.md` files into `sample_docs/`. Three example files are included so you can test
immediately without adding anything.

## 3. Build the indexes

```bash
python ingest.py
```

This chunks every document (`CHUNK_SIZE=300` words, `CHUNK_OVERLAP=50`, configurable in
`src/config.py`), builds a BM25 index and a dense embedding index, and writes both plus the
docstore to `storage/`. The embedding model (`all-MiniLM-L6-v2` by default) downloads from
Hugging Face the first time you run this — it needs internet access once, then is cached locally.

## 4. Run the API

```bash
python app.py
```

Server starts on `http://localhost:5000`.

### `GET /health`
```bash
curl http://localhost:5000/health
```

### `POST /query`
```bash
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is BM25 used for?", "top_k": 3}'
```

Response:
```json
{
  "question": "What is BM25 used for?",
  "results": [
    {
      "chunk_id": "...",
      "source": "bm25_notes.txt",
      "text": "BM25 (Best Matching 25) is a ranking function ...",
      "score": 0.91,
      "bm25_score": 4.83,
      "dense_score": 0.77
    }
  ]
}
```

Add `"generate": true` to also get an LLM-written answer grounded in the retrieved chunks (set
`LLM_PROVIDER=anthropic` or `openai` and the matching API key in `.env` first — otherwise
retrieval-only results are returned).

### `POST /ingest`
Rebuilds the indexes without restarting the server — handy after adding new documents:
```bash
curl -X POST http://localhost:5000/ingest
```

## 5. Run the tests

```bash
python tests/test_pipeline.py
```

This is a full smoke test (chunking → BM25 → embeddings → hybrid fusion → save/load → Flask
endpoints) that uses a deterministic mock embedder, so it runs in seconds with **no model
download or API key required** — useful for CI.

## 6. Docker

```bash
docker build -t rag-platform .
docker run -p 5000:5000 rag-platform
```

The image builds the indexes at build time (`RUN python ingest.py`), so the container is ready to
serve as soon as it starts.

## Tuning hybrid search

`HYBRID_ALPHA` in `src/config.py` (or the `HYBRID_ALPHA` env var) controls the BM25/dense blend:

- `1.0` → pure dense/semantic search
- `0.0` → pure BM25/keyword search
- `0.5` (default) → balanced hybrid, a reasonable starting point

Raise it toward `1.0` if queries use different wording than your documents (semantic matching
matters more); lower it toward `0.0` if queries rely on exact keywords, IDs, or jargon that must
match literally.

## What was fixed vs. a typical broken hybrid-search repo

- **Circular / implicit imports** — every module (`bm25_index`, `embedding_index`,
  `hybrid_retriever`, `docstore`) is self-contained and only imports what it needs; no global
  state is created at import time.
- **Model reload on every request** — `app.py` loads indexes and the embedding model once into
  `_state`, not per-request.
- **Score fusion bugs** — raw BM25 scores and cosine similarities live on different scales; they
  are min-max normalized before being combined, so one score type can't silently dominate.
- **No error handling** — every endpoint validates its input and returns a proper 4xx with a
  message instead of a raw 500 stack trace.
- **Untestable pipeline** — `EmbeddingIndex` takes an injectable `encode_fn`, so the whole
  retrieval pipeline can be tested without downloading a transformer model or hitting an API.
