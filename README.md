# RAG Platform with Hybrid Search & API Deployment

A retrieval-augmented generation (RAG) system that answers questions from your own documents.
It combines keyword search (BM25) with semantic search (sentence-transformer embeddings), reranks
the results, and generates grounded answers with source citations through a Flask REST API and a
React web interface.

## Features

- **Hybrid retrieval**: BM25 (sparse) + dense embeddings, fused with Reciprocal Rank Fusion (RRF)
- **Query-aware reranking** and content-aware filtering of retrieved passages
- **Multi-format ingestion**: `.txt`, `.pdf`, `.docx`, `.pptx` with page / slide metadata
- **Grounded answers** from an LLM (Gemini, Anthropic or OpenAI) with cited sources
- **Flask REST API** with request IDs, input validation, structured errors and CORS
- **React (Vite) frontend**: ask questions, see the answer, sources and retrieved passages
- **One-click launcher** for Windows (`start_rag.bat`)

## Architecture

```
documents (txt / pdf / docx / pptx)
        |  loaders + chunking
        v
   Chunk objects  ---------------------------+
        |                                    |
   BM25 index                        Embedding index
  (keyword search)                  (semantic search)
        |                                    |
        +----------->  Hybrid retriever (RRF fusion)  <----+
                               |
                          Reranker
                               |
                        Flask API (app.py)
                     /query  /ingest  /health
                               |
                    React frontend (localhost:5173)
```

Project layout:

```
app.py            Flask API
ingest.py         builds the indexes from sample_docs/
src/              retrieval pipeline (loaders, chunking, BM25, embeddings, hybrid, reranker, LLM client)
frontend/         React + Vite web interface
sample_docs/      put your documents here
storage/          generated indexes (not committed)
tests/            pipeline tests
start_rag.bat     Windows one-click launcher
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cd frontend
npm install
cd ..
```

## Add your documents

Put `.txt`, `.pdf`, `.docx` or `.pptx` files in `sample_docs/`, then build the indexes:

```powershell
python ingest.py
```

The embedding model downloads from Hugging Face the first time, so internet access is needed once.
Re-run this command whenever you add or change documents.

## Configure the LLM (for generated answers)

Set these environment variables (see `.env.example`):

```powershell
$env:LLM_PROVIDER = "gemini"            # gemini, anthropic or openai
$env:LLM_MODEL = "<a model your key supports>"
$env:GEMINI_API_KEY = "<your key>"      # or ANTHROPIC_API_KEY / OPENAI_API_KEY
```

Without a provider, the API still returns the retrieved passages, just no generated answer.
Never commit API keys.

## Run

Backend (terminal 1):

```powershell
python app.py
```

Frontend (terminal 2):

```powershell
cd frontend
npm run dev
```

Open http://localhost:5173. On Windows you can instead double-click `start_rag.bat`, which starts
both and opens the browser.

## API

| Method | Route     | Purpose                                              |
|--------|-----------|------------------------------------------------------|
| GET    | `/health` | Liveness check                                       |
| GET    | `/ready`  | Readiness (indexes loaded)                           |
| GET    | `/status` | Index and configuration details                      |
| POST   | `/ingest` | Rebuild the indexes without restarting               |
| POST   | `/query`  | Retrieve passages and optionally generate an answer  |

Example:

```powershell
curl.exe -X POST http://127.0.0.1:5000/query -H "Content-Type: application/json" -d "{\"question\":\"What is BM25?\",\"top_k\":5,\"generate\":true}"
```

The response includes `answer`, `results` (passages with scores), `sources` (file plus page or
slide), and `retrieval` / timing diagnostics. Set `"generate": true` to get an answer; without it,
`answer` is `null`.

## Tests

```powershell
python tests/test_pipeline.py
```

## Tech stack

Python, Flask, BM25 (rank_bm25), sentence-transformers, PyTorch, Google Gemini API, React, Vite.