"""
Central configuration for the RAG Platform.
All tunable parameters live here so nothing is hard-coded deep inside the pipeline.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(BASE_DIR, "sample_docs")
INDEX_DIR = os.path.join(BASE_DIR, "storage")
BM25_INDEX_PATH = os.path.join(INDEX_DIR, "bm25_index.pkl")
EMBED_INDEX_PATH = os.path.join(INDEX_DIR, "embed_index.pkl")
DOCSTORE_PATH = os.path.join(INDEX_DIR, "docstore.pkl")

# --- Chunking ---
CHUNK_SIZE = 300        # words per chunk
CHUNK_OVERLAP = 50      # words of overlap between consecutive chunks

# --- Embedding model ---
# Any sentence-transformers model works. all-MiniLM-L6-v2 is small and fast.
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# --- Hybrid search weighting ---
# final_score = ALPHA * normalized_dense_score + (1 - ALPHA) * normalized_bm25_score
HYBRID_ALPHA = float(os.environ.get("HYBRID_ALPHA", "0.5"))
TOP_K = int(os.environ.get("TOP_K", "5"))

# --- LLM generation (optional, used only by /query when generate=true) ---
# Supports "anthropic", "openai", or "none" (extractive-only, no LLM call).
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "none")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "claude-sonnet-4-6")

os.makedirs(INDEX_DIR, exist_ok=True)
