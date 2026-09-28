"""
Semantic Caching RAG — Project Configuration
Central place for all constants and settings.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ─────────────────────────────────────────────────────────────────
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

# ── Model Names ───────────────────────────────────────────────────────────────
FLASH_MODEL = "gemini-2.5-flash"
PRO_MODEL = "gemini-2.5-pro"

# ── Semantic Cache ────────────────────────────────────────────────────────────
CACHE_SIMILARITY_THRESHOLD: float = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.90"))
CACHE_MAX_SIZE: int               = int(os.getenv("CACHE_MAX_SIZE", "1000"))

# ── Complexity Router ─────────────────────────────────────────────────────────
COMPLEXITY_THRESHOLD: float = float(os.getenv("COMPLEXITY_THRESHOLD", "0.60"))

# ── Cost Table (USD per 1 000 tokens) ─────────────────────────────────────────
COST_TABLE: dict[str, dict[str, float]] = {
    FLASH_MODEL: {
        "input":  float(os.getenv("FLASH_INPUT_COST",  "0.000075")),
        "output": float(os.getenv("FLASH_OUTPUT_COST", "0.0003")),
    },
    PRO_MODEL: {
        "input":  float(os.getenv("PRO_INPUT_COST",  "0.00125")),
        "output": float(os.getenv("PRO_OUTPUT_COST", "0.005")),
    },
}

# ── ChromaDB ──────────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR: str     = "./data/chroma_db"
CHROMA_COLLECTION_NAME: str = "ml_papers"

# ── Embedding Model ───────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # local, fast, free

# ── RAG ───────────────────────────────────────────────────────────────────────
TOP_K_RETRIEVAL: int = 5
CHUNK_SIZE: int      = 800
CHUNK_OVERLAP: int   = 100

# ── Metrics ───────────────────────────────────────────────────────────────────
METRICS_FILE: str = "./data/metrics.jsonl"
