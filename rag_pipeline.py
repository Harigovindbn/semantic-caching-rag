"""
RAG Pipeline
─────────────
Handles:
  1. Document ingestion  — PDF / plain-text → chunks → ChromaDB
  2. Retrieval           — embed query → top-k similar chunks
  3. Generation          — build prompt + call modern Gemini API (google-genai)
"""

import time
import os
import re
from pathlib import Path
from typing import Generator

from google import genai
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np

from config import (
    GEMINI_API_KEY,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL,
    TOP_K_RETRIEVAL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    FLASH_MODEL,
    PRO_MODEL,
)


# ── Prompt template ───────────────────────────────────────────────────────────
RAG_PROMPT = """You are an expert AI/ML research assistant. Answer the question below
using ONLY the provided context. If the context does not contain enough information,
say so honestly rather than hallucinating.

Context:
{context}

Question: {question}

Answer (be concise but precise):"""


class RAGPipeline:
    """Full RAG pipeline: embed → retrieve → generate."""

    def __init__(self) -> None:
        # Initialize the modern Google GenAI client
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set in .env")
        
        self.client = genai.Client(api_key=GEMINI_API_KEY)

        # Embedding model (shared with cache & router)
        print("⏳ Loading embedding model …")
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"✅ Embedding model ready: {EMBEDDING_MODEL}")

        # ChromaDB vector store
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.collection = self.chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"✅ ChromaDB ready — {self.collection.count()} chunks indexed")

    # ── Embedding ──────────────────────────────────────────────────────────────

    def embed(self, text: str) -> np.ndarray:
        """Return a normalised embedding vector for text."""
        return self.embed_model.encode(text, normalize_embeddings=True)

    # ── Ingestion ──────────────────────────────────────────────────────────────

    def ingest_text(self, text: str, doc_id: str, metadata: dict | None = None) -> int:
        """Chunk raw text and upsert into ChromaDB. Returns chunk count."""
        chunks = list(self._chunk(text))
        if not chunks:
            return 0

        embeddings = self.embed_model.encode(chunks, normalize_embeddings=True).tolist()
        ids        = [f"{doc_id}::chunk_{i}" for i in range(len(chunks))]
        metas      = [{"source": doc_id, **(metadata or {})} for _ in chunks]

        self.collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metas,
        )
        return len(chunks)

    def ingest_file(self, filepath: str | Path) -> int:
        """Ingest a .txt or .pdf file into ChromaDB."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(filepath)

        if filepath.suffix.lower() == ".pdf":
            text = self._read_pdf(filepath)
        else:
            text = filepath.read_text(encoding="utf-8", errors="ignore")

        doc_id = filepath.stem
        n = self.ingest_text(text, doc_id, {"filename": filepath.name})
        print(f"✅ Ingested '{filepath.name}' → {n} chunks")
        return n

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def retrieve(self, query_embedding: np.ndarray, top_k: int = TOP_K_RETRIEVAL) -> list[str]:
        """Return top-k relevant text chunks for a query embedding."""
        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, self.collection.count()),
            include=["documents"],
        )
        docs = results.get("documents", [[]])[0]
        return [d for d in docs if d]

    # ── Generation ─────────────────────────────────────────────────────────────

    def generate(
        self,
        query: str,
        model_name: str,
        context_chunks: list[str],
    ) -> tuple[str, int, int]:
        """
        Call Gemini via client.models.generate_content and return (answer, input_tokens, output_tokens).
        """
        context = "\n\n---\n\n".join(context_chunks) if context_chunks else "No context available."
        prompt  = RAG_PROMPT.format(context=context, question=query)

        response = self.client.models.generate_content(
            model=model_name,
            contents=prompt,
        )

        answer     = response.text
        usage      = response.usage_metadata
        input_toks  = getattr(usage, "prompt_token_count",     0) or 0
        output_toks = getattr(usage, "candidates_token_count", 0) or 0

        return answer, input_toks, output_toks

    # ── Chunking ───────────────────────────────────────────────────────────────

    @staticmethod
    def _chunk(text: str) -> Generator[str, None, None]:
        """Sliding-window character chunker."""
        text = re.sub(r"\s+", " ", text).strip()
        start = 0
        while start < len(text):
            end   = start + CHUNK_SIZE
            chunk = text[start:end].strip()
            if chunk:
                yield chunk
            start += CHUNK_SIZE - CHUNK_OVERLAP

    @staticmethod
    def _read_pdf(path: Path) -> str:
        """Extract text from a PDF using pypdf."""
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except ImportError:
            raise ImportError("Install pypdf: pip install pypdf")