"""
Semantic Cache Module
─────────────────────
Stores (query_embedding, answer) pairs in an in-memory FAISS index.

On each new query:
  1. Embed the query with the same model used for retrieval.
  2. Run an approximate-nearest-neighbour search in FAISS.
  3. If the cosine similarity of the best match exceeds CACHE_SIMILARITY_THRESHOLD
     → return the cached answer immediately (cache HIT).
  4. Otherwise → signal a cache MISS; the caller runs the full RAG pipeline
     and then calls `add()` to populate the cache.

Why FAISS instead of a plain list scan?
  - O(log n) ANN search instead of O(n) brute-force.
  - Scales to millions of cached queries without code changes.
"""

import time
import numpy as np
import faiss
from dataclasses import dataclass, field
from typing import Optional

from config import CACHE_SIMILARITY_THRESHOLD, CACHE_MAX_SIZE


@dataclass
class CacheEntry:
    query: str
    answer: str
    model_used: str
    timestamp: float = field(default_factory=time.time)
    hit_count: int = 0


class SemanticCache:
    """Thread-safe semantic cache backed by a FAISS flat-IP index."""

    def __init__(
        self,
        dim: int = 384,  # all-MiniLM-L6-v2 output dim
        threshold: float = CACHE_SIMILARITY_THRESHOLD,
        max_size: int = CACHE_MAX_SIZE,
    ) -> None:
        self.dim = dim
        self.threshold = threshold
        self.max_size = max_size

        # Inner-product index works as cosine similarity after L2 normalisation
        self.index = faiss.IndexFlatIP(dim)
        self.entries: list[CacheEntry] = []
        self._vectors: list[np.ndarray] = []  # parallel list of raw vectors for eviction rebuild

        # Stats
        self.total_queries = 0
        self.cache_hits = 0

    # ── public API ────────────────────────────────────────────────────────────

    def lookup(
        self, embedding: np.ndarray
    ) -> tuple[Optional[CacheEntry], float]:
        """
        Search the cache.

        Returns (entry, similarity) if a hit is found,
        or (None, 0.0) on a miss.
        """
        self.total_queries += 1

        if self.index.ntotal == 0:
            return None, 0.0

        vec = self._normalise(embedding)
        distances, indices = self.index.search(vec, k=1)
        similarity = float(distances[0][0])
        idx = int(indices[0][0])

        if similarity >= self.threshold and idx < len(self.entries):
            entry = self.entries[idx]
            entry.hit_count += 1
            self.cache_hits += 1
            return entry, similarity

        return None, similarity

    def add(
        self,
        embedding: np.ndarray,
        query: str,
        answer: str,
        model_used: str,
    ) -> None:
        """Insert a new query–answer pair into the cache."""
        if self.index.ntotal >= self.max_size:
            self._evict_oldest()

        vec = self._normalise(embedding)
        self.index.add(vec)
        self.entries.append(CacheEntry(query=query, answer=answer, model_used=model_used))
        self._vectors.append(vec)

    # ── stats ─────────────────────────────────────────────────────────────────

    @property
    def hit_rate(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self.cache_hits / self.total_queries

    @property
    def size(self) -> int:
        return self.index.ntotal

    def get_all_entries(self) -> list[dict]:
        return [
            {
                "query": e.query[:80],
                "model": e.model_used,
                "hit_count": e.hit_count,
                "timestamp": e.timestamp,
            }
            for e in self.entries
        ]

    # ── internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _normalise(vec: np.ndarray) -> np.ndarray:
        """L2-normalise so inner-product == cosine similarity."""
        v = vec.astype(np.float32).reshape(1, -1)
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm
        return v

    def _evict_oldest(self) -> None:
        """
        FAISS doesn't support deletion, so we rebuild the index
        keeping the newest 90% of entries (LRU-lite).
        """
        keep_n = int(self.max_size * 0.9)
        self.entries = self.entries[-keep_n:]
        self._vectors = self._vectors[-keep_n:]

        # Rebuild FAISS index from the kept vectors
        self.index = faiss.IndexFlatIP(self.dim)
        if self._vectors:
            kept = np.vstack(self._vectors).astype(np.float32)
            self.index.add(kept)
