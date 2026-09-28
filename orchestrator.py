"""
Main Orchestrator
──────────────────
Wires together: SemanticCache + ComplexityRouter + RAGPipeline + MetricsTracker

Query flow:
  1. Embed query
  2. Check semantic cache
     ├─ HIT  → return cached answer, log cache metrics
     └─ MISS → route query to Flash or Pro
               → retrieve context from ChromaDB
               → generate answer
               → populate cache
               → log full metrics
"""

import time
from dataclasses import dataclass

import numpy as np

from config import FLASH_MODEL
from semantic_cache import SemanticCache
from complexity_router import ComplexityRouter, RoutingDecision
from rag_pipeline import RAGPipeline
from metrics_tracker import MetricsTracker, RequestMetrics


@dataclass
class QueryResult:
    answer: str
    cache_hit: bool
    model_used: str
    latency_ms: float
    cost_usd: float
    complexity_score: float
    similarity_score: float
    routing_decision: RoutingDecision | None
    input_tokens: int
    output_tokens: int


class RAGOrchestrator:
    """Single entry point that ties every layer together."""

    def __init__(self) -> None:
        print("🚀 Initialising RAG Orchestrator …")
        self.pipeline = RAGPipeline()
        self.cache    = SemanticCache(dim=384)
        self.router   = ComplexityRouter(embed_model=self.pipeline.embed_model)
        self.tracker  = MetricsTracker()
        print("✅ All components ready.\n")

    # ── Public API ─────────────────────────────────────────────────────────────

    def query(self, question: str) -> QueryResult:
        """Process a user question end-to-end."""
        t_start = time.perf_counter()

        # 1. Embed query once (reused by cache + router + retrieval)
        q_emb: np.ndarray = self.pipeline.embed(question)

        # 2. Semantic cache lookup
        cached_entry, sim_score = self.cache.lookup(q_emb)

        if cached_entry is not None:
            # ── CACHE HIT ────────────────────────────────────────────────────
            latency_ms = (time.perf_counter() - t_start) * 1000
            metrics = RequestMetrics(
                query=question,
                cache_hit=True,
                model_used="cache",
                latency_ms=latency_ms,
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                similarity_score=sim_score,
                complexity_score=0.0,
            )
            self.tracker.log(metrics)
            return QueryResult(
                answer=cached_entry.answer,
                cache_hit=True,
                model_used="cache",
                latency_ms=latency_ms,
                cost_usd=0.0,
                complexity_score=0.0,
                similarity_score=sim_score,
                routing_decision=None,
                input_tokens=0,
                output_tokens=0,
            )

        # 3. Route to model
        decision: RoutingDecision = self.router.route(question, q_emb)

        # 4. Retrieve context
        chunks: list[str] = self.pipeline.retrieve(q_emb)

        # 5. Generate answer
        answer, input_toks, output_toks = self.pipeline.generate(
            query=question,
            model_name=decision.model,
            context_chunks=chunks,
        )

        # 6. Populate cache
        self.cache.add(
            embedding=q_emb,
            query=question,
            answer=answer,
            model_used=decision.model,
        )

        # 7. Log metrics
        latency_ms = (time.perf_counter() - t_start) * 1000
        metrics = RequestMetrics(
            query=question,
            cache_hit=False,
            model_used=decision.model,
            latency_ms=latency_ms,
            input_tokens=input_toks,
            output_tokens=output_toks,
            similarity_score=sim_score,
            complexity_score=decision.complexity_score,
            routing_signals=decision.signals,
        )
        self.tracker.log(metrics)

        return QueryResult(
            answer=answer,
            cache_hit=False,
            model_used=decision.model,
            latency_ms=latency_ms,
            cost_usd=metrics.cost_usd,
            complexity_score=decision.complexity_score,
            similarity_score=sim_score,
            routing_decision=decision,
            input_tokens=input_toks,
            output_tokens=output_toks,
        )

    # ── Convenience wrappers ───────────────────────────────────────────────────

    def ingest_file(self, filepath: str) -> int:
        return self.pipeline.ingest_file(filepath)

    def ingest_text(self, text: str, doc_id: str = "manual") -> int:
        return self.pipeline.ingest_text(text, doc_id)

    @property
    def metrics(self) -> MetricsTracker:
        return self.tracker

    @property
    def cache_stats(self) -> dict:
        return {
            "size":     self.cache.size,
            "hit_rate": self.cache.hit_rate,
            "entries":  self.cache.get_all_entries(),
        }
