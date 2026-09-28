"""
FastAPI REST API
─────────────────
Exposes the RAG Orchestrator as a REST service.

Endpoints:
  POST /query          — run a query through the full pipeline
  POST /ingest/text    — ingest raw text into the knowledge base
  POST /ingest/file    — ingest a file path (server-side)
  GET  /metrics        — raw metrics list
  GET  /metrics/summary— aggregated stats
  GET  /cache/stats    — cache size, hit rate, entries
  GET  /health         — liveness probe
"""

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from orchestrator import RAGOrchestrator


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────
orchestrator: Optional[RAGOrchestrator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global orchestrator
    orchestrator = RAGOrchestrator()
    yield
    # cleanup (none needed)


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Semantic Caching RAG API",
    description=(
        "Cost-aware RAG with semantic caching, complexity-based model routing, "
        "and real-time cost/latency tracking."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ──────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    cache_hit: bool
    model_used: str
    latency_ms: float
    cost_usd: float
    complexity_score: float
    similarity_score: float
    input_tokens: int
    output_tokens: int
    routing_signals: dict


class IngestTextRequest(BaseModel):
    text: str
    doc_id: str = "manual"


class IngestFileRequest(BaseModel):
    filepath: str


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if not orchestrator:
        raise HTTPException(503, "Orchestrator not ready")
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty")

    result = orchestrator.query(req.question)

    signals = {}
    if result.routing_decision:
        signals = result.routing_decision.signals

    return QueryResponse(
        answer=result.answer,
        cache_hit=result.cache_hit,
        model_used=result.model_used,
        latency_ms=round(result.latency_ms, 2),
        cost_usd=round(result.cost_usd, 8),
        complexity_score=round(result.complexity_score, 4),
        similarity_score=round(result.similarity_score, 4),
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        routing_signals=signals,
    )


@app.post("/ingest/text")
async def ingest_text(req: IngestTextRequest):
    if not orchestrator:
        raise HTTPException(503, "Orchestrator not ready")
    n = orchestrator.ingest_text(req.text, req.doc_id)
    return {"chunks_ingested": n, "doc_id": req.doc_id}


@app.post("/ingest/file")
async def ingest_file(req: IngestFileRequest):
    if not orchestrator:
        raise HTTPException(503, "Orchestrator not ready")
    try:
        n = orchestrator.ingest_file(req.filepath)
        return {"chunks_ingested": n, "filepath": req.filepath}
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


@app.get("/metrics")
async def get_metrics():
    if not orchestrator:
        raise HTTPException(503, "Orchestrator not ready")
    return {"records": orchestrator.metrics.get_all()}


@app.get("/metrics/summary")
async def get_summary():
    if not orchestrator:
        raise HTTPException(503, "Orchestrator not ready")
    return orchestrator.metrics.summary()


@app.get("/cache/stats")
async def get_cache_stats():
    if not orchestrator:
        raise HTTPException(503, "Orchestrator not ready")
    return orchestrator.cache_stats


# ── Dev server entry-point ────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
