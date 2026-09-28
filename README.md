# ⚡ Semantic Caching RAG — Cost-Aware Routing at Scale

> **The infrastructure layer that decides whether a RAG product is affordable at scale.**

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────┐
│    Semantic Cache Layer     │  ← Vector similarity index (cosine similarity)
│   cosine_sim ≥ 0.88 → HIT   │
└──────────┬──────────────────┘
           │ MISS
           ▼
┌─────────────────────────────┐
│    Complexity Router        │  ← 3-signal ensemble
│   heuristic + embedding     │    (length, jargon, seed similarity)
│   score < 0.6 → Flash       │
│   score ≥ 0.6 → Pro         │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│    RAG Pipeline             │
│   ChromaDB retrieval        │
│   Google GenAI generation   │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│   Metrics Logger (JSONL)    │  ← cost · latency · tokens · routing signals
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│   Streamlit Dashboard       │  ← real-time analytics
└─────────────────────────────┘

```

## ✨ Key Features

| 

| **Feature** | **Details** | 
| **Semantic Cache** | Vector similarity index using `SentenceTransformer` and cosine ANN search | 
| **Cost-Aware Router** | 3-signal ensemble: heuristic + embedding similarity to hard/easy seeds | 
| **Dual-Model Routing** | `gemini-2.5-flash` for simple queries, `gemini-2.5-pro` for complex reasoning | 
| **RAG Pipeline** | ChromaDB + `all-MiniLM-L6-v2` embeddings, sliding-window chunker | 
| **Google GenAI SDK** | Upgraded to official modern `google-genai` SDK | 
| **Metrics Tracker** | JSONL append-only log with real-time cost/savings aggregation | 
| **Analytics Dashboard** | Cache hit rate gauge, cost waterfall, latency box plots, routing signal breakdown | 
| **REST API** | FastAPI with Pydantic validation and OpenAPI docs at `/docs` | 

## 🚀 Quick Start

### 1. Clone & Install

```
git clone https://github.com/Harigovindbn/semantic-caching-rag.git
cd "Semantic caching RAG"

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
# or: source venv/bin/activate  # Linux/Mac

# Install dependencies
python -m pip install -r requirements.txt

```

### 2. Configure API Key

Create a `.env` file in the root directory:

```
GEMINI_API_KEY=AIzaSyYourActualGeminiApiKeyHere

```

Get a free Gemini API key at [aistudio.google.com](https://aistudio.google.com?utm_source=gemini)

### 3. Ingest Knowledge Base

```
python ingest_data.py

```

This loads landmark ML paper excerpts (Transformers, BERT, GPT-3, RAG, LoRA, FlashAttention, Chinchilla, Vector DBs).

### 4. Start the API Server

```
python api.py
# API docs available at: http://localhost:8000/docs

```

### 5. Launch the Dashboard

In a second terminal window:

```
.\venv\Scripts\activate
python -m streamlit run dashboard.py
# Dashboard available at: http://localhost:8501

```

## 📊 Dashboard Panels

| **Panel** | **Visualisation** | 
| Query Interface | Real-time answer with routing decision breakdown | 
| KPI Row | Total requests · Cache hit rate · Total cost · Savings · Avg latency | 
| Cache Hit Gauge | Animated gauge with colour-coded zones | 
| Model Usage | Donut chart (Cache / Flash / Pro split) | 
| Cost Waterfall | Per-request cost bars + cumulative line | 
| Latency Distribution | Box plots by model | 
| Complexity Histogram | Score distribution with routing threshold | 
| Cache Browser | Sortable table + most-hit queries bar chart | 

## 💰 Cost Model

| **Model** | **Input (per 1K tokens)** | **Output (per 1K tokens)** | 
| `gemini-2.5-flash` | \$0.000075 | \$0.0003 | 
| `gemini-2.5-pro` | \$0.00125 | \$0.005 | 
| Cache hit | \$0.000000 | \$0.0000 | 

At 1,000 requests with 50% cache hit rate and 80% Flash / 20% Pro routing:

* **Naive (all Pro):** \~\$2.50

* **With caching + routing:** \~\$0.15

* **Savings: \~94%** ✅

## 🔬 Routing Signal Details

The complexity router uses a **3-signal ensemble**:

1. **Heuristic (60% weight)**

   * Query length (capped at 40 words → 0.4 score)

   * Multi-part detection: "compare", "vs", "contrast" (+0.25)

   * Technical jargon density (8+ ML terms → +0.28)

   * Mathematical request: "derive", "prove" (+0.20)

   * Deep reasoning: "why", "how does" (+0.10)

2. **Embedding (40% weight)**

   * Cosine similarity to 6 hard seed questions

   * Cosine similarity to 6 easy seed questions

   * Score = (hard_sim − easy_sim + 1) / 2

3. **Decision**

   * Combined score ≥ 0.60 → `gemini-2.5-pro`

   * Combined score < 0.60 → `gemini-2.5-flash`

## 📁 Project Structure

```
Semantic caching RAG/
├── config.py            # Settings and model mappings
├── semantic_cache.py    # Vector-backed cache with cosine similarity
├── complexity_router.py # 3-signal complexity classifier
├── rag_pipeline.py      # Ingest → Retrieve → Generate (google-genai SDK)
├── orchestrator.py      # Core workflow orchestrator
├── metrics_tracker.py   # JSONL cost/latency logger
├── api.py               # FastAPI REST endpoints
├── dashboard.py         # Streamlit analytics dashboard
├── ingest_data.py       # Knowledge base population utility
├── requirements.txt
├── .env
└── data/                # Created at runtime
    ├── chroma_db/       # ChromaDB vector store
    └── metrics.jsonl    # Per-request metrics log

```

## 🛠️ API Endpoints

```
POST /query            Query the RAG pipeline
POST /ingest/text        Ingest raw text
POST /ingest/file        Ingest a file (.pdf / .txt)
GET  /metrics            Raw metrics records
GET  /metrics/summary    Aggregated statistics
GET  /cache/stats        Cache size, hit rate, entries
GET  /health             Liveness probe
GET  /docs               Swagger UI documentation

```

## 🎓 What This Demonstrates (Resume Talking Points)

* **Systems thinking**: Cost-latency tradeoffs in production ML systems

* **Algorithmic depth**: Approximate Nearest Neighbor (ANN) search, cosine similarity, embedding spaces

* **LLM integration**: Modern `google-genai` SDK integration with multi-model routing

* **LLMOps & Observability**: Metrics tracking, token cost logging, and analytics for AI systems

* **Software engineering**: Modular architecture, FastAPI REST framework, type annotations

Built with ❤️ as a demonstration of production-grade RAG infrastructure.