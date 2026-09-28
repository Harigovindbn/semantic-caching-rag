"""
Streamlit Dashboard — Semantic Caching RAG
───────────────────────────────────────────
A premium, real-time analytics dashboard that shows:
  • Live query interface with routing decision breakdown
  • Cost & savings tracker
  • Cache hit rate gauge
  • Latency distribution
  • Model usage split (Flash vs Pro vs Cache)
  • Per-request cost waterfall
  • Cache entry browser

Run with:  streamlit run dashboard.py
"""

import time
import json
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import requests

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Semantic Caching RAG — Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1530 50%, #0a1628 100%);
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
    border: 1px solid rgba(99, 179, 237, 0.2);
    border-radius: 16px;
    padding: 20px;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
}
.metric-card:hover {
    border-color: rgba(99, 179, 237, 0.5);
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(99, 179, 237, 0.15);
}

/* Glowing header */
h1 {
    background: linear-gradient(120deg, #63b3ed, #9f7aea, #68d391);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700 !important;
    font-size: 2.2rem !important;
}

/* Answer box */
.answer-box {
    background: linear-gradient(135deg, rgba(99,179,237,0.1) 0%, rgba(159,122,234,0.05) 100%);
    border-left: 3px solid #63b3ed;
    border-radius: 0 12px 12px 0;
    padding: 16px 20px;
    margin: 12px 0;
    font-size: 0.95rem;
    line-height: 1.6;
}

/* Cache HIT badge */
.badge-hit {
    display: inline-block;
    background: linear-gradient(90deg, #38a169, #48bb78);
    color: white;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.badge-miss {
    display: inline-block;
    background: linear-gradient(90deg, #d69e2e, #ecc94b);
    color: #1a1a1a;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.badge-flash {
    display: inline-block;
    background: linear-gradient(90deg, #3182ce, #63b3ed);
    color: white;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
}
.badge-pro {
    display: inline-block;
    background: linear-gradient(90deg, #6b46c1, #9f7aea);
    color: white;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(10, 14, 26, 0.95) !important;
    border-right: 1px solid rgba(99, 179, 237, 0.15) !important;
}

/* Streamlit elements */
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 12px;
    padding: 12px;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
API_URL   = "http://localhost:8000"
METRICS_FILE = "./data/metrics.jsonl"

# ── Helpers ────────────────────────────────────────────────────────────────────

def api_post(endpoint: str, payload: dict) -> dict | None:
    try:
        r = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=120)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("⚠️ API server not running. Start it with: `python api.py`")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_get(endpoint: str) -> dict | None:
    try:
        r = requests.get(f"{API_URL}{endpoint}", timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def load_metrics_local() -> list[dict]:
    """Fallback: read JSONL directly when API is down."""
    path = Path(METRICS_FILE)
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return records


def model_label(model_str: str) -> str:
    if model_str == "cache":
        return "⚡ Cache"
    if "flash" in model_str.lower():
        return "🔵 Flash"
    if "pro" in model_str.lower():
        return "🟣 Pro"
    return model_str


def model_badge(model_str: str) -> str:
    if model_str == "cache":
        return '<span class="badge-hit">⚡ CACHE HIT</span>'
    if "flash" in model_str.lower():
        return '<span class="badge-flash">🔵 FLASH</span>'
    return '<span class="badge-pro">🟣 PRO</span>'


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ RAG Control Panel")
    st.markdown("---")

    # API status
    health = api_get("/health")
    if health:
        st.markdown("🟢 **API:** Connected")
    else:
        st.markdown("🔴 **API:** Offline (using local data)")

    st.markdown("---")

    # Ingest section
    st.markdown("### 📄 Ingest Documents")
    ingest_tab = st.radio("Source", ["Paste Text", "File Path"], label_visibility="collapsed")

    if ingest_tab == "Paste Text":
        doc_text = st.text_area("Document text", height=150, placeholder="Paste ML paper abstract or content…")
        doc_id   = st.text_input("Document ID", value="manual_doc")
        if st.button("📥 Ingest Text", use_container_width=True):
            if doc_text.strip():
                res = api_post("/ingest/text", {"text": doc_text, "doc_id": doc_id})
                if res:
                    st.success(f"✅ Ingested {res['chunks_ingested']} chunks")
    else:
        file_path = st.text_input("File path", placeholder="./data/papers/attention.pdf")
        if st.button("📥 Ingest File", use_container_width=True):
            if file_path.strip():
                res = api_post("/ingest/file", {"filepath": file_path})
                if res:
                    st.success(f"✅ Ingested {res['chunks_ingested']} chunks")

    st.markdown("---")

    # Cache threshold control
    st.markdown("### ⚙️ Live Settings")
    st.caption("(restart API to apply changes to .env)")
    cache_thresh = st.slider("Cache similarity threshold", 0.70, 0.99, 0.90, 0.01)
    complex_thresh = st.slider("Complexity threshold", 0.30, 0.90, 0.60, 0.01)

    st.markdown("---")
    st.markdown("### 📚 Sample Questions")
    sample_qs = [
        "What is attention mechanism?",
        "Compare BERT vs GPT architecture in detail",
        "What year was the transformer paper published?",
        "Derive the ELBO for a VAE mathematically",
        "What is dropout used for?",
        "How does LoRA reduce parameters during fine-tuning?",
    ]
    selected_sample = st.selectbox("Load sample", ["— pick one —"] + sample_qs)


# ── Main layout ────────────────────────────────────────────────────────────────
st.markdown("# ⚡ Semantic Caching RAG")
st.markdown(
    "**Cost-aware routing** · **Semantic cache** · **Real-time analytics** "
    "— the infrastructure layer that makes RAG affordable at scale."
)
st.markdown("---")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_query, tab_analytics, tab_cache = st.tabs(
    ["💬 Query Interface", "📊 Analytics Dashboard", "🗂️ Cache Browser"]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Query Interface
# ══════════════════════════════════════════════════════════════════════════════
with tab_query:
    col_q, col_meta = st.columns([2, 1])

    with col_q:
        st.markdown("### Ask a Question")
        default_q = selected_sample if selected_sample != "— pick one —" else ""
        question = st.text_area(
            "Your question",
            value=default_q,
            height=100,
            placeholder="e.g. What is the attention mechanism in transformers?",
            label_visibility="collapsed",
        )

        send_col, clear_col = st.columns([3, 1])
        with send_col:
            send_btn = st.button("🚀 Ask", use_container_width=True, type="primary")
        with clear_col:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.pop("last_result", None)

    with col_meta:
        st.markdown("### Routing Preview")
        if question.strip():
            # Simple client-side heuristic for preview only
            words = len(question.split())
            est_complexity = min(words / 40 + (0.25 if "compare" in question.lower() else 0), 1.0)
            est_model = "🟣 Pro (gemini-1.5-pro)" if est_complexity >= 0.6 else "🔵 Flash (gemini-1.5-flash)"
            st.metric("Estimated complexity", f"{est_complexity:.2f}")
            st.markdown(f"**Est. model:** {est_model}")
            st.caption("Final routing happens server-side with embedding signals.")

    # ── Send query ─────────────────────────────────────────────────────────────
    if send_btn and question.strip():
        with st.spinner("⏳ Processing query…"):
            result = api_post("/query", {"question": question})
            if result:
                st.session_state["last_result"] = result

    # ── Display result ─────────────────────────────────────────────────────────
    if "last_result" in st.session_state:
        r = st.session_state["last_result"]

        st.markdown("---")
        st.markdown("#### Answer")

        # Badge row
        badge_html = model_badge(r["model_used"])
        if r["cache_hit"]:
            status_html = '<span class="badge-hit">⚡ CACHE HIT</span>'
        else:
            status_html = '<span class="badge-miss">🔄 CACHE MISS</span>'
        st.markdown(f"{status_html} &nbsp; {badge_html}", unsafe_allow_html=True)

        st.markdown(f'<div class="answer-box">{r["answer"]}</div>', unsafe_allow_html=True)

        # Metrics row
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("⏱️ Latency",      f"{r['latency_ms']:.0f} ms")
        m2.metric("💰 Cost",         f"${r['cost_usd']:.6f}")
        m3.metric("🔀 Complexity",   f"{r['complexity_score']:.2f}")
        m4.metric("🔍 Cache sim.",   f"{r['similarity_score']:.3f}")
        m5.metric("🔢 Tokens",       f"{r['input_tokens']+r['output_tokens']:,}")

        # Routing signals expander
        if r["routing_signals"]:
            with st.expander("🔬 Routing Signal Breakdown"):
                sig = r["routing_signals"]
                sig_df = pd.DataFrame([
                    {"Signal": "Heuristic",   "Score": sig.get("heuristic", 0)},
                    {"Signal": "Embedding",   "Score": sig.get("embedding", 0)},
                    {"Signal": "Combined",    "Score": sig.get("combined",  0)},
                ])
                fig_bar = px.bar(
                    sig_df, x="Score", y="Signal", orientation="h",
                    color="Score", color_continuous_scale=["#3182ce", "#9f7aea"],
                    range_x=[0, 1],
                    height=180,
                )
                fig_bar.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e2e8f0",
                    showlegend=False,
                    margin=dict(l=0, r=0, t=0, b=0),
                    coloraxis_showscale=False,
                )
                fig_bar.add_vline(x=0.6, line_dash="dash", line_color="#fc8181", annotation_text="threshold")
                st.plotly_chart(fig_bar, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Analytics Dashboard
# ══════════════════════════════════════════════════════════════════════════════
with tab_analytics:
    # Load data
    metrics_data = api_get("/metrics")
    summary_data = api_get("/metrics/summary")

    records: list[dict] = []
    if metrics_data:
        records = metrics_data.get("records", [])
    else:
        records = load_metrics_local()

    summary: dict = summary_data or {}

    if not records:
        st.info("💡 No requests yet. Ask some questions in the Query tab to see analytics here!")
    else:
        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df["model_label"] = df["model_used"].apply(model_label)

        # ── KPI Row ───────────────────────────────────────────────────────────
        st.markdown("### 📈 Key Performance Indicators")
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Total Requests",    summary.get("total_requests", len(df)))
        k2.metric("Cache Hit Rate",    f"{summary.get('hit_rate', 0)*100:.1f}%")
        k3.metric("Total Cost",        f"${summary.get('total_cost_usd', 0):.6f}")
        k4.metric("Est. Savings",      f"${summary.get('estimated_savings_usd', 0):.6f}")
        k5.metric("Avg Latency",       f"{summary.get('avg_latency_ms', 0):.0f} ms")
        k6.metric("Flash / Pro",
                  f"{summary.get('flash_requests',0)} / {summary.get('pro_requests',0)}")

        st.markdown("---")

        row1_col1, row1_col2 = st.columns([1, 1])

        # ── Cache hit rate gauge ───────────────────────────────────────────────
        with row1_col1:
            st.markdown("#### 🎯 Cache Hit Rate")
            hit_rate = summary.get("hit_rate", 0) * 100
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=hit_rate,
                number={"suffix": "%", "font": {"color": "#e2e8f0", "size": 40}},
                delta={"reference": 50, "increasing": {"color": "#48bb78"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#718096"},
                    "bar":  {"color": "#63b3ed", "thickness": 0.3},
                    "steps": [
                        {"range": [0,  40], "color": "rgba(252,129,129,0.2)"},
                        {"range": [40, 70], "color": "rgba(236,201,75,0.2)"},
                        {"range": [70,100], "color": "rgba(72,187,120,0.2)"},
                    ],
                    "threshold": {
                        "line": {"color": "#fc8181", "width": 3},
                        "thickness": 0.8,
                        "value": 40,
                    },
                },
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                height=280,
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        # ── Model usage donut ──────────────────────────────────────────────────
        with row1_col2:
            st.markdown("#### 🔀 Model Usage Split")
            model_counts = df["model_label"].value_counts().reset_index()
            model_counts.columns = ["Model", "Count"]
            fig_donut = px.pie(
                model_counts,
                names="Model",
                values="Count",
                hole=0.55,
                color="Model",
                color_discrete_map={
                    "⚡ Cache": "#48bb78",
                    "🔵 Flash": "#63b3ed",
                    "🟣 Pro":   "#9f7aea",
                },
            )
            fig_donut.update_traces(
                textposition="inside",
                textinfo="percent+label",
                marker=dict(line=dict(color="rgba(0,0,0,0.3)", width=2)),
            )
            fig_donut.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                legend=dict(font=dict(color="#e2e8f0")),
                height=280,
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=True,
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        # ── Cost per request waterfall ─────────────────────────────────────────
        st.markdown("#### 💰 Cost Per Request Over Time")
        df_sorted = df.sort_values("timestamp")
        df_sorted["cumulative_cost"] = df_sorted["cost_usd"].cumsum()
        df_sorted["request_no"] = range(1, len(df_sorted)+1)

        fig_cost = go.Figure()
        # Bar per request (colored by model)
        color_map = {"⚡ Cache": "#48bb78", "🔵 Flash": "#63b3ed", "🟣 Pro": "#9f7aea"}
        for mlabel, group in df_sorted.groupby("model_label"):
            fig_cost.add_trace(go.Bar(
                x=group["request_no"],
                y=group["cost_usd"],
                name=mlabel,
                marker_color=color_map.get(mlabel, "#a0aec0"),
                opacity=0.85,
            ))
        # Cumulative line
        fig_cost.add_trace(go.Scatter(
            x=df_sorted["request_no"],
            y=df_sorted["cumulative_cost"],
            name="Cumulative Cost",
            line=dict(color="#fc8181", width=2.5, dash="dot"),
            yaxis="y2",
        ))
        fig_cost.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            xaxis=dict(title="Request #", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Cost (USD)", gridcolor="rgba(255,255,255,0.05)"),
            yaxis2=dict(title="Cumulative Cost", overlaying="y", side="right", showgrid=False),
            legend=dict(font=dict(color="#e2e8f0")),
            height=360,
            barmode="stack",
            margin=dict(l=20, r=60, t=20, b=40),
        )
        st.plotly_chart(fig_cost, use_container_width=True)

        # ── Latency distribution ───────────────────────────────────────────────
        row2_col1, row2_col2 = st.columns([1, 1])

        with row2_col1:
            st.markdown("#### ⏱️ Latency Distribution by Model")
            fig_lat = px.box(
                df_sorted, x="model_label", y="latency_ms",
                color="model_label",
                color_discrete_map={"⚡ Cache": "#48bb78", "🔵 Flash": "#63b3ed", "🟣 Pro": "#9f7aea"},
                labels={"latency_ms": "Latency (ms)", "model_label": "Model"},
                height=320,
            )
            fig_lat.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                showlegend=False,
                margin=dict(l=20, r=20, t=20, b=40),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            )
            st.plotly_chart(fig_lat, use_container_width=True)

        with row2_col2:
            st.markdown("#### 🔍 Complexity Score Distribution")
            fig_hist = px.histogram(
                df_sorted[df_sorted["complexity_score"] > 0],
                x="complexity_score",
                nbins=20,
                color_discrete_sequence=["#9f7aea"],
                labels={"complexity_score": "Complexity Score"},
                height=320,
            )
            fig_hist.add_vline(x=0.6, line_dash="dash", line_color="#fc8181",
                               annotation_text="routing threshold", annotation_font_color="#fc8181")
            fig_hist.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                showlegend=False,
                margin=dict(l=20, r=20, t=20, b=40),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # ── Recent requests table ──────────────────────────────────────────────
        st.markdown("#### 📋 Recent Requests")
        display_cols = ["timestamp", "query", "model_label", "cache_hit",
                        "latency_ms", "cost_usd", "complexity_score", "similarity_score"]
        available = [c for c in display_cols if c in df_sorted.columns]
        recent = df_sorted.sort_values("timestamp", ascending=False).head(20)[available].copy()
        recent["latency_ms"]     = recent["latency_ms"].round(1)
        recent["cost_usd"]       = recent["cost_usd"].apply(lambda x: f"${x:.6f}")
        recent["complexity_score"] = recent["complexity_score"].round(3)
        recent["similarity_score"] = recent["similarity_score"].round(3)
        st.dataframe(recent, use_container_width=True, height=300)

        # ── Savings analysis ───────────────────────────────────────────────────
        savings = summary.get("estimated_savings_usd", 0)
        total   = summary.get("total_cost_usd", 0)
        if savings > 0:
            st.markdown("---")
            pct = savings / (savings + total) * 100 if (savings + total) > 0 else 0
            st.success(
                f"💰 **Cost Savings Report:** Semantic caching + routing saved you "
                f"**${savings:.6f}** ({pct:.1f}%) vs sending every query to Gemini Pro."
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Cache Browser
# ══════════════════════════════════════════════════════════════════════════════
with tab_cache:
    st.markdown("### 🗂️ Semantic Cache Inspector")

    cache_data = api_get("/cache/stats")
    if cache_data is None:
        st.warning("API offline — cache data unavailable.")
    else:
        c1, c2 = st.columns(2)
        c1.metric("Cache Size",     cache_data.get("size", 0))
        c2.metric("Hit Rate",       f"{cache_data.get('hit_rate', 0)*100:.1f}%")

        entries = cache_data.get("entries", [])
        if entries:
            st.markdown("---")
            st.markdown("#### Cached Entries")
            cache_df = pd.DataFrame(entries)
            cache_df["timestamp"] = pd.to_datetime(cache_df["timestamp"], unit="s")
            cache_df = cache_df.sort_values("hit_count", ascending=False)
            st.dataframe(cache_df, use_container_width=True, height=400)

            # Hit count bar chart
            st.markdown("#### Most-Hit Queries")
            top_entries = cache_df.head(10)
            fig_hits = px.bar(
                top_entries,
                x="hit_count",
                y="query",
                orientation="h",
                color="hit_count",
                color_continuous_scale=["#3182ce", "#9f7aea", "#68d391"],
                height=350,
                labels={"hit_count": "Cache Hits", "query": "Query"},
            )
            fig_hits.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                coloraxis_showscale=False,
                showlegend=False,
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)", autorange="reversed"),
            )
            st.plotly_chart(fig_hits, use_container_width=True)
        else:
            st.info("Cache is empty — ask some questions first!")

# ── Auto-refresh ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    auto_refresh = st.toggle("🔄 Auto-refresh (30s)", value=False)
    if auto_refresh:
        import time as _time
        _time.sleep(30)
        st.rerun()

    st.markdown("---")
    st.caption("Built with ❤️ · Semantic Caching RAG v1.0")
