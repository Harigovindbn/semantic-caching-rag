"""
Cost & Latency Tracker
───────────────────────
Persists per-request metrics to a JSONL file.
Each record captures:
  - timestamp
  - query (truncated)
  - cache_hit (bool)
  - model_used (flash | pro | cache)
  - latency_ms
  - input_tokens / output_tokens
  - cost_usd
  - similarity_score (cache lookup score)
  - complexity_score (router score)
  - routing_signals (dict)

The JSONL format makes it trivial to stream into Pandas for the dashboard.
"""

import json
import time
import os
from dataclasses import dataclass, asdict, field
from typing import Optional

from config import COST_TABLE, METRICS_FILE


@dataclass
class RequestMetrics:
    query: str
    cache_hit: bool
    model_used: str                # "cache", flash_model, pro_model
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    similarity_score: float = 0.0  # best FAISS score during cache lookup
    complexity_score: float = 0.0  # router complexity estimate
    routing_signals: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def compute_cost(self) -> None:
        """Fill in cost_usd from token counts and the cost table."""
        if self.model_used in COST_TABLE:
            rates = COST_TABLE[self.model_used]
            self.cost_usd = (
                self.input_tokens  / 1_000 * rates["input"] +
                self.output_tokens / 1_000 * rates["output"]
            )


class MetricsTracker:
    """Append-only JSONL logger + in-memory aggregation."""

    def __init__(self, filepath: str = METRICS_FILE) -> None:
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self._records: list[dict] = []
        self._load_existing()

    # ── write ─────────────────────────────────────────────────────────────────

    def log(self, metrics: RequestMetrics) -> None:
        metrics.compute_cost()
        record = asdict(metrics)
        # Truncate query for storage
        record["query"] = metrics.query[:200]
        self._records.append(record)
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    # ── read / aggregation ────────────────────────────────────────────────────

    def get_all(self) -> list[dict]:
        return list(self._records)

    def summary(self) -> dict:
        if not self._records:
            return {
                "total_requests": 0,
                "cache_hits": 0,
                "hit_rate": 0.0,
                "total_cost_usd": 0.0,
                "avg_latency_ms": 0.0,
                "flash_requests": 0,
                "pro_requests": 0,
                "estimated_savings_usd": 0.0,
            }

        hits        = sum(1 for r in self._records if r["cache_hit"])
        total       = len(self._records)
        total_cost  = sum(r["cost_usd"] for r in self._records)
        avg_latency = sum(r["latency_ms"] for r in self._records) / total
        flash_reqs  = sum(1 for r in self._records if "flash" in r["model_used"])
        pro_reqs    = sum(1 for r in self._records if "pro"   in r["model_used"])

        # Hypothetical cost if every request used Pro (no cache, no router)
        from config import PRO_MODEL, COST_TABLE
        pro_rate_in  = COST_TABLE[PRO_MODEL]["input"]
        pro_rate_out = COST_TABLE[PRO_MODEL]["output"]
        naive_cost = sum(
            r["input_tokens"]  / 1_000 * pro_rate_in +
            r["output_tokens"] / 1_000 * pro_rate_out
            for r in self._records
        )
        savings = naive_cost - total_cost

        return {
            "total_requests":       total,
            "cache_hits":           hits,
            "hit_rate":             hits / total if total else 0.0,
            "total_cost_usd":       round(total_cost, 6),
            "avg_latency_ms":       round(avg_latency, 1),
            "flash_requests":       flash_reqs,
            "pro_requests":         pro_reqs,
            "estimated_savings_usd": round(max(savings, 0), 6),
        }

    def _load_existing(self) -> None:
        if os.path.exists(self.filepath):
            with open(self.filepath, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self._records.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
