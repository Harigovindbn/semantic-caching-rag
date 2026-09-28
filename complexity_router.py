"""
Complexity Router
──────────────────
Classifies an incoming query as "simple" or "complex" and selects
the appropriate Gemini model.

Routing logic (3-signal ensemble):
  1. Heuristic features   — question length, multi-part detection,
                            technical jargon density, temporal scope.
  2. Embedding-based      — cosine distance from a set of seed
                            "hard" and "easy" example questions.
  3. (optional) LLM judge — fast Flash call that returns a 0-1 score;
                            only used when heuristics are ambiguous.

The three signals are blended with configurable weights. If the
combined complexity score >= COMPLEXITY_THRESHOLD → Pro model,
otherwise → Flash model.
"""

import re
import numpy as np
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer

from config import (
    COMPLEXITY_THRESHOLD,
    FLASH_MODEL,
    PRO_MODEL,
    EMBEDDING_MODEL,
)


# ── Hard vs Easy seed questions (for embedding-based signal) ──────────────────
HARD_SEEDS = [
    "Compare the mathematical foundations of transformers and RNNs in terms of gradient flow",
    "How does the attention mechanism in BERT differ from GPT in terms of causal masking?",
    "Derive the ELBO for a variational autoencoder and explain each term",
    "What are the tradeoffs between contrastive learning and generative pre-training?",
    "Explain how LoRA approximates full fine-tuning and derive its parameter count",
    "How does Flash Attention reduce memory from O(n²) to O(n)?",
]

EASY_SEEDS = [
    "What is a transformer?",
    "What does RAG stand for?",
    "Define overfitting",
    "What is the purpose of dropout?",
    "What is a learning rate?",
    "What year was GPT-3 released?",
]

# Technical jargon list (increases complexity score)
JARGON_TERMS = {
    "gradient", "backpropagation", "attention", "embedding", "softmax",
    "transformer", "bert", "gpt", "llm", "fine-tun", "hyperparameter",
    "regularization", "loss function", "optimizer", "batch normalization",
    "convolution", "autoencoder", "latent", "contrastive", "distillation",
    "quantization", "lora", "rlhf", "reward model", "kl divergence",
    "elbo", "variational", "bayesian", "markov", "stochastic",
    "causal", "masked", "positional encoding", "token", "context window",
}


@dataclass
class RoutingDecision:
    model: str
    complexity_score: float
    reasoning: str
    signals: dict


class ComplexityRouter:
    """Routes queries to the appropriate model based on complexity."""

    def __init__(self, embed_model: SentenceTransformer | None = None) -> None:
        self._embed_model = embed_model  # shared with RAG pipeline
        self._hard_embeddings: np.ndarray | None = None
        self._easy_embeddings: np.ndarray | None = None

    def _ensure_seeds_embedded(self) -> None:
        if self._hard_embeddings is None and self._embed_model is not None:
            self._hard_embeddings = self._embed_model.encode(HARD_SEEDS, normalize_embeddings=True)
            self._easy_embeddings = self._embed_model.encode(EASY_SEEDS, normalize_embeddings=True)

    # ── public API ────────────────────────────────────────────────────────────

    def route(self, query: str, query_embedding: np.ndarray | None = None) -> RoutingDecision:
        """Decide which model to use for this query."""
        signals = {}

        # Signal 1: heuristic features
        heuristic_score = self._heuristic_score(query)
        signals["heuristic"] = round(heuristic_score, 3)

        # Signal 2: embedding similarity to hard/easy seeds
        embed_score = 0.5  # neutral default
        if query_embedding is not None and self._embed_model is not None:
            self._ensure_seeds_embedded()
            embed_score = self._embedding_score(query_embedding)
        signals["embedding"] = round(embed_score, 3)

        # Blend: 60% heuristic, 40% embedding
        if query_embedding is not None and self._embed_model is not None:
            combined = 0.6 * heuristic_score + 0.4 * embed_score
        else:
            combined = heuristic_score
        signals["combined"] = round(combined, 3)

        # Route
        if combined >= COMPLEXITY_THRESHOLD:
            model = PRO_MODEL
            reasoning = f"Score {combined:.2f} ≥ threshold {COMPLEXITY_THRESHOLD} → complex query"
        else:
            model = FLASH_MODEL
            reasoning = f"Score {combined:.2f} < threshold {COMPLEXITY_THRESHOLD} → simple query"

        return RoutingDecision(
            model=model,
            complexity_score=combined,
            reasoning=reasoning,
            signals=signals,
        )

    # ── internal signals ──────────────────────────────────────────────────────

    def _heuristic_score(self, query: str) -> float:
        score = 0.0
        q = query.lower()

        # Length penalty (longer queries tend to be harder)
        words = q.split()
        n = len(words)
        score += min(n / 40, 0.4)  # cap at 0.4

        # Multi-part question ("and", "vs", "compare", sub-questions)
        if re.search(r"\b(compare|contrast|vs\.?|versus|difference between)\b", q):
            score += 0.25
        if q.count("?") > 1 or re.search(r"\b(also|additionally|furthermore|moreover)\b", q):
            score += 0.15

        # Jargon density
        jargon_hits = sum(1 for term in JARGON_TERMS if term in q)
        score += min(jargon_hits * 0.07, 0.28)

        # Mathematical / derivation requests
        if re.search(r"\b(derive|proof|mathematically|formula|equation|theorem)\b", q):
            score += 0.2

        # "Why" / "How does" → deeper reasoning required
        if re.search(r"^(why|how does|explain why|what causes)\b", q):
            score += 0.1

        return min(score, 1.0)

    def _embedding_score(self, query_embedding: np.ndarray) -> float:
        """Cosine similarity to hard seeds minus similarity to easy seeds."""
        q = query_embedding.astype(np.float32)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm

        hard_sim = float(np.max(self._hard_embeddings @ q))
        easy_sim = float(np.max(self._easy_embeddings @ q))

        # Map difference to [0, 1]
        raw = (hard_sim - easy_sim + 1) / 2
        return float(np.clip(raw, 0.0, 1.0))
