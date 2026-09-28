"""
Demo script — runs a batch of queries showcasing all system behaviours:
  • First query  → cache MISS, simple query → Flash model
  • Second query → cache MISS, complex query → Pro model
  • Third query  → semantically similar to first → cache HIT
  • Prints a cost/savings summary at the end

Usage: python demo.py
"""

from orchestrator import RAGOrchestrator

DEMO_QUERIES = [
    # (question, expected_behavior)
    ("What is the attention mechanism?",
     "Simple → Flash model"),
    ("Compare the mathematical foundations of BERT and GPT in terms of masking strategies "
     "and derive why bidirectional attention cannot be used for autoregressive generation",
     "Complex → Pro model"),
    ("What is attention in neural networks?",
     "Semantically similar to Q1 → Cache HIT"),
    ("What year was the transformer paper published and by which company?",
     "Simple fact → Flash model"),
    ("Derive the parameter count savings from LoRA for a GPT-3 sized model with rank r=4 "
     "and explain the mathematical intuition behind the low-rank assumption",
     "Complex derivation → Pro model"),
    ("How does LoRA reduce the number of parameters?",
     "Similar to Q5 → likely Cache HIT"),
]


def print_separator():
    print("\n" + "═"*70 + "\n")


def main():
    print_separator()
    print("  SEMANTIC CACHING RAG — DEMO")
    print("  Cost-aware routing + semantic cache demonstration")
    print_separator()

    orch = RAGOrchestrator()

    for i, (question, note) in enumerate(DEMO_QUERIES, 1):
        print(f"Query {i}: {note}")
        print(f"  Q: {question[:90]}{'…' if len(question)>90 else ''}")

        result = orch.query(question)

        cache_str = "✅ HIT" if result.cache_hit else "❌ MISS"
        print(f"  Cache:      {cache_str}  (similarity: {result.similarity_score:.3f})")
        print(f"  Model:      {result.model_used}")
        print(f"  Latency:    {result.latency_ms:.0f} ms")
        print(f"  Cost:       ${result.cost_usd:.6f}")
        if result.routing_decision:
            print(f"  Complexity: {result.complexity_score:.3f}")
            print(f"  Signals:    {result.routing_decision.signals}")
        print(f"  Answer:     {result.answer[:150]}…")
        print_separator()

    # Summary
    summary = orch.metrics.summary()
    print("FINAL SUMMARY")
    print(f"  Total requests:   {summary['total_requests']}")
    print(f"  Cache hits:       {summary['cache_hits']}  ({summary['hit_rate']*100:.1f}%)")
    print(f"  Flash requests:   {summary['flash_requests']}")
    print(f"  Pro requests:     {summary['pro_requests']}")
    print(f"  Total cost:       ${summary['total_cost_usd']:.6f}")
    print(f"  Estimated savings: ${summary['estimated_savings_usd']:.6f}")
    print("\nDone! Start the dashboard with: streamlit run dashboard.py")


if __name__ == "__main__":
    main()
