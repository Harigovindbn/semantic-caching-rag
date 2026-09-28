"""
Ingest sample ML paper content into ChromaDB.
Run this once before starting the API / dashboard.

Usage:
    python ingest_data.py
    python ingest_data.py --file path/to/paper.pdf
"""

import argparse
import sys
from pathlib import Path

# Seed knowledge base with landmark ML paper excerpts
SAMPLE_DOCUMENTS = [
    {
        "id": "attention_is_all_you_need",
        "title": "Attention Is All You Need (2017)",
        "text": """
        The dominant sequence transduction models are based on complex recurrent or convolutional neural networks
        that include an encoder and a decoder. The best performing models also connect the encoder and decoder
        through an attention mechanism. We propose a new simple network architecture, the Transformer,
        based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.
        Experiments on two machine translation tasks show these models to be superior in quality while being
        more parallelizable and requiring significantly less time to train.

        The Transformer uses multi-head attention in three different ways:
        1. In "encoder-decoder attention" layers, the queries come from the previous decoder layer,
           and the memory keys and values come from the output of the encoder.
        2. The encoder contains self-attention layers where all keys, values and queries come from the same place.
        3. The decoder contains self-attention layers that allow each position to attend to all positions
           up to and including that position.

        Multi-head attention allows the model to jointly attend to information from different representation
        subspaces at different positions. Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V.
        The scaling factor 1/sqrt(d_k) counteracts the vanishing gradient problem.

        Positional encoding adds information about position since the model contains no recurrence or convolution.
        Sinusoidal functions of different frequencies are used: PE(pos,2i) = sin(pos/10000^(2i/d_model)).

        The Transformer achieved 28.4 BLEU on the WMT 2014 English-to-German translation task,
        improving over best results including ensembles by over 2 BLEU. Training took 3.5 days
        on 8 P100 GPUs.
        """,
    },
    {
        "id": "bert_2018",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers (2018)",
        "text": """
        We introduce a new language representation model called BERT, which stands for Bidirectional
        Encoder Representations from Transformers. Unlike recent language representation models,
        BERT is designed to pre-train deep bidirectional representations from unlabeled text by
        jointly conditioning on both left and right context in all layers.

        BERT uses two pre-training tasks:
        1. Masked Language Model (MLM): 15% of tokens are masked at random. The model predicts
           the original vocabulary item. 80% replaced with [MASK], 10% with random word, 10% unchanged.
        2. Next Sentence Prediction (NSP): Binary classification whether sentence B follows sentence A.

        BERT base: 12 layers, 768 hidden size, 12 attention heads, 110M parameters.
        BERT large: 24 layers, 1024 hidden size, 16 attention heads, 340M parameters.

        BERT fine-tuning achieves state-of-the-art on 11 NLP tasks including GLUE benchmark (80.5%),
        SQuAD 1.1 (93.2% F1), SQuAD 2.0 (83.1% F1), and SWAG (86.3%).

        Key difference from GPT: BERT uses bidirectional self-attention (sees both directions),
        while GPT uses causal (left-to-right) attention to enable autoregressive generation.
        """,
    },
    {
        "id": "gpt3_2020",
        "title": "Language Models are Few-Shot Learners (GPT-3, 2020)",
        "text": """
        We train GPT-3, an autoregressive language model with 175 billion parameters, 10x more than
        any previous non-sparse language model. GPT-3 achieves strong performance on many NLP datasets,
        including translation, question-answering, and cloze tasks.

        GPT-3 demonstrates few-shot learning: given only a few examples in the prompt (in-context learning),
        it matches or exceeds fine-tuned models on several tasks. Zero-shot performance is weaker but
        improving with scale.

        Architecture: 96 layers, 96 attention heads, 12288 hidden size, context window of 2048 tokens.
        Trained on 300 billion tokens from Common Crawl, WebText, Books, and Wikipedia.

        Scaling laws: loss scales as a power law with both compute and dataset size. The optimal
        allocation is approximately equal scaling of model size and data. This was later refined in
        the Chinchilla paper (2022) which showed GPT-3 was significantly undertrained.

        In-context learning differs fundamentally from meta-learning: no gradient updates are performed.
        The model adapts purely via the attention mechanism over the prompt.
        """,
    },
    {
        "id": "rag_2020",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2020)",
        "text": """
        Retrieval-Augmented Generation (RAG) combines a parametric memory (the LLM weights) with
        a non-parametric memory (a dense vector index over external documents) for knowledge-intensive tasks.

        Architecture:
        - Retriever: DPR (Dense Passage Retrieval) — a bi-encoder that maps queries and passages
          to the same embedding space. Top-k passages retrieved via MIPS (Maximum Inner Product Search).
        - Generator: BART seq2seq model conditioned on the query + retrieved passages.

        Two formulations:
        1. RAG-Sequence: uses the same retrieved document for the entire sequence.
        2. RAG-Token: can attend to different documents for each token.

        RAG achieves state-of-the-art on open-domain QA (TriviaQA, WebQ, Natural Questions),
        fact verification, and knowledge-grounded generation without requiring full model retraining
        when the knowledge base changes (just update the index).

        Key advantage: knowledge can be updated without expensive retraining. The retriever
        can access facts that were not in the training data.
        """,
    },
    {
        "id": "lora_2021",
        "title": "LoRA: Low-Rank Adaptation of Large Language Models (2021)",
        "text": """
        LoRA (Low-Rank Adaptation) proposes freezing the pre-trained model weights and injecting
        trainable rank decomposition matrices into each layer of the Transformer architecture.
        This greatly reduces the number of trainable parameters for downstream tasks.

        Key insight: the weight updates during model adaptation have a low intrinsic rank.
        For a pre-trained weight matrix W ∈ R^(d×k), LoRA constrains the update by representing
        it with a low-rank decomposition: W + ΔW = W + BA, where B ∈ R^(d×r), A ∈ R^(r×k),
        and rank r << min(d, k).

        During training, W is frozen and only A and B receive gradient updates.
        Parameter count: instead of d×k parameters, we train d×r + r×k = r(d+k) parameters.
        For GPT-3 (r=4), LoRA reduces trainable parameters by 10,000x with no inference latency
        compared to adapters (since BA can be merged back into W).

        LoRA matches full fine-tuning quality on most tasks while being far more storage-efficient.
        One pre-trained model can serve many fine-tuned tasks by swapping LoRA adapters.

        Applied to GPT-3 175B with r=4: only 4.7M trainable parameters vs 175B total.
        """,
    },
    {
        "id": "flash_attention",
        "title": "FlashAttention: Fast and Memory-Efficient Exact Attention (2022)",
        "text": """
        Standard attention has O(N²) memory complexity in sequence length N, because the attention
        matrix of size N×N must be materialized in HBM (high-bandwidth memory).

        FlashAttention uses tiling to compute attention in blocks that fit in SRAM (on-chip memory),
        avoiding materializing the full N×N attention matrix. The algorithm:
        1. Divides Q, K, V matrices into blocks.
        2. Loads blocks from HBM to SRAM one at a time.
        3. Computes attention output incrementally using the online softmax trick.

        Complexity: O(N²d) FLOPs (same as standard attention), but O(N) memory instead of O(N²).
        In practice, FlashAttention is 2-4x faster than standard attention on A100 GPUs and
        uses 5-20x less memory, enabling much longer context windows.

        FlashAttention-2 (2023) further improves by reducing non-matmul FLOPs and better
        parallelization across attention heads, achieving ~2x speedup over FlashAttention.

        This enables training context lengths of 64K+ tokens that were previously infeasible.
        """,
    },
    {
        "id": "chinchilla_2022",
        "title": "Training Compute-Optimal Large Language Models (Chinchilla, 2022)",
        "text": """
        The Chinchilla paper revisits scaling laws for language models and finds that most large
        language models (including GPT-3) are significantly undertrained relative to their size.

        Key finding: for compute-optimal training, model size and training tokens should be
        scaled equally. The optimal ratio is approximately 20 training tokens per parameter.

        Chinchilla (70B parameters, 1.4T tokens) outperforms Gopher (280B, 300B tokens) on
        most tasks while using the same training compute, showing that smaller models trained
        on more data outperform larger models trained on less data.

        The scaling law: L(N, D) = E + A/N^α + B/D^β, where N = model params, D = training tokens,
        E is the irreducible loss, and α ≈ β ≈ 0.5.

        Implications:
        - Llama (Meta 2023) applied this by training a 65B model on 1.4T tokens, achieving
          competitive performance with GPT-3 (175B) at far lower inference cost.
        - Inference efficiency matters: a smaller model that matches quality is cheaper to deploy.
        """,
    },
    {
        "id": "vector_databases",
        "title": "Vector Databases and Approximate Nearest Neighbor Search",
        "text": """
        Vector databases store high-dimensional embeddings and support efficient similarity search.
        They are essential for RAG systems, recommendation engines, and semantic search.

        Key algorithms:
        1. HNSW (Hierarchical Navigable Small World): Graph-based ANN. Builds a multi-layer graph
           where upper layers have longer-range connections for fast navigation. O(log N) query time.
           Used by: ChromaDB, Weaviate, Qdrant.

        2. FAISS (Facebook AI Similarity Search): Supports multiple index types including:
           - IndexFlatIP: Exact inner product search, O(N).
           - IndexIVFFlat: Inverted file index, partitions space into Voronoi cells.
           - IndexPQ: Product quantization for compressed search.
           FAISS achieves billion-scale search with GPU support.

        3. ANNOY (Approximate Nearest Neighbors Oh Yeah): Forest of random projection trees.
           Good for static datasets with many queries.

        Cosine similarity vs Euclidean distance: for normalized vectors, cosine similarity equals
        the dot product and inner product search equals cosine similarity search.

        Semantic caching uses vector databases to find queries with similar meaning (high cosine
        similarity) to previously answered queries, avoiding redundant LLM calls and reducing cost.
        """,
    },
]


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG knowledge base")
    parser.add_argument("--file", help="Path to a PDF or text file to ingest", default=None)
    parser.add_argument("--clear", action="store_true", help="Clear existing data before ingesting")
    args = parser.parse_args()

    # Late import so config is loaded first
    from orchestrator import RAGOrchestrator

    orch = RAGOrchestrator()

    if args.clear:
        print("⚠️  Clearing existing ChromaDB collection …")
        orch.pipeline.chroma_client.delete_collection("ml_papers")
        orch.pipeline.collection = orch.pipeline.chroma_client.get_or_create_collection(
            name="ml_papers",
            metadata={"hnsw:space": "cosine"},
        )
        print("✅ Collection cleared.")

    if args.file:
        print(f"📄 Ingesting file: {args.file}")
        n = orch.ingest_file(args.file)
        print(f"✅ Done: {n} chunks")
        return

    # Ingest sample documents
    print(f"\n📚 Ingesting {len(SAMPLE_DOCUMENTS)} sample ML paper documents …\n")
    total_chunks = 0
    for doc in SAMPLE_DOCUMENTS:
        n = orch.ingest_text(doc["text"], doc_id=doc["id"])
        total_chunks += n
        print(f"  ✅ {doc['title']} → {n} chunks")

    print(f"\n🎉 Ingestion complete! {total_chunks} total chunks indexed in ChromaDB.")
    print("   You can now start the API server:\n   python api.py\n")


if __name__ == "__main__":
    main()
