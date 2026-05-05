"""
Evaluation Framework
====================
Measures:
  • Retrieval: Recall@k, MRR (Mean Reciprocal Rank)
  • Answer quality: Correctness (ROUGE-L token F1), Faithfulness, Groundedness
  • Latency: total, retrieval, generation

Provides a benchmark suite that compares hybrid RAG vs. keyword-only baseline.
"""

from __future__ import annotations

import re
import statistics
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from agentic_rag.modules.retriever import HybridRetriever, RetrievedDoc
from agentic_rag.modules.pipeline import PipelineResult


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark queries with ground-truth relevant paper IDs
# ─────────────────────────────────────────────────────────────────────────────

BENCHMARK_QUERIES: List[Dict] = [
    {
        "query": "How does dense retrieval compare to sparse keyword search for open-domain QA?",
        "relevant_ids": ["2004.04906", "2208.11970"],
        "reference_answer": (
            "Dense retrieval using dual-encoder frameworks like DPR substantially outperforms "
            "sparse BM25/TF-IDF methods on open-domain QA benchmarks, as shown by Karpukhin et al. "
            "RAG combines dense retrieval with generative models for knowledge-intensive tasks."
        ),
    },
    {
        "query": "What is the ReAct framework and how does it use tool use to improve reasoning?",
        "relevant_ids": ["2210.11610", "2302.07842"],
        "reference_answer": (
            "ReAct interleaves reasoning traces with actions to synergise chain-of-thought reasoning "
            "and tool use. Toolformer extends this by teaching LLMs to call APIs self-supervised."
        ),
    },
    {
        "query": "How do self-reflective RAG systems like Self-RAG improve faithfulness?",
        "relevant_ids": ["2310.06825", "2312.10997"],
        "reference_answer": (
            "Self-RAG trains models to retrieve on-demand and generate special reflection tokens "
            "to critique their own outputs, improving faithfulness over standard RAG pipelines."
        ),
    },
    {
        "query": "What are the tradeoffs between RAG and fine-tuning for LLMs?",
        "relevant_ids": ["2401.10020", "2208.11970"],
        "reference_answer": (
            "RAG is preferred when knowledge is dynamic or domain-specific without requiring "
            "parameter updates, while fine-tuning embeds knowledge into weights but is costly "
            "to update. RAG vs Fine-tuning paper shows task-dependent tradeoffs."
        ),
    },
    {
        "query": "How does chain-of-thought prompting relate to few-shot learning in large language models?",
        "relevant_ids": ["2201.11903", "2005.14165"],
        "reference_answer": (
            "Chain-of-thought prompting extends few-shot learning by providing intermediate reasoning "
            "steps as exemplars. GPT-3 demonstrated that few-shot learning emerges with scale; "
            "chain-of-thought further unlocks complex arithmetic and reasoning capabilities."
        ),
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Fraction of relevant docs found in the top-k results."""
    if not relevant_ids:
        return 1.0
    top_k = set(retrieved_ids[:k])
    hits = sum(1 for rid in relevant_ids if rid in top_k)
    return hits / len(relevant_ids)


def reciprocal_rank(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """MRR: 1/rank of first relevant document."""
    rel_set = set(relevant_ids)
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in rel_set:
            return 1.0 / rank
    return 0.0


def rouge_l_f1(hypothesis: str, reference: str) -> float:
    """
    Approximate ROUGE-L using token-level F1 (longest common subsequence skipped
    for simplicity; we use unigram overlap as a fast proxy).
    """
    def tokenize(text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    hyp_tokens = tokenize(hypothesis)
    ref_tokens = tokenize(reference)
    if not hyp_tokens or not ref_tokens:
        return 0.0

    hyp_set = set(hyp_tokens)
    ref_set = set(ref_tokens)
    overlap = len(hyp_set & ref_set)
    precision = overlap / len(hyp_set)
    recall = overlap / len(ref_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ─────────────────────────────────────────────────────────────────────────────
# Per-query result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QueryEvalResult:
    query: str
    # Retrieval
    hybrid_recall_at_5: float
    keyword_recall_at_5: float
    hybrid_mrr: float
    keyword_mrr: float
    # Answer
    rouge_l_f1: float
    faithfulness_score: float
    groundedness_score: float
    # Latency (ms)
    total_latency_ms: float
    retrieval_latency_ms: float
    answer_latency_ms: float


@dataclass
class BenchmarkReport:
    query_results: List[QueryEvalResult] = field(default_factory=list)

    # Aggregate means
    mean_hybrid_recall_at_5: float = 0.0
    mean_keyword_recall_at_5: float = 0.0
    recall_improvement_pct: float = 0.0

    mean_hybrid_mrr: float = 0.0
    mean_keyword_mrr: float = 0.0

    mean_rouge_l: float = 0.0
    mean_faithfulness: float = 0.0
    mean_groundedness: float = 0.0

    mean_total_latency_ms: float = 0.0
    mean_retrieval_latency_ms: float = 0.0
    mean_answer_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0

    def compute_aggregates(self):
        if not self.query_results:
            return
        qr = self.query_results

        self.mean_hybrid_recall_at_5 = statistics.mean(r.hybrid_recall_at_5 for r in qr)
        self.mean_keyword_recall_at_5 = statistics.mean(r.keyword_recall_at_5 for r in qr)

        if self.mean_keyword_recall_at_5 > 0:
            self.recall_improvement_pct = (
                (self.mean_hybrid_recall_at_5 - self.mean_keyword_recall_at_5)
                / self.mean_keyword_recall_at_5
            ) * 100
        else:
            self.recall_improvement_pct = 0.0

        self.mean_hybrid_mrr = statistics.mean(r.hybrid_mrr for r in qr)
        self.mean_keyword_mrr = statistics.mean(r.keyword_mrr for r in qr)
        self.mean_rouge_l = statistics.mean(r.rouge_l_f1 for r in qr)
        self.mean_faithfulness = statistics.mean(r.faithfulness_score for r in qr)
        self.mean_groundedness = statistics.mean(r.groundedness_score for r in qr)
        self.mean_total_latency_ms = statistics.mean(r.total_latency_ms for r in qr)
        self.mean_retrieval_latency_ms = statistics.mean(r.retrieval_latency_ms for r in qr)
        self.mean_answer_latency_ms = statistics.mean(r.answer_latency_ms for r in qr)

        latencies = sorted(r.total_latency_ms for r in qr)
        p95_idx = min(int(len(latencies) * 0.95), len(latencies) - 1)
        self.p95_latency_ms = latencies[p95_idx]


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────────────────────────────────────

class Evaluator:
    """
    Runs the benchmark suite against the full pipeline and keyword baseline.
    """

    def __init__(self, retriever: HybridRetriever, pipeline):
        self.retriever = retriever
        self.pipeline = pipeline

    def run_benchmark(
        self,
        queries: List[Dict] | None = None,
        top_k: int = 5,
        progress_callback=None,
    ) -> BenchmarkReport:
        queries = queries or BENCHMARK_QUERIES
        report = BenchmarkReport()

        for i, item in enumerate(queries):
            q = item["query"]
            relevant = item["relevant_ids"]
            reference = item["reference_answer"]

            if progress_callback:
                progress_callback(i, len(queries), q)

            # Run full pipeline
            result: PipelineResult = self.pipeline.run(q)

            # Keyword-only baseline
            kw_docs, kw_ms = self.retriever.keyword_retrieve(q, top_k=top_k)

            # IDs
            hybrid_ids = [d.paper.paper_id for d in result.all_docs]
            kw_ids = [d.paper.paper_id for d in kw_docs]

            # Metrics
            h_recall = recall_at_k(hybrid_ids, relevant, top_k)
            k_recall = recall_at_k(kw_ids, relevant, top_k)
            h_mrr = reciprocal_rank(hybrid_ids, relevant)
            k_mrr = reciprocal_rank(kw_ids, relevant)
            rl = rouge_l_f1(result.final_answer, reference)

            faith = (
                result.verification.faithfulness_score
                if result.verification else 0.5
            )
            ground = (
                result.verification.groundedness_score
                if result.verification else 0.5
            )

            report.query_results.append(
                QueryEvalResult(
                    query=q,
                    hybrid_recall_at_5=h_recall,
                    keyword_recall_at_5=k_recall,
                    hybrid_mrr=h_mrr,
                    keyword_mrr=k_mrr,
                    rouge_l_f1=rl,
                    faithfulness_score=faith,
                    groundedness_score=ground,
                    total_latency_ms=result.total_latency_ms,
                    retrieval_latency_ms=result.retrieval_latency_ms,
                    answer_latency_ms=result.answer_latency_ms,
                )
            )

        report.compute_aggregates()
        return report
