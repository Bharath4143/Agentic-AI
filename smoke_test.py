#!/usr/bin/env python3
"""
smoke_test.py — end-to-end validation of the agentic RAG system.
Run from the repo root: python smoke_test.py
"""

import sys
import os

# Ensure package is importable from the project root
sys.path.insert(0, os.path.dirname(__file__))

from rich.console import Console
from rich.rule import Rule

console = Console()

def section(title: str):
    console.print()
    console.print(Rule(f"[bold cyan]{title}"))


# ── 1. Corpus ─────────────────────────────────────────────────────────────────
section("1 · Corpus")
from agentic_rag.corpus.mock_arxiv import get_all_papers, get_paper_by_id
papers = get_all_papers()
console.print(f"  Loaded {len(papers)} papers")
p = get_paper_by_id("2210.11610")
console.print(f"  Lookup test: {p.title}")
assert len(papers) >= 10
assert p is not None

# ── 2. Retriever ──────────────────────────────────────────────────────────────
section("2 · Hybrid Retriever")
from agentic_rag.modules.retriever import HybridRetriever
with console.status("Building FAISS index…"):
    retriever = HybridRetriever(alpha=0.6)

docs, ms = retriever.retrieve("dense retrieval open-domain QA", top_k=5)
console.print(f"  Retrieved {len(docs)} docs in {ms:.1f} ms")
for d in docs[:3]:
    console.print(f"    [{d.rank}] {d.paper.title[:55]} — hybrid={d.hybrid_score:.4f}")

kw_docs, kw_ms = retriever.keyword_retrieve("transformer attention", top_k=3)
console.print(f"  Keyword baseline: {len(kw_docs)} docs in {kw_ms:.1f} ms")

# ── 3. Planner ────────────────────────────────────────────────────────────────
section("3 · Planner")
from agentic_rag.modules.planner import Planner
planner = Planner()
plan = planner.plan(
    "How does dense retrieval compare to sparse search, and how does RAG use it?"
)
console.print(f"  Sub-queries ({len(plan.sub_queries)}): {plan.sub_queries}")
console.print(f"  Reasoning: {plan.reasoning}")
assert len(plan.sub_queries) >= 1

# ── 4. Verifier ───────────────────────────────────────────────────────────────
section("4 · Verifier")
from agentic_rag.modules.verifier import Verifier
verifier = Verifier()
answer = (
    "Dense retrieval methods like DPR use dual-encoder BERT models to embed queries "
    "and passages into a shared vector space, substantially outperforming BM25 on "
    "open-domain QA. RAG combines this dense retrieval with a seq2seq generator."
)
vr = verifier.verify(answer, docs[:5], "dense retrieval vs sparse")
console.print(f"  Faithful: {vr.faithful}  faith={vr.faithfulness_score:.3f}  ground={vr.groundedness_score:.3f}")
if vr.warnings:
    console.print(f"  Warnings: {vr.warnings[:2]}")

# ── 5. Pipeline ───────────────────────────────────────────────────────────────
section("5 · ReAct Pipeline")
from agentic_rag.modules.pipeline import ReActPipeline
pipeline = ReActPipeline(retriever, top_k=5, max_hops=3)
result = pipeline.run("What is retrieval-augmented generation and how does it work?")
console.print(f"  Steps: {len(result.steps)}  |  All docs: {len(result.all_docs)}")
console.print(f"  Total latency: {result.total_latency_ms:.1f} ms")
console.print(f"  Answer snippet: {result.final_answer[:120]}…")
assert len(result.all_docs) > 0

# ── 6. Evaluator ─────────────────────────────────────────────────────────────
section("6 · Evaluator")
from agentic_rag.eval.evaluator import Evaluator, BENCHMARK_QUERIES

# Run on just 2 queries to keep smoke test fast
evaluator = Evaluator(retriever, pipeline)
report = evaluator.run_benchmark(queries=BENCHMARK_QUERIES[:2])
report.compute_aggregates()
console.print(f"  Hybrid Recall@5:  {report.mean_hybrid_recall_at_5:.3f}")
console.print(f"  Keyword Recall@5: {report.mean_keyword_recall_at_5:.3f}")
console.print(f"  Recall improvement: {report.recall_improvement_pct:+.1f}%")
console.print(f"  Mean faithfulness: {report.mean_faithfulness:.3f}")
console.print(f"  Mean latency: {report.mean_total_latency_ms:.1f} ms")

# ── Done ──────────────────────────────────────────────────────────────────────
console.print()
console.print(Rule("[bold green]✓ All smoke tests passed"))
console.print()
