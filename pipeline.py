"""
ReAct Pipeline
==============
Orchestrates the full agentic RAG loop:

  Plan → [Retrieve → Generate → Verify] × hops → Synthesize

Each "hop" is a Thought / Action / Observation cycle (ReAct-style).
The final answer is synthesised from all collected evidence.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional

from agentic_rag.modules.planner import Planner, QueryPlan
from agentic_rag.modules.retriever import HybridRetriever, RetrievedDoc
from agentic_rag.modules.verifier import Verifier, VerificationResult


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ReActStep:
    hop: int
    sub_query: str
    thought: str
    action: str          # "retrieve" | "synthesize" | "done"
    observation: str     # retrieved doc titles + snippets
    docs: List[RetrievedDoc] = field(default_factory=list)
    retrieval_ms: float = 0.0


@dataclass
class PipelineResult:
    query: str
    plan: QueryPlan
    steps: List[ReActStep]
    final_answer: str
    verification: Optional[VerificationResult]
    all_docs: List[RetrievedDoc]
    total_latency_ms: float
    answer_latency_ms: float     # time spent in generation only
    retrieval_latency_ms: float  # time spent in retrieval only


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class ReActPipeline:
    """
    Multi-hop agentic RAG pipeline.

    Parameters
    ----------
    retriever : HybridRetriever
    top_k : int
        Documents retrieved per hop.
    max_hops : int
        Maximum retrieval hops (caps at len(sub_queries)).
    cache : bool
        Cache identical query embeddings for repeated calls.
    """

    SYNTH_SYSTEM = """You are an expert NLP research assistant. Given a question and
retrieved passages from research papers, write a comprehensive, accurate answer.
Cite papers by their titles. Be precise and technical. Do not hallucinate."""

    SYNTH_FALLBACK = (
        "Based on the retrieved papers, here is a synthesis:\n\n{context}\n\n"
        "The above papers collectively address: {query}"
    )

    def __init__(
        self,
        retriever: HybridRetriever,
        top_k: int = 5,
        max_hops: int = 4,
        cache: bool = True,
    ):
        self.retriever = retriever
        self.top_k = top_k
        self.max_hops = max_hops
        self.planner = Planner()
        self.verifier = Verifier()
        self._has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
        self._cache: dict = {} if cache else None

    # ─────────────────────────────────────────────────────────────────────────

    def run(self, query: str) -> PipelineResult:
        t_total = time.perf_counter()

        # ── 1. Plan ───────────────────────────────────────────────────────────
        plan = self.planner.plan(query)
        sub_queries = plan.sub_queries[: self.max_hops]

        steps: List[ReActStep] = []
        all_docs: List[RetrievedDoc] = []
        seen_ids: set = set()
        total_retrieval_ms = 0.0

        # ── 2. Retrieve loop ──────────────────────────────────────────────────
        for hop, sub_q in enumerate(sub_queries, start=1):
            thought = (
                f"I need to find papers about: '{sub_q}'. "
                f"I will retrieve the top-{self.top_k} documents."
            )

            # Cache lookup
            cache_key = f"{sub_q}|{self.top_k}"
            if self._cache is not None and cache_key in self._cache:
                docs, r_ms = self._cache[cache_key]
            else:
                docs, r_ms = self.retriever.retrieve(
                    sub_q, top_k=self.top_k, exclude_ids=list(seen_ids)
                )
                if self._cache is not None:
                    self._cache[cache_key] = (docs, r_ms)

            total_retrieval_ms += r_ms

            for d in docs:
                if d.paper.paper_id not in seen_ids:
                    seen_ids.add(d.paper.paper_id)
                    all_docs.append(d)

            observation = self._format_observation(docs)
            action = "retrieve" if hop < len(sub_queries) else "synthesize"

            steps.append(
                ReActStep(
                    hop=hop,
                    sub_query=sub_q,
                    thought=thought,
                    action=action,
                    observation=observation,
                    docs=docs,
                    retrieval_ms=r_ms,
                )
            )

        # ── 3. Synthesize ─────────────────────────────────────────────────────
        t_ans = time.perf_counter()
        final_answer = self._synthesize(query, all_docs)
        answer_latency_ms = (time.perf_counter() - t_ans) * 1000

        # ── 4. Verify ─────────────────────────────────────────────────────────
        verification = self.verifier.verify(final_answer, all_docs, query)

        total_latency_ms = (time.perf_counter() - t_total) * 1000

        return PipelineResult(
            query=query,
            plan=plan,
            steps=steps,
            final_answer=final_answer,
            verification=verification,
            all_docs=all_docs,
            total_latency_ms=total_latency_ms,
            answer_latency_ms=answer_latency_ms,
            retrieval_latency_ms=total_retrieval_ms,
        )

    # ─────────────────────────────────────────────────────────────────────────

    def _synthesize(self, query: str, docs: List[RetrievedDoc]) -> str:
        if self._has_api_key:
            return self._llm_synthesize(query, docs)
        return self._fallback_synthesize(query, docs)

    def _llm_synthesize(self, query: str, docs: List[RetrievedDoc]) -> str:
        try:
            import anthropic

            context = "\n\n".join(
                f"[{i+1}] **{d.paper.title}** ({d.paper.year})\n"
                f"Authors: {', '.join(d.paper.authors[:3])}\n"
                f"Abstract: {d.paper.abstract}"
                for i, d in enumerate(docs[:8])
            )
            prompt = f"Question: {query}\n\nRetrieved Papers:\n{context}"

            client = anthropic.Anthropic()
            msg = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                system=self.SYNTH_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text
        except Exception as e:
            return self._fallback_synthesize(query, docs)

    def _fallback_synthesize(self, query: str, docs: List[RetrievedDoc]) -> str:
        """Template-based synthesis when LLM is unavailable."""
        lines = [f"Synthesized answer for: {query}\n"]
        lines.append("Based on the following retrieved papers:\n")
        for i, d in enumerate(docs[:6], start=1):
            lines.append(
                f"  [{i}] {d.paper.title} ({d.paper.year}) "
                f"— hybrid score {d.hybrid_score:.3f}"
            )
            # Include first 2 sentences of abstract
            sentences = re.split(r"(?<=[.!?])\s+", d.paper.abstract)
            lines.append(f"      {' '.join(sentences[:2])}\n")

        lines.append(
            "\nThese papers collectively address the query through the lens of "
            "retrieval-augmented generation, language model scaling, and reasoning "
            "techniques in NLP research."
        )
        return "\n".join(lines)

    @staticmethod
    def _format_observation(docs: List[RetrievedDoc]) -> str:
        parts = []
        for d in docs[:3]:
            snippet = d.paper.abstract[:120].rstrip() + "…"
            parts.append(
                f"• [{d.paper.paper_id}] {d.paper.title} "
                f"(score={d.hybrid_score:.3f}) — {snippet}"
            )
        return "\n".join(parts)
