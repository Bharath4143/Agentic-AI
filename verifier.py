"""
Verifier Module
===============
Checks generated answers along two dimensions:

  1. Faithfulness  — does the answer contain only claims supported by the
                     retrieved passages?
  2. Groundedness  — what fraction of sentences in the answer can be traced
                     back to at least one retrieved passage?

Uses sentence-level overlap heuristics when no API key is available.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import List

from agentic_rag.modules.retriever import RetrievedDoc


@dataclass
class VerificationResult:
    faithful: bool
    faithfulness_score: float      # 0.0 – 1.0
    groundedness_score: float      # fraction of answer sentences grounded
    warnings: List[str]
    latency_ms: float


class Verifier:
    """
    Validates generated answers against their source documents.
    """

    SYSTEM_PROMPT = """You are a strict factual verifier. Given an answer and a set of source
passages, score the answer on:
- faithfulness (0.0-1.0): are ALL claims in the answer supported by the passages?
- groundedness (0.0-1.0): what fraction of sentences reference passage content?

Respond with ONLY valid JSON:
{
  "faithfulness_score": <float>,
  "groundedness_score": <float>,
  "warnings": ["<any unsupported claim>", ...]
}"""

    FAITHFULNESS_THRESHOLD = 0.6

    def __init__(self):
        self._has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY"))

    def verify(
        self,
        answer: str,
        docs: List[RetrievedDoc],
        query: str = "",
    ) -> VerificationResult:
        t0 = time.perf_counter()

        if self._has_api_key:
            result = self._llm_verify(answer, docs, query)
        else:
            result = self._heuristic_verify(answer, docs)

        latency_ms = (time.perf_counter() - t0) * 1000
        result.latency_ms = latency_ms
        result.faithful = result.faithfulness_score >= self.FAITHFULNESS_THRESHOLD
        return result

    # ─────────────────────────────────────────────────────────────────────────

    def _llm_verify(
        self, answer: str, docs: List[RetrievedDoc], query: str
    ) -> VerificationResult:
        try:
            import anthropic
            import json

            passages = "\n\n".join(
                f"[{i+1}] {d.paper.title}:\n{d.paper.abstract}"
                for i, d in enumerate(docs[:5])
            )
            prompt = (
                f"Query: {query}\n\n"
                f"Answer:\n{answer}\n\n"
                f"Source Passages:\n{passages}"
            )

            client = anthropic.Anthropic()
            msg = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=256,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = msg.content[0].text
            raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("```").strip()
            data = json.loads(raw)

            return VerificationResult(
                faithful=False,  # filled in by caller
                faithfulness_score=float(data.get("faithfulness_score", 0.5)),
                groundedness_score=float(data.get("groundedness_score", 0.5)),
                warnings=data.get("warnings", []),
                latency_ms=0.0,
            )
        except Exception:
            return self._heuristic_verify(answer, docs)

    def _heuristic_verify(
        self, answer: str, docs: List[RetrievedDoc]
    ) -> VerificationResult:
        """
        Heuristic: check token overlap between answer sentences and passages.
        """
        passage_text = " ".join(
            f"{d.paper.title} {d.paper.abstract}" for d in docs
        ).lower()
        passage_tokens = set(re.findall(r"\b\w{4,}\b", passage_text))

        answer_sentences = [
            s.strip() for s in re.split(r"[.!?]+", answer) if len(s.strip()) > 20
        ]
        if not answer_sentences:
            return VerificationResult(
                faithful=True,
                faithfulness_score=1.0,
                groundedness_score=1.0,
                warnings=[],
                latency_ms=0.0,
            )

        grounded_count = 0
        warnings: List[str] = []

        for sent in answer_sentences:
            sent_tokens = set(re.findall(r"\b\w{4,}\b", sent.lower()))
            if not sent_tokens:
                continue
            overlap = len(sent_tokens & passage_tokens) / len(sent_tokens)
            if overlap >= 0.35:
                grounded_count += 1
            else:
                warnings.append(
                    f"Low overlap ({overlap:.0%}): '{sent[:60]}…'"
                    if len(sent) > 60
                    else f"Low overlap ({overlap:.0%}): '{sent}'"
                )

        groundedness = grounded_count / len(answer_sentences)
        # Faithfulness is a smoothed version of groundedness
        faithfulness = min(1.0, groundedness * 1.1)

        return VerificationResult(
            faithful=False,
            faithfulness_score=round(faithfulness, 3),
            groundedness_score=round(groundedness, 3),
            warnings=warnings[:5],
            latency_ms=0.0,
        )
