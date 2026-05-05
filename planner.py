"""
Planner Module
==============
Decomposes a complex research query into a sequence of sub-queries
that can each be answered by a single retrieval step.

Uses a lightweight rule-based fallback when no API key is available
so the system can run entirely offline.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import List


@dataclass
class QueryPlan:
    original_query: str
    sub_queries: List[str]
    reasoning: str
    latency_ms: float


class Planner:
    """
    Decomposes a natural-language research question into sub-queries.

    Strategy
    --------
    1. If ANTHROPIC_API_KEY is present, calls Claude to generate a structured
       JSON plan.
    2. Otherwise falls back to a heuristic decomposer that splits on question
       words and connectives.
    """

    SYSTEM_PROMPT = """You are a research query planner. Given a complex research question
about NLP/ML papers, decompose it into 2-4 focused sub-queries.
Each sub-query should be answerable by reading 1-2 papers.

Respond with ONLY valid JSON in this exact format:
{
  "reasoning": "<one sentence on why you split it this way>",
  "sub_queries": ["<sub-query 1>", "<sub-query 2>", ...]
}"""

    def __init__(self):
        self._has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY"))

    def plan(self, query: str) -> QueryPlan:
        t0 = time.perf_counter()

        if self._has_api_key:
            plan_data = self._llm_plan(query)
        else:
            plan_data = self._heuristic_plan(query)

        latency_ms = (time.perf_counter() - t0) * 1000
        return QueryPlan(
            original_query=query,
            sub_queries=plan_data["sub_queries"],
            reasoning=plan_data["reasoning"],
            latency_ms=latency_ms,
        )

    # ─────────────────────────────────────────────────────────────────────────

    def _llm_plan(self, query: str) -> dict:
        try:
            import anthropic

            client = anthropic.Anthropic()
            msg = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=512,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": query}],
            )
            raw = msg.content[0].text
            # Strip markdown fences if present
            raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("```").strip()
            data = json.loads(raw)
            # Validate
            assert "sub_queries" in data and isinstance(data["sub_queries"], list)
            assert "reasoning" in data
            return data
        except Exception as e:
            # Graceful fallback
            return self._heuristic_plan(query)

    def _heuristic_plan(self, query: str) -> dict:
        """Rule-based query decomposition."""
        q = query.strip()
        sub_queries: List[str] = []

        # Split on "and", "compared to", "vs", "how does X relate to Y"
        splits = re.split(
            r"\band\b|\bcompared to\b|\bvs\.?\b|\bversus\b|\brelation(?:ship)? between\b",
            q,
            flags=re.IGNORECASE,
        )
        splits = [s.strip(" ?,") for s in splits if len(s.strip()) > 10]

        if len(splits) >= 2:
            sub_queries = splits[:3]
            reasoning = "Split on conjunctions / comparative keywords."
        elif "?" in q and len(q) > 80:
            # Split long question in half by words
            words = q.split()
            mid = len(words) // 2
            sub_queries = [
                " ".join(words[:mid]).rstrip(",?"),
                " ".join(words[mid:]).lstrip(",?"),
            ]
            reasoning = "Halved long query by word count."
        else:
            # No decomposition needed
            sub_queries = [q]
            reasoning = "Query is atomic; no decomposition needed."

        # Always include the original as a synthesis step if we split
        if len(sub_queries) > 1:
            sub_queries.append(q)

        return {"sub_queries": sub_queries, "reasoning": reasoning}
