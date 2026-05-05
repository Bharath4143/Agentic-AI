"""
Hybrid Retriever
================
Combines:
  1. Dense retrieval  — TF-IDF + Truncated SVD (LSA) embeddings + FAISS inner-product index
                        (simulates Sentence-BERT dense retrieval without network access;
                         swap _LocalDenseEncoder for SentenceTransformer when online)
  2. Sparse retrieval — BM25-style TF-IDF keyword search (scikit-learn)

Final score = α * dense_score + (1-α) * sparse_score
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from agentic_rag.corpus.mock_arxiv import Paper, get_all_papers


@dataclass
class RetrievedDoc:
    paper: Paper
    dense_score: float
    sparse_score: float
    hybrid_score: float
    rank: int


class _LocalDenseEncoder:
    """
    Local dense encoder using TF-IDF + Truncated SVD (Latent Semantic Analysis).

    This faithfully simulates Sentence-BERT style dense retrieval:
      - TF-IDF converts text to high-dimensional sparse vectors
      - TruncatedSVD projects to a dense 128-d semantic space
      - Vectors are L2-normalised so dot-product == cosine similarity

    In a production setting, swap this for:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
    """

    EMBEDDING_DIM = 128

    def __init__(self, corpus: List[str]):
        self._tfidf = TfidfVectorizer(
            ngram_range=(1, 3), max_features=30_000, sublinear_tf=True
        )
        tfidf_matrix = self._tfidf.fit_transform(corpus)
        n_components = min(self.EMBEDDING_DIM, tfidf_matrix.shape[1] - 1)
        self._svd = TruncatedSVD(n_components=n_components, random_state=42)
        self._svd.fit(tfidf_matrix)

    def encode(self, texts: List[str]) -> np.ndarray:
        tfidf = self._tfidf.transform(texts)
        dense = self._svd.transform(tfidf)
        return normalize(dense, norm="l2").astype("float32")


class HybridRetriever:
    """
    Local-dense (LSA) + TF-IDF hybrid retriever over the mock arXiv corpus.

    Parameters
    ----------
    alpha : float
        Weight given to dense scores (0 = pure sparse, 1 = pure dense).

    Notes
    -----
    The dense encoder uses TF-IDF + SVD (LSA) to produce 128-d embeddings
    without requiring any external model downloads.  To upgrade to
    Sentence-BERT, replace _LocalDenseEncoder with SentenceTransformer
    and call .encode() the same way.
    """

    def __init__(self, alpha: float = 0.6):
        self.alpha = alpha
        self.papers: List[Paper] = get_all_papers()
        self._corpus_texts = [
            f"{p.title}. {p.abstract}" for p in self.papers
        ]

        # ── Dense index ──────────────────────────────────────────────────────
        self._encoder = _LocalDenseEncoder(self._corpus_texts)
        dense_matrix = self._encoder.encode(self._corpus_texts)

        if FAISS_AVAILABLE:
            dim = dense_matrix.shape[1]
            self._faiss_index = faiss.IndexFlatIP(dim)
            self._faiss_index.add(dense_matrix)
            self._dense_matrix = None
        else:
            self._faiss_index = None
            self._dense_matrix = dense_matrix

        # ── Sparse index ─────────────────────────────────────────────────────
        self._tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=20_000)
        self._tfidf_matrix = self._tfidf.fit_transform(self._corpus_texts)

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        exclude_ids: List[str] | None = None,
    ) -> Tuple[List[RetrievedDoc], float]:
        """
        Retrieve top-k documents for *query*.

        Returns
        -------
        docs : list of RetrievedDoc
        latency_ms : float
        """
        t0 = time.perf_counter()

        dense_scores = self._dense_scores(query)
        sparse_scores = self._sparse_scores(query)

        # Normalise to [0, 1]
        dense_norm = self._minmax(dense_scores)
        sparse_norm = self._minmax(sparse_scores)

        hybrid = self.alpha * dense_norm + (1.0 - self.alpha) * sparse_norm

        exclude = set(exclude_ids or [])
        ranked = sorted(
            [
                (i, hybrid[i], dense_norm[i], sparse_norm[i])
                for i in range(len(self.papers))
                if self.papers[i].paper_id not in exclude
            ],
            key=lambda x: x[1],
            reverse=True,
        )

        results: List[RetrievedDoc] = []
        for rank, (idx, hy, dn, sn) in enumerate(ranked[:top_k], start=1):
            results.append(
                RetrievedDoc(
                    paper=self.papers[idx],
                    dense_score=float(dn),
                    sparse_score=float(sn),
                    hybrid_score=float(hy),
                    rank=rank,
                )
            )

        latency_ms = (time.perf_counter() - t0) * 1000
        return results, latency_ms

    def keyword_retrieve(
        self, query: str, top_k: int = 5, exclude_ids: List[str] | None = None
    ) -> Tuple[List[RetrievedDoc], float]:
        """Sparse-only retrieval (BM25-like TF-IDF). Used for baseline comparison."""
        t0 = time.perf_counter()
        sparse_scores = self._sparse_scores(query)
        sparse_norm = self._minmax(sparse_scores)

        exclude = set(exclude_ids or [])
        ranked = sorted(
            [
                (i, sparse_norm[i])
                for i in range(len(self.papers))
                if self.papers[i].paper_id not in exclude
            ],
            key=lambda x: x[1],
            reverse=True,
        )

        results: List[RetrievedDoc] = []
        for rank, (idx, sn) in enumerate(ranked[:top_k], start=1):
            results.append(
                RetrievedDoc(
                    paper=self.papers[idx],
                    dense_score=0.0,
                    sparse_score=float(sn),
                    hybrid_score=float(sn),
                    rank=rank,
                )
            )

        latency_ms = (time.perf_counter() - t0) * 1000
        return results, latency_ms

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _dense_scores(self, query: str) -> np.ndarray:
        q_emb = self._encoder.encode([query])  # shape (1, dim)

        if FAISS_AVAILABLE and self._faiss_index is not None:
            scores, indices = self._faiss_index.search(q_emb, len(self.papers))
            out = np.zeros(len(self.papers))
            for s, i in zip(scores[0], indices[0]):
                out[i] = s
            return out
        else:
            return (q_emb @ self._dense_matrix.T).flatten()

    def _sparse_scores(self, query: str) -> np.ndarray:
        q_vec = self._tfidf.transform([query])
        return cosine_similarity(q_vec, self._tfidf_matrix).flatten()

    @staticmethod
    def _minmax(arr: np.ndarray) -> np.ndarray:
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-9:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
