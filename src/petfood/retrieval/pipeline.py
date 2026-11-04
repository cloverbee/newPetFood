"""Hybrid retrieval: dense + BM25 → Reciprocal Rank Fusion → cross-encoder rerank → top-k.

Fusion is done client-side (not Qdrant's built-in RRF) on purpose: we keep each retriever's
rank for every candidate, which the Phase 6 evaluation uses for ablations
(dense-only vs BM25-only vs hybrid vs hybrid+rerank) and which is handy when debugging.
"""
from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from petfood.retrieval.embeddings import Reranker
from petfood.retrieval.store import ChunkStore, Hit, SearchFilters


def rrf(rankings: dict[str, Sequence[str]], k: int = 60,
        weights: dict[str, float] | None = None) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion. rankings: retriever name -> ids in rank order (best first).

    score(d) = Σ_r  w_r / (k + rank_r(d)),  rank starting at 1.
    k=60 is the value from the original paper (Cormack et al., 2009); it damps the influence of
    the very top ranks so one retriever can't dominate.
    """
    weights = weights or {}
    scores: dict[str, float] = {}
    for name, ids in rankings.items():
        w = weights.get(name, 1.0)
        for rank, doc_id in enumerate(ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


@dataclass
class Candidate:
    id: str
    payload: dict
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rrf_score: float = 0.0
    rrf_rank: int | None = None
    rerank_score: float | None = None
    final_rank: int | None = None

    @property
    def text(self) -> str:
        return self.payload.get("text", "")


@dataclass
class RetrievalResult:
    query: str
    results: list[Candidate]
    candidates: list[Candidate]                 # everything that went into the reranker
    timings_ms: dict[str, float] = field(default_factory=dict)


class HybridRetriever:
    def __init__(self, store: ChunkStore, reranker: Reranker | None = None,
                 k_dense: int = 30, k_sparse: int = 30, k_rerank: int = 30,
                 rrf_k: int = 60, low_value_penalty: float = 0.5):
        self.store = store
        self.reranker = reranker
        self.k_dense, self.k_sparse, self.k_rerank = k_dense, k_sparse, k_rerank
        self.rrf_k = rrf_k
        self.low_value_penalty = low_value_penalty   # multiplier on rerank score for Methods chunks

    def retrieve(self, query: str, top_k: int = 5, filters: SearchFilters | None = None,
                 mode: str = "hybrid_rerank", max_per_doc: int | None = 2) -> RetrievalResult:
        """mode: dense | sparse | hybrid | hybrid_rerank  (the first three exist for evaluation)."""
        t: dict[str, float] = {}
        t0 = time.perf_counter()
        dense: list[Hit] = []
        sparse: list[Hit] = []
        if mode != "sparse":
            dense = self.store.search_dense(query, self.k_dense, filters)
        t["dense"] = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        if mode != "dense":
            sparse = self.store.search_sparse(query, self.k_sparse, filters)
        t["sparse"] = (time.perf_counter() - t1) * 1000

        by_id: dict[str, Candidate] = {}
        for rank, h in enumerate(dense, start=1):
            by_id.setdefault(h.id, Candidate(h.id, h.payload)).dense_rank = rank
        for rank, h in enumerate(sparse, start=1):
            by_id.setdefault(h.id, Candidate(h.id, h.payload)).sparse_rank = rank

        fused = rrf({"dense": [h.id for h in dense], "sparse": [h.id for h in sparse]}, k=self.rrf_k)
        ordered: list[Candidate] = []
        for i, (cid, score) in enumerate(fused, start=1):
            c = by_id[cid]
            c.rrf_score, c.rrf_rank = score, i
            ordered.append(c)
        candidates = ordered[: self.k_rerank]

        t2 = time.perf_counter()
        if mode == "hybrid_rerank" and self.reranker and candidates:
            scores = self.reranker.score(query, [c.text for c in candidates])
            for c, s in zip(candidates, scores):
                if c.payload.get("low_value"):   # push Methods chunks down (scores may be negative logits)
                    s = s * self.low_value_penalty if s > 0 else s / self.low_value_penalty
                c.rerank_score = s
            ranked = sorted(candidates, key=lambda c: -(c.rerank_score or 0.0))
        else:
            ranked = list(candidates)
        t["rerank"] = (time.perf_counter() - t2) * 1000

        results = (dedupe_by_doc(ranked, max_per_doc) if max_per_doc else ranked)[:top_k]
        for i, c in enumerate(results, start=1):
            c.final_rank = i
        t["total"] = (time.perf_counter() - t0) * 1000
        return RetrievalResult(query=query, results=results, candidates=candidates, timings_ms=t)


def dedupe_by_doc(cands: list[Candidate], max_per_doc: int = 2) -> list[Candidate]:
    """Keep at most N chunks per paper so the LLM sees evidence from several studies."""
    seen: dict[tuple, int] = {}
    out = []
    for c in cands:
        key = (c.payload.get("doc_type"), c.payload.get("doc_id"))
        if seen.get(key, 0) < max_per_doc:
            out.append(c)
            seen[key] = seen.get(key, 0) + 1
    return out
