"""Embedding + reranking models behind small interfaces.

Real implementations use fastembed (ONNX on CPU — works well on Apple Silicon, no PyTorch).
Tests swap in the deterministic fakes at the bottom, so CI never downloads models.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass
class Sparse:
    indices: list[int]
    values: list[float]


class DenseEncoder(Protocol):
    dim: int
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class SparseEncoder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[Sparse]: ...
    def embed_query(self, text: str) -> Sparse: ...


class Reranker(Protocol):
    def score(self, query: str, docs: Sequence[str]) -> list[float]: ...


# ───────────── fastembed implementations ─────────────
class FastEmbedDense:
    def __init__(self, model_name: str, batch_size: int = 64):
        from fastembed import TextEmbedding
        self.model = TextEmbedding(model_name=model_name)
        self.batch_size = batch_size
        self.dim = len(next(iter(self.model.embed(["probe"]))))

    def embed_documents(self, texts):
        return [v.tolist() for v in self.model.embed(list(texts), batch_size=self.batch_size)]

    def embed_query(self, text):
        # bge-*-v1.5 works well without a query instruction; query_embed applies any model-specific handling
        return next(iter(self.model.query_embed(text))).tolist()


class FastEmbedBM25:
    """Qdrant/bm25: produces term-frequency sparse vectors; the IDF part is applied by Qdrant
    (collection created with Modifier.IDF), so scores are true BM25 over the live corpus."""

    def __init__(self, model_name: str = "Qdrant/bm25"):
        from fastembed import SparseTextEmbedding
        self.model = SparseTextEmbedding(model_name=model_name)

    def embed_documents(self, texts):
        return [Sparse(e.indices.tolist(), e.values.tolist()) for e in self.model.embed(list(texts))]

    def embed_query(self, text):
        e = next(iter(self.model.query_embed(text)))
        return Sparse(e.indices.tolist(), e.values.tolist())


class FastEmbedReranker:
    def __init__(self, model_name: str):
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        self.model = TextCrossEncoder(model_name=model_name)

    def score(self, query, docs):
        return [float(s) for s in self.model.rerank(query, list(docs))]


# ───────────── deterministic fakes (tests / offline dev) ─────────────
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "of", "and", "or", "in", "for", "to", "with", "is", "are", "on", "my", "what", "which"}


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


def _h(token: str, mod: int) -> int:
    return int(hashlib.md5(token.encode()).hexdigest(), 16) % mod


class HashDense:
    """Bag-of-words hashed into a fixed-size unit vector. Not semantic, but deterministic."""

    def __init__(self, dim: int = 128):
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for t in _tokens(text):
            v[_h(t, self.dim)] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class HashSparse:
    def _sp(self, text: str) -> Sparse:
        c = Counter(_h(t, 2**20) for t in _tokens(text))
        idx = sorted(c)
        return Sparse(idx, [float(c[i]) for i in idx])

    def embed_documents(self, texts):
        return [self._sp(t) for t in texts]

    def embed_query(self, text):
        q = self._sp(text)
        return Sparse(q.indices, [1.0] * len(q.indices))


class OverlapReranker:
    """Scores by query-term overlap ratio — enough to test that reranking reorders."""

    def score(self, query, docs):
        q = set(_tokens(query))
        return [len(q & set(_tokens(d))) / (len(q) or 1) for d in docs]
