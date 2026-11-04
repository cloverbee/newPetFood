"""Postgres (source of truth) → chunks table + Qdrant (derived index)."""
from __future__ import annotations

import json
import logging

import psycopg

from petfood.retrieval.chunking import Chunk, chunk_paper, chunk_product
from petfood.retrieval.store import ChunkStore

log = logging.getLogger(__name__)


def load_paper_chunks(conn: psycopg.Connection) -> list[Chunk]:
    papers = conn.execute("SELECT * FROM papers ORDER BY id").fetchall()
    secs = conn.execute("SELECT * FROM paper_sections ORDER BY paper_id, ord").fetchall()
    by_paper: dict[int, list[dict]] = {}
    for s in secs:
        by_paper.setdefault(s["paper_id"], []).append(s)
    chunks: list[Chunk] = []
    for p in papers:
        sections = by_paper.get(p["id"])
        if not sections and p.get("abstract"):
            sections = [{"section_type": "abstract", "heading": "Abstract", "text": p["abstract"]}]
        chunks.extend(chunk_paper(p, sections or []))
    return chunks


def load_product_chunks(conn: psycopg.Connection) -> list[Chunk]:
    return [chunk_product(p) for p in conn.execute("SELECT * FROM products ORDER BY id").fetchall()]


def save_chunks(conn: psycopg.Connection, chunks: list[Chunk], doc_type: str) -> None:
    """Replace all chunks of a doc_type in Postgres (cheap at this scale, always consistent)."""
    conn.execute("DELETE FROM chunks WHERE doc_type = %s", (doc_type,))
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks (id, doc_type, doc_id, ord, text, token_count, metadata) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            [(c.id, c.doc_type, c.doc_id, c.ord, c.text, c.token_count,
              json.dumps({k: v for k, v in c.payload.items() if k != "body"}, default=str))
             for c in chunks],
        )
    conn.commit()


def build_index(conn: psycopg.Connection, store: ChunkStore, doc_types=("paper", "product"),
                recreate: bool = False) -> dict[str, int]:
    """Full rebuild per doc type. Deterministic chunk ids make re-runs idempotent; stale points
    of a doc type are cleared first so deleted/changed docs don't linger."""
    store.ensure_collection(recreate=recreate)
    stats: dict[str, int] = {}
    loaders = {"paper": load_paper_chunks, "product": load_product_chunks}
    for dt in doc_types:
        chunks = loaders[dt](conn)
        save_chunks(conn, chunks, dt)
        if not recreate:
            from qdrant_client import models
            store.client.delete(store.collection, points_selector=models.FilterSelector(filter=models.Filter(
                must=[models.FieldCondition(key="doc_type", match=models.MatchValue(value=dt))])))
        log.info("indexing %d %s chunks", len(chunks), dt)
        stats[dt] = store.upsert(chunks)
    stats["qdrant_points"] = store.count()
    return stats
