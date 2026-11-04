"""Runs against a real Postgres. Enable with: PETFOOD_TEST_DB=postgresql://... pytest -m integration"""
import json
import os
from pathlib import Path

import pytest

from petfood import db
from petfood.ingest import europepmc, openpetfoodfacts

FIX = Path(__file__).parent / "fixtures"
URL = os.environ.get("PETFOOD_TEST_DB")
pytestmark = [pytest.mark.integration,
              pytest.mark.skipif(not URL, reason="set PETFOOD_TEST_DB to run")]


@pytest.fixture
def conn():
    with db.connect(URL) as c:
        db.apply_schema(c)
        c.execute("TRUNCATE papers, paper_sections, products, chunks, ingestion_runs RESTART IDENTITY CASCADE")
        c.commit()
        yield c


def test_paper_upsert_is_idempotent(conn):
    rec = json.loads((FIX / "epmc_search_page.json").read_text())["resultList"]["result"][0]
    paper = europepmc.record_to_paper(rec)
    europepmc.attach_full_text(paper, (FIX / "sample_jats.xml").read_bytes(), "papers/x.xml")
    id1 = db.upsert_paper(conn, paper)
    id2 = db.upsert_paper(conn, paper)
    conn.commit()
    assert id1 == id2
    assert conn.execute("SELECT count(*) n FROM paper_sections").fetchone()["n"] == 8
    assert db.paper_exists(conn, "europepmc", "PMC0000001")
    row = conn.execute("SELECT species, authors FROM papers").fetchone()
    assert row["species"] == ["cat"] and row["authors"] == ["Doe J", "Roe R"]


def test_product_upsert_and_allergen_filter(conn):
    rec = json.loads((FIX / "opff_product.json").read_text())
    db.upsert_product(conn, openpetfoodfacts.record_to_product(rec))
    rec2 = dict(rec, code="2", product_name="Duck Grain-Free Cat Food",
                ingredients_text="Duck, peas, potato")
    db.upsert_product(conn, openpetfoodfacts.record_to_product(rec2))
    conn.commit()
    # the kind of hard filter the recommender will use: "cat, no chicken"
    rows = conn.execute(
        "SELECT name FROM products WHERE species='cat' AND NOT ('chicken' = ANY(allergen_tags))"
    ).fetchall()
    assert [r["name"] for r in rows] == ["Duck Grain-Free Cat Food"]


def test_runs_and_stats(conn):
    rid = db.start_run(conn, "test", {"a": 1})
    db.finish_run(conn, rid, "ok", 3, 2, 1)
    s = db.stats(conn)
    assert s["recent_runs"][0]["status"] == "ok"


def test_build_index_from_postgres(conn):
    from qdrant_client import QdrantClient

    from petfood.retrieval.embeddings import HashDense, HashSparse
    from petfood.retrieval.indexer import build_index
    from petfood.retrieval.store import ChunkStore, SearchFilters

    rec = json.loads((FIX / "epmc_search_page.json").read_text())["resultList"]["result"][0]
    paper = europepmc.record_to_paper(rec)
    europepmc.attach_full_text(paper, (FIX / "sample_jats.xml").read_bytes(), "papers/x.xml")
    db.upsert_paper(conn, paper)
    db.upsert_product(conn, openpetfoodfacts.record_to_product(json.loads((FIX / "opff_product.json").read_text())))
    conn.commit()

    store = ChunkStore(QdrantClient(":memory:"), "it", HashDense(), HashSparse())
    stats = build_index(conn, store)
    assert stats["paper"] >= 1 and stats["product"] == 1
    assert stats["qdrant_points"] == stats["paper"] + stats["product"]
    n_pg = conn.execute("SELECT count(*) n FROM chunks").fetchone()["n"]
    assert n_pg == stats["qdrant_points"]
    # rebuild is idempotent
    assert build_index(conn, store)["qdrant_points"] == stats["qdrant_points"]
    hits = store.search_sparse("protein weight", 5, SearchFilters(doc_type="paper", species="cat"))
    assert hits and hits[0].payload["source_id"] == "PMC0000001"
