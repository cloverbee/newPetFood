"""Postgres access layer (psycopg 3). Plain SQL, no ORM — easy to read and explain."""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

from petfood.config import get_settings
from petfood.models import Paper, Product

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "db" / "init" / "001_schema.sql"


@contextmanager
def connect(url: str | None = None):
    with psycopg.connect(url or get_settings().database_url, row_factory=dict_row) as conn:
        yield conn


def apply_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA_PATH.read_text())
    conn.commit()


# ───────────── papers ─────────────
_PAPER_UPSERT = """
INSERT INTO papers (source, source_id, pmid, doi, title, abstract, authors, journal, pub_year,
                    received_year, license, species, topics, s3_key, fulltext_available)
VALUES (%(source)s, %(source_id)s, %(pmid)s, %(doi)s, %(title)s, %(abstract)s, %(authors)s,
        %(journal)s, %(pub_year)s, %(received_year)s, %(license)s, %(species)s, %(topics)s, %(s3_key)s,
        %(fulltext_available)s)
ON CONFLICT (source, source_id) DO UPDATE SET
    pmid = EXCLUDED.pmid, doi = EXCLUDED.doi, title = EXCLUDED.title,
    abstract = EXCLUDED.abstract, authors = EXCLUDED.authors, journal = EXCLUDED.journal,
    pub_year = EXCLUDED.pub_year,
    received_year = COALESCE(EXCLUDED.received_year, papers.received_year),
    license = EXCLUDED.license, species = EXCLUDED.species,
    topics = EXCLUDED.topics, s3_key = COALESCE(EXCLUDED.s3_key, papers.s3_key),
    fulltext_available = EXCLUDED.fulltext_available OR papers.fulltext_available,
    updated_at = now()
RETURNING id
"""


def upsert_paper(conn: psycopg.Connection, paper: Paper) -> int:
    params = paper.model_dump(exclude={"sections"})
    params["authors"] = json.dumps(paper.authors)
    paper_id = conn.execute(_PAPER_UPSERT, params).fetchone()["id"]
    if paper.sections:
        conn.execute("DELETE FROM paper_sections WHERE paper_id = %s", (paper_id,))
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO paper_sections (paper_id, ord, section_type, heading, text) "
                "VALUES (%s, %s, %s, %s, %s)",
                [(paper_id, s.ord, s.section_type, s.heading, s.text) for s in paper.sections],
            )
    return paper_id


def paper_exists(conn: psycopg.Connection, source: str, source_id: str) -> bool:
    row = conn.execute(
        "SELECT fulltext_available FROM papers WHERE source=%s AND source_id=%s", (source, source_id)
    ).fetchone()
    return bool(row and row["fulltext_available"])


# ───────────── products ─────────────
_PRODUCT_COLS = [
    "source", "source_id", "name", "brand", "species", "life_stage", "food_form",
    "ingredients_text", "ingredients", "protein_sources", "allergen_tags", "grain_free",
    "protein_pct", "fat_pct", "fiber_pct", "moisture_pct", "kcal_per_100g", "labels",
    "url", "image_url", "raw",
]
_PRODUCT_UPSERT = (
    f"INSERT INTO products ({', '.join(_PRODUCT_COLS)}) VALUES ("
    + ", ".join(f"%({c})s" for c in _PRODUCT_COLS)
    + ") ON CONFLICT (source, source_id) DO UPDATE SET "
    + ", ".join(f"{c} = EXCLUDED.{c}" for c in _PRODUCT_COLS[2:])
    + ", updated_at = now() RETURNING id"
)


def upsert_product(conn: psycopg.Connection, product: Product) -> int:
    params = product.model_dump()
    params["raw"] = json.dumps(product.raw) if product.raw is not None else None
    return conn.execute(_PRODUCT_UPSERT, params).fetchone()["id"]


# ───────────── ingestion runs ─────────────
def start_run(conn: psycopg.Connection, source: str, params: dict) -> int:
    run_id = conn.execute(
        "INSERT INTO ingestion_runs (source, params) VALUES (%s, %s) RETURNING id",
        (source, json.dumps(params)),
    ).fetchone()["id"]
    conn.commit()
    return run_id


def finish_run(conn: psycopg.Connection, run_id: int, status: str, n_fetched: int,
               n_upserted: int, n_skipped: int, error: str | None = None) -> None:
    conn.execute(
        "UPDATE ingestion_runs SET finished_at=now(), status=%s, n_fetched=%s, n_upserted=%s, "
        "n_skipped=%s, error=%s WHERE id=%s",
        (status, n_fetched, n_upserted, n_skipped, error, run_id),
    )
    conn.commit()


def stats(conn: psycopg.Connection) -> dict:
    q = lambda sql: conn.execute(sql).fetchall()  # noqa: E731
    return {
        "papers": q("SELECT count(*) AS n, count(*) FILTER (WHERE fulltext_available) AS with_fulltext FROM papers")[0],
        "papers_by_species": q("SELECT unnest(species) AS species, count(*) AS n FROM papers GROUP BY 1 ORDER BY 2 DESC"),
        "papers_by_topic": q("SELECT unnest(topics) AS topic, count(*) AS n FROM papers GROUP BY 1 ORDER BY 2 DESC"),
        "papers_by_study_year_status": q("SELECT study_year_status AS status, count(*) AS n FROM papers GROUP BY 1 ORDER BY 2 DESC"),
        "products_by_source_species": q("SELECT source, species, count(*) AS n FROM products GROUP BY 1,2 ORDER BY 3 DESC"),
        "products_missing_nutrients": q("SELECT count(*) AS n FROM products WHERE protein_pct IS NULL")[0],
        "recent_runs": q("SELECT id, source, status, n_fetched, n_upserted, n_skipped, started_at, finished_at FROM ingestion_runs ORDER BY id DESC LIMIT 5"),
    }
