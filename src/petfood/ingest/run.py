"""Ingestion CLI.

    petfood-ingest init-db
    petfood-ingest papers   --limit 200 [--query "..."] [--skip-existing] [--max-new N]
    petfood-ingest products --limit 500 [--species cat dog] [--from-file dump.jsonl.gz]
    petfood-ingest curated  --path data/products_curated.csv
    petfood-ingest prune [--yes]
    petfood-ingest study-years [--force] [--limit N]
    petfood-ingest stats
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from petfood import db
from petfood.config import get_settings

log = logging.getLogger("petfood.ingest")


def cmd_init_db(_args) -> None:
    with db.connect() as conn:
        db.apply_schema(conn)
    print("schema applied")


def cmd_papers(args) -> None:
    from petfood.ingest.europepmc import (DEFAULT_QUERY, EuropePMCClient, attach_full_text,
                                          license_ok, record_to_paper)
    from petfood.storage import ObjectStore

    s = get_settings()
    client = EuropePMCClient(contact_email=s.contact_email)
    store = ObjectStore(s)
    store.ensure_bucket()
    query = args.query or DEFAULT_QUERY

    with db.connect() as conn:
        run_id = db.start_run(conn, "europepmc", {"limit": args.limit, "query": query})
        fetched = upserted = skipped = 0
        try:
            for rec in client.search(query=query, limit=args.limit):
                fetched += 1
                paper = record_to_paper(rec)
                if paper is None or not license_ok(paper.license):
                    skipped += 1
                    continue
                if args.skip_existing and db.paper_exists(conn, paper.source, paper.source_id):
                    skipped += 1
                    continue
                key = f"papers/europepmc/{paper.source_id}.xml"
                xml = client.full_text_xml(paper.source_id)
                if xml:
                    store.put_bytes(key, xml)
                    attach_full_text(paper, xml, key)
                db.upsert_paper(conn, paper)
                conn.commit()
                upserted += 1
                if upserted % 20 == 0:
                    log.info("papers: %d upserted / %d fetched", upserted, fetched)
                if args.max_new and upserted >= args.max_new:
                    break
            db.finish_run(conn, run_id, "ok", fetched, upserted, skipped)
        except Exception as e:
            conn.rollback()
            db.finish_run(conn, run_id, "failed", fetched, upserted, skipped, repr(e))
            raise
    print(f"papers: fetched={fetched} upserted={upserted} skipped={skipped}")


def cmd_products(args) -> None:
    from petfood.ingest.openpetfoodfacts import OPFFClient, iter_dump, record_to_product

    s = get_settings()
    with db.connect() as conn:
        run_id = db.start_run(conn, "openpetfoodfacts", vars(args) | {"func": None})
        fetched = upserted = skipped = 0
        try:
            if args.from_file:
                sources = [(None, iter_dump(args.from_file))]
            else:
                client = OPFFClient(contact_email=s.contact_email)
                sources = [(sp, client.search(sp, limit=args.limit, country=args.country)) for sp in args.species]
            for hint, records in sources:
                n_this = 0
                for rec in records:
                    fetched += 1
                    product = record_to_product(rec, species_hint=hint)
                    if product is None or (args.require_ingredients and not product.ingredients):
                        skipped += 1
                        continue
                    db.upsert_product(conn, product)
                    upserted += 1
                    n_this += 1
                    if upserted % 100 == 0:
                        conn.commit()
                        log.info("products: %d upserted", upserted)
                    if args.from_file and n_this >= args.limit:
                        break
            conn.commit()
            db.finish_run(conn, run_id, "ok", fetched, upserted, skipped)
        except Exception as e:
            conn.rollback()
            db.finish_run(conn, run_id, "failed", fetched, upserted, skipped, repr(e))
            raise
    print(f"products: fetched={fetched} upserted={upserted} skipped={skipped}")


def cmd_curated(args) -> None:
    from petfood.ingest.curated import load_csv

    products = load_csv(args.path)
    with db.connect() as conn:
        run_id = db.start_run(conn, "curated", {"path": args.path})
        for p in products:
            db.upsert_product(conn, p)
        conn.commit()
        db.finish_run(conn, run_id, "ok", len(products), len(products), 0)
    print(f"curated: upserted={len(products)}")


def cmd_reparse(_args) -> None:
    """Re-parse every stored paper from its raw XML in S3 (no re-download) and replace its
    sections. Use after improving the JATS parser."""
    from petfood.ingest.jats import parse_front_meta, parse_jats
    from petfood.storage import ObjectStore

    store = ObjectStore()
    n_papers = n_secs = n_tables = 0
    with db.connect() as conn:
        rows = conn.execute("SELECT id, s3_key FROM papers WHERE s3_key IS NOT NULL ORDER BY id").fetchall()
        for r in rows:
            xml = store.get_bytes(r["s3_key"])
            _abstract, sections = parse_jats(xml)
            if not sections:
                continue
            conn.execute("DELETE FROM paper_sections WHERE paper_id=%s", (r["id"],))
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO paper_sections (paper_id, ord, section_type, heading, text) VALUES (%s,%s,%s,%s,%s)",
                    [(r["id"], s.ord, s.section_type, s.heading, s.text) for s in sections])
            conn.execute("UPDATE papers SET fulltext_available = %s, received_year = %s, updated_at = now() "
                         "WHERE id = %s", (any(s.section_type != "abstract" for s in sections),
                                           parse_front_meta(xml)["received_year"], r["id"]))
            n_papers += 1
            n_secs += len(sections)
            n_tables += sum(s.section_type == "table" for s in sections)
        conn.commit()
    print(f"reparsed {n_papers} papers: {n_secs} sections, {n_tables} tables")


def cmd_retag(_args) -> None:
    """Recompute paper species/topics with the current tagging rules (no re-download)."""
    from petfood.ingest.tagging import detect_species, detect_topics

    with db.connect() as conn:
        papers = conn.execute("SELECT id, title, abstract FROM papers").fetchall()
        for p in papers:
            secs = conn.execute(
                "SELECT text FROM paper_sections WHERE paper_id=%s ORDER BY ord LIMIT 6", (p["id"],)
            ).fetchall()
            head = f"{p['title']} {p['abstract'] or ''}"
            body = " ".join(s["text"] for s in secs)
            conn.execute("UPDATE papers SET species=%s, topics=%s, updated_at=now() WHERE id=%s",
                         (detect_species(head) or detect_species(body),
                          sorted(set(detect_topics(head)) | set(detect_topics(body))), p["id"]))
        conn.commit()
    print(f"retagged {len(papers)} papers")


def cmd_prune(args) -> None:
    """Remove papers that fail the current scope rule (europepmc.in_scope) from Postgres and
    their raw XML from S3. Dry run unless --yes."""
    from petfood.ingest.europepmc import in_scope
    from petfood.storage import ObjectStore

    with db.connect() as conn:
        rows = conn.execute("SELECT id, source_id, title, abstract, s3_key FROM papers ORDER BY id").fetchall()
        out = [r for r in rows if not in_scope(r["title"], r["abstract"])]
        for r in out:
            print(f"  {r['source_id']:<12} {r['title'][:100]}")
        if not args.yes:
            print(f"prune: {len(out)} of {len(rows)} papers out of scope (dry run; --yes to delete)")
            return
        store = ObjectStore()
        for r in out:   # S3 first: if it fails midway, Postgres still lists what's left to clean
            if r["s3_key"]:
                store.delete(r["s3_key"])
        ids = [r["id"] for r in out]
        conn.execute("DELETE FROM chunks WHERE doc_type='paper' AND doc_id = ANY(%s)", (ids,))
        conn.execute("DELETE FROM papers WHERE id = ANY(%s)", (ids,))   # sections cascade
        conn.commit()
    print(f"prune: deleted {len(out)} of {len(rows)} papers (Postgres + S3). "
          "Re-run `make index` to update Qdrant.")


def cmd_study_years(args) -> None:
    """Extract each paper's study period (DeepSeek + checks); see ingest/study_years.py.
    Skips papers already done unless --force. Re-run `make index` afterwards."""
    from collections import Counter

    from petfood.ingest.jats import parse_front_meta
    from petfood.ingest.study_years import extract_study_years
    from petfood.llm.deepseek import DeepSeekClient
    from petfood.storage import ObjectStore

    store = ObjectStore()
    llm = DeepSeekClient().chat_json
    counts: Counter[str] = Counter()
    with db.connect() as conn:
        where = "" if args.force else "WHERE study_year_status IS NULL"
        papers = conn.execute(f"SELECT * FROM papers {where} ORDER BY id LIMIT %s",
                              (args.limit,)).fetchall()
        for i, p in enumerate(papers, 1):
            meta = parse_front_meta(store.get_bytes(p["s3_key"])) if p["s3_key"] else {}
            p["received_year"] = p["received_year"] or meta.get("received_year")
            secs = conn.execute("SELECT section_type, text FROM paper_sections WHERE paper_id=%s ORDER BY ord",
                                (p["id"],)).fetchall()
            try:
                sy = extract_study_years(p, secs, meta.get("article_type"), llm)
            except Exception as e:   # one bad response shouldn't stop the run; rerun picks it up
                log.warning("study-years: %s failed: %r", p["source_id"], e)
                counts["error"] += 1
                continue
            conn.execute(
                "UPDATE papers SET received_year=%s, study_year_start=%s, study_year_end=%s, "
                "study_year_status=%s, study_year_evidence=%s, study_year_note=%s, updated_at=now() "
                "WHERE id=%s",
                (p["received_year"], sy.start, sy.end, sy.status, sy.evidence, sy.note, p["id"]))
            conn.commit()
            counts[sy.status] += 1
            if sy.status == "found":
                log.info("%s: %s–%s", p["source_id"], sy.start, sy.end)
            if i % 20 == 0:
                log.info("study-years: %d / %d", i, len(papers))
    print(f"study-years: {len(papers)} papers {dict(counts)}")


def cmd_stats(_args) -> None:
    with db.connect() as conn:
        print(json.dumps(db.stats(conn), indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="petfood-ingest")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init-db").set_defaults(func=cmd_init_db)

    pp = sub.add_parser("papers", help="open-access vet papers from Europe PMC")
    pp.add_argument("--limit", type=int, default=200)
    pp.add_argument("--query", default=None, help="override the Europe PMC query")
    pp.add_argument("--skip-existing", action="store_true")
    pp.add_argument("--max-new", type=int, default=None,
                    help="stop after N papers are stored (--limit caps search results read)")
    pp.set_defaults(func=cmd_papers)

    pr = sub.add_parser("products", help="cat/dog food from Open Pet Food Facts")
    pr.add_argument("--limit", type=int, default=500, help="per species (API) or total (dump)")
    pr.add_argument("--species", nargs="+", default=["cat", "dog"], choices=["cat", "dog"])
    pr.add_argument("--country", default=None,
                    help='only products sold in this country, e.g. "united-states"')
    pr.add_argument("--from-file", default=None, help="OPFF JSONL(.gz) dump instead of the API")
    pr.add_argument("--require-ingredients", action="store_true",
                    help="skip products without an ingredient list")
    pr.set_defaults(func=cmd_products)

    pc = sub.add_parser("curated", help="hand-curated products CSV")
    pc.add_argument("--path", default="data/products_curated.csv")
    pc.set_defaults(func=cmd_curated)

    sub.add_parser("reparse", help="re-parse stored XML from S3 into sections").set_defaults(func=cmd_reparse)
    sub.add_parser("retag", help="recompute paper species/topic tags").set_defaults(func=cmd_retag)
    pn = sub.add_parser("prune", help="delete papers outside the current scope rule (dry run by default)")
    pn.add_argument("--yes", action="store_true", help="actually delete")
    pn.set_defaults(func=cmd_prune)

    sy = sub.add_parser("study-years", help="extract when each study's data was collected (DeepSeek)")
    sy.add_argument("--force", action="store_true", help="redo papers that already have a result")
    sy.add_argument("--limit", type=int, default=None, help="only the first N papers (for a trial run)")
    sy.set_defaults(func=cmd_study_years)
    sub.add_parser("stats").set_defaults(func=cmd_stats)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
