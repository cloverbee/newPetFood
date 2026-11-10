"""Index + search CLI.

    petfood-index build [--recreate] [--only paper|product]
    petfood-index search "food for a cat with kidney disease" --species cat [--type paper]
                         [--exclude chicken fish] [--mode hybrid_rerank] [-k 5]
                         [--pub-from 2015] [--pub-to 2020] [--study-from 2010] [--study-to 2018]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap

from petfood.config import get_settings


def make_store(fake: bool = False):
    from qdrant_client import QdrantClient

    from petfood.retrieval.embeddings import FastEmbedBM25, FastEmbedDense, HashDense, HashSparse
    from petfood.retrieval.store import ChunkStore

    s = get_settings()
    if fake:
        return ChunkStore(QdrantClient(":memory:"), s.qdrant_collection, HashDense(), HashSparse())
    return ChunkStore(QdrantClient(url=s.qdrant_url), s.qdrant_collection,
                      FastEmbedDense(s.dense_model, cache_dir=s.fastembed_cache_dir),
                      FastEmbedBM25(s.sparse_model, cache_dir=s.fastembed_cache_dir))


def make_retriever(store=None, rerank: bool = True):
    from petfood.retrieval.embeddings import FastEmbedReranker
    from petfood.retrieval.pipeline import HybridRetriever

    s = get_settings()
    reranker = FastEmbedReranker(s.rerank_model, cache_dir=s.fastembed_cache_dir) if rerank else None
    return HybridRetriever(store or make_store(), reranker, rerank_max_words=s.rerank_max_words)


def cmd_build(args) -> None:
    from petfood import db
    from petfood.retrieval.indexer import build_index

    store = make_store()
    with db.connect() as conn:
        stats = build_index(conn, store, doc_types=[args.only] if args.only else ("paper", "product"),
                            recreate=args.recreate)
    print(json.dumps(stats, indent=2))


def _years(p: dict) -> str:
    if p.get("study_year_start"):
        study = f"{p['study_year_start']}–{p['study_year_end']}"
    else:
        study = f"≤{p['study_year_latest']}" if p.get("study_year_latest") else "?"
    return f"pub {p.get('pub_year') or '?'} · study {study}"


def cmd_search(args) -> None:
    from petfood.retrieval.store import SearchFilters

    retriever = make_retriever(rerank=args.mode == "hybrid_rerank")
    f = SearchFilters(doc_type=args.type, species=args.species, exclude_allergens=args.exclude or (),
                      life_stage=args.life_stage, exclude_low_value=args.no_methods,
                      pub_year_from=args.pub_from, pub_year_to=args.pub_to,
                      study_year_from=args.study_from, study_year_to=args.study_to)
    res = retriever.retrieve(args.query, top_k=args.k, filters=f, mode=args.mode)
    for c in res.results:
        p = c.payload
        title = p.get("title") or p.get("name")
        where = f"{p.get('source_id', '')} · {p.get('heading', '')} · {_years(p)}" if p.get("doc_type") == "paper" \
            else f"{p.get('brand') or ''} · allergens={p.get('allergen_tags')}"
        score = f"rerank={c.rerank_score:.3f}" if c.rerank_score is not None else f"rrf={c.rrf_score:.4f}"
        print(f"\n#{c.final_rank} [{p.get('doc_type')}] {title}\n   {where}\n   {score}  "
              f"dense_rank={c.dense_rank} bm25_rank={c.sparse_rank}")
        body = p.get("body") or p.get("text", "")
        print(textwrap.indent(textwrap.shorten(body, 400), "   > "))
    print("\ntimings (ms):", {k: round(v, 1) for k, v in res.timings_ms.items()})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="petfood-index")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="chunk Postgres docs and (re)index them in Qdrant")
    b.add_argument("--recreate", action="store_true", help="drop and recreate the collection")
    b.add_argument("--only", choices=["paper", "product"])
    b.set_defaults(func=cmd_build)

    s = sub.add_parser("search", help="query the hybrid index")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=5)
    s.add_argument("--type", choices=["paper", "product"])
    s.add_argument("--species", choices=["cat", "dog"])
    s.add_argument("--exclude", nargs="+", help="allergen tags to exclude, e.g. chicken fish")
    s.add_argument("--life-stage", choices=["kitten", "puppy", "adult", "senior"])
    s.add_argument("--pub-from", type=int, metavar="YEAR", help="published in or after YEAR")
    s.add_argument("--pub-to", type=int, metavar="YEAR", help="published in or before YEAR")
    s.add_argument("--study-from", type=int, metavar="YEAR", help="study may have run in or after YEAR")
    s.add_argument("--study-to", type=int, metavar="YEAR", help="study may have run in or before YEAR")
    s.add_argument("--no-methods", action="store_true", help="drop Methods-section chunks")
    s.add_argument("--mode", default="hybrid_rerank", choices=["dense", "sparse", "hybrid", "hybrid_rerank"])
    s.set_defaults(func=cmd_search)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
