"""Compare reranker settings on the eval questions, search only (no LLM, deterministic).

    .venv/bin/python scripts/compare_rerank.py                 # default settings below
    .venv/bin/python scripts/compare_rerank.py --words 96 128  # chosen word limits

The baseline is the current setup: 30 candidates, full text. For each other setting it reports
the rerank time and how much of the baseline's selection survives: the 5 paper excerpts and up
to 8 products the LLM would get, and whether the #1 excerpt is the same (D21).
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from petfood.eval.run import load_questions                       # noqa: E402
from petfood.recommend.answer import gather_context, profile_from_question  # noqa: E402
from petfood.retrieval.cli import make_retriever                 # noqa: E402


def run(retriever, qs, k_rerank, max_words):
    retriever.k_rerank, retriever.rerank_max_words = k_rerank, max_words
    out = {}
    for q in qs:
        pet = profile_from_question(q.question, q.pet)
        t = {}
        prods, evid = gather_context(retriever, q.question, pet, timings=t)
        out[q.id] = {"papers": [c.id for c in evid], "products": [c.id for c in prods],
                     "paper_ms": t["papers.rerank"], "product_ms": t["products.rerank"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", type=int, nargs="+", default=[64, 96, 128, 160, 200])
    ap.add_argument("--k", type=int, nargs="+", default=[20], help="also try fewer candidates, full text")
    args = ap.parse_args()

    qs = load_questions()
    r = make_retriever()
    run(r, qs[:1], 30, None)   # warm-up
    t0 = time.perf_counter()
    base = run(r, qs, 30, None)
    print(f"baseline (30 candidates, full text) took {time.perf_counter() - t0:.0f} s for {len(qs)} questions\n")

    settings = [("full text, k=30 (baseline)", 30, None)]
    settings += [(f"first {w} words, k=30", 30, w) for w in args.words]
    settings += [(f"full text, k={k}", k, None) for k in args.k]
    print(f"{'setting':<28} {'paper rerank':>12} {'product rerank':>14}   {'papers kept':>11} {'#1 same':>8} {'products kept':>13}")
    for label, k, w in settings:
        res = base if w is None and k == 30 else run(r, qs, k, w)
        pk = sum(len(set(res[i]["papers"]) & set(base[i]["papers"])) for i in base)
        pt = sum(len(base[i]["papers"]) for i in base)
        top = sum(bool(res[i]["papers"]) and res[i]["papers"][:1] == base[i]["papers"][:1] for i in base)
        rk = sum(len(set(res[i]["products"]) & set(base[i]["products"])) for i in base)
        rt = sum(len(base[i]["products"]) for i in base)
        pm = statistics.mean(v["paper_ms"] for v in res.values()) / 1000
        prm = statistics.mean(v["product_ms"] for v in res.values()) / 1000
        print(f"{label:<28} {pm:>10.2f} s {prm:>12.2f} s   {pk:>4}/{pt:<3} {pk / pt:4.0%} {top:>4}/{len(qs)}  "
              f"{rk:>4}/{rt:<3} {rk / max(rt, 1):4.0%}", flush=True)


if __name__ == "__main__":
    main()
