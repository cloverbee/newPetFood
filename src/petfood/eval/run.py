"""Run the test questions in eval/questions.toml through answer() and score them.

    petfood-eval                        # all questions, DeepSeek
    petfood-eval --llm local            # same questions, local Qwen
    petfood-eval --only kitten vegan    # a few questions
    petfood-eval --k-rerank 15          # try a cheaper retrieval setting

Models load once for the whole run. Each run writes eval/results/<time>-<llm>.jsonl (one full
answer per line, plus its checks) and prints a table. Checks are automatic and simple; they say
whether an answer did the basic things right, not whether it's a *good* answer. Read the
answers for that.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from petfood.recommend.answer import Answer, PetProfile, answer
from petfood.recommend.latency import format_many
from petfood.retrieval.store import LIFE_STAGE_MATCHES, UNSTATED_LIFE_STAGE_OK

ROOT = Path(__file__).resolve().parents[3]
QUESTIONS = ROOT / "eval" / "questions.toml"
RESULTS = ROOT / "eval" / "results"

# column order in the table
CHECKS = ["filters", "recs", "recs_fit", "vet_note", "paper_found", "paper_cited", "no_removals"]


@dataclass
class Question:
    id: str
    question: str
    exclude: list[str] = field(default_factory=list)
    condition: list[str] = field(default_factory=list)
    expect: dict = field(default_factory=dict)

    @property
    def pet(self) -> PetProfile:
        return PetProfile(exclude_allergens=tuple(self.exclude), conditions=tuple(self.condition))


def load_questions(path: Path = QUESTIONS) -> list[Question]:
    data = tomllib.loads(path.read_text())
    qs = [Question(**q) for q in data.get("q", [])]
    ids = [q.id for q in qs]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate question ids in {path}")
    return qs


def _stage_ok(wanted: str, stated: str | None) -> bool:
    if not stated:
        return wanted in UNSTATED_LIFE_STAGE_OK
    return stated in LIFE_STAGE_MATCHES.get(wanted, [wanted, "all"])


def product_problems(p: dict, profile: dict) -> list[str]:
    """Why a recommended product doesn't fit the profile the answer used (empty = fits)."""
    out = []
    if profile.get("food_form") and p.get("food_form") != profile["food_form"]:
        out.append(f"form {p.get('food_form')} ≠ {profile['food_form']}")
    if profile.get("life_stage") and not _stage_ok(profile["life_stage"], p.get("life_stage")):
        out.append(f"life stage {p.get('life_stage') or 'unstated'} ≠ {profile['life_stage']}")
    excluded = set(profile.get("exclude_allergens") or ())
    if excluded:
        if clash := excluded & set(p.get("allergen_tags") or ()):
            out.append(f"contains {', '.join(sorted(clash))}")
        if not p.get("ingredients"):
            out.append("no ingredient list to check allergens")
    return out


def score(q: Question, a: dict, products: dict[int, dict]) -> tuple[dict[str, bool | None], list[str]]:
    """Checks for one answer (a = Answer.to_dict()). None = not applicable. Also returns notes
    explaining each failure."""
    e, notes = q.expect, []
    checks: dict[str, bool | None] = dict.fromkeys(CHECKS)

    wanted = {k: e[k] for k in ("life_stage", "food_form") if k in e}
    if wanted:
        got = {k: a["profile"].get(k) or "none" for k in wanted}
        checks["filters"] = got == wanted
        if not checks["filters"]:
            notes.append(f"filters {got} ≠ {wanted}")

    recs = a["recommendations"]
    if "min_recs" in e:
        checks["recs"] = len(recs) >= e["min_recs"]
        if not checks["recs"]:
            notes.append(f"{len(recs)} recommendations < {e['min_recs']}")

    if recs:
        bad = [f"{r['name']}: {'; '.join(probs)}" for r in recs
               if (probs := product_problems(products.get(r["product_id"], {}), a["profile"]))]
        checks["recs_fit"] = not bad
        notes += bad

    if "vet_note" in e:
        checks["vet_note"] = bool(a["vet_note"]) == e["vet_note"]
        if not checks["vet_note"]:
            notes.append("vet note missing" if e["vet_note"] else "vet note not expected")

    if e.get("papers"):
        frags = [f.lower() for f in e["papers"]]
        def hit(s): return any(f in (s["title"] or "").lower() for f in frags)
        checks["paper_found"] = any(hit(s) for s in a["sources"])
        checks["paper_cited"] = any(hit(s) and s["cited"] for s in a["sources"])
        if not checks["paper_found"]:
            notes.append(f"none of {e['papers']} retrieved")

    checks["no_removals"] = not a["warnings"]
    notes += [f"removed: {w}" for w in a["warnings"]]
    return checks, notes


def _load_products(ids: set[int]) -> dict[int, dict]:
    if not ids:
        return {}
    from petfood.db import connect
    with connect() as conn:
        rows = conn.execute("SELECT id, name, food_form, life_stage, allergen_tags, ingredients "
                            "FROM products WHERE id = ANY(%s)", (list(ids),)).fetchall()
    return {r["id"]: r for r in rows}


def _mark(v: bool | None) -> str:
    return "·" if v is None else ("✓" if v else "✗")


def print_table(rows: list[dict]) -> None:
    w = max(len(r["id"]) for r in rows)
    heads = ["filters", "recs", "fit", "vet", "found", "cited", "clean"]
    print(f"\n{'question':<{w}}  " + "  ".join(f"{h:^5}" for h in heads) + "   #recs   secs")
    for r in rows:
        if r.get("error"):
            print(f"{r['id']:<{w}}  ERROR: {r['error']}")
            continue
        marks = "  ".join(f"{_mark(r['checks'][c]):^5}" for c in CHECKS)
        print(f"{r['id']:<{w}}  {marks}   {len(r['answer']['recommendations']):^5}  "
              f"{r['answer']['timings_ms'].get('total', 0) / 1000:5.1f}")
    print("\nPassed (of questions where the check applies):")
    for c, h in zip(CHECKS, heads):
        vals = [r["checks"][c] for r in rows if not r.get("error") and r["checks"][c] is not None]
        if vals:
            print(f"  {h:<6} {sum(vals):>2}/{len(vals)}")
    times = [r["answer"]["timings_ms"].get("total", 0) / 1000 for r in rows if not r.get("error")]
    if times:
        print(f"  time   {sum(times) / len(times):.1f} s average, {max(times):.1f} s slowest")
    fails = [(r["id"], n) for r in rows for n in r.get("notes", [])]
    if fails:
        print("\nWhy:")
        for qid, n in fails:
            print(f"  {qid}: {n}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="petfood-eval", description=__doc__.split("\n\n")[0])
    p.add_argument("--llm", choices=["deepseek", "local"], default="deepseek")
    p.add_argument("--only", nargs="+", metavar="ID", help="run only these question ids")
    p.add_argument("--questions", type=Path, default=QUESTIONS)
    p.add_argument("--k-rerank", type=int, help="candidates the reranker scores (default 30)")
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    qs = load_questions(args.questions)
    if args.only:
        missing = set(args.only) - {q.id for q in qs}
        if missing:
            sys.exit(f"unknown question ids: {', '.join(sorted(missing))}")
        qs = [q for q in qs if q.id in args.only]

    from petfood.recommend.cli import make_llm
    from petfood.retrieval.cli import make_retriever
    t0 = time.perf_counter()
    retriever = make_retriever(rerank=not args.no_rerank)
    if args.k_rerank:
        retriever.k_rerank = args.k_rerank
    llm, client = make_llm(args.llm)
    model = client.model
    print(f"models loaded in {time.perf_counter() - t0:.1f} s (once for the whole run)")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{time.strftime('%Y%m%d-%H%M%S')}-{args.llm}.jsonl"
    settings = {"llm": args.llm, "model": model, "rerank": not args.no_rerank,
                "k_rerank": retriever.k_rerank}
    rows = []
    with out.open("w") as f:
        for i, q in enumerate(qs, 1):
            print(f"[{i}/{len(qs)}] {q.id} …", end=" ", flush=True)
            row = {"id": q.id, "question": q.question, "settings": settings}
            try:
                a: Answer = answer(q.question, q.pet, retriever, llm)
                a.model, a.llm_stats = model, dict(client.last_stats)
                ad = a.to_dict()
                products = _load_products({r["product_id"] for r in ad["recommendations"]})
                row["checks"], row["notes"] = score(q, ad, products)
                row["answer"] = ad
                print(f"{ad['timings_ms'].get('total', 0) / 1000:.1f} s")
            except Exception as ex:   # one bad question shouldn't stop the run
                row["error"] = f"{type(ex).__name__}: {ex}"
                print("error")
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            f.flush()

    print_table(rows)
    ok = [r["answer"] for r in rows if not r.get("error")]
    print("\n" + format_many([a["timings_ms"] for a in ok], [a["llm_stats"] for a in ok]))
    print(f"\nFull answers: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
