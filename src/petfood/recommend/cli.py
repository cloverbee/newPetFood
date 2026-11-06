"""Terminal wrapper around recommend.answer.answer().

    petfood-answer "My senior cat is overweight. What wet food?" \\
        --species cat --life-stage senior --food-form wet --exclude chicken --condition overweight
    petfood-answer "..." --json          # full Answer as JSON (sources, warnings, timings)
    petfood-answer "..." --show-prompt   # print the prompt sent to the LLM
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap

from petfood.recommend.answer import (Answer, PaperFilters, PetProfile, answer, build_prompt,
                                      gather_context)


def _print(a: Answer) -> None:
    print(f"\nQ: {a.question}\n")
    print(textwrap.fill(a.summary or "(no summary)", 100))
    if a.recommendations:
        print("\nRecommended:")
        for i, r in enumerate(a.recommendations, 1):
            brand = f" — {r.brand}" if r.brand else ""
            print(f"  {i}. {r.name}{brand}")
            print(textwrap.indent(textwrap.fill(r.reason, 94), "     "))
    else:
        print("\nRecommended: none")
    cited = [s for s in a.sources if s.cited]
    if cited:
        print("\nSources:")
        for s in cited:
            print(f"  [{s.label}] {s.source_id} ({s.pub_year}) {s.title} — {s.heading}")
    if a.vet_note:
        print(f"\n⚠️  {a.vet_note}")
    if a.warnings:
        print("\nChecks removed:")
        for w in a.warnings:
            print(f"  - {w}")
    print(f"\n(context: {a.n_products} products, {a.n_evidence} excerpts · "
          f"timings ms: { {k: round(v) for k, v in a.timings_ms.items()} })")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="petfood-answer", description="Answer a pet food question with citations")
    p.add_argument("question")
    p.add_argument("--species", default="cat", choices=["cat", "dog"])
    p.add_argument("--life-stage", choices=["kitten", "puppy", "adult", "senior"])
    p.add_argument("--food-form", choices=["dry", "wet", "raw", "treat"])
    p.add_argument("--exclude", nargs="+", default=[], help="allergen tags to avoid, e.g. chicken fish")
    p.add_argument("--condition", nargs="+", default=[], help="health conditions, e.g. overweight ckd")
    p.add_argument("--pub-from", type=int, metavar="YEAR")
    p.add_argument("--pub-to", type=int, metavar="YEAR")
    p.add_argument("--study-from", type=int, metavar="YEAR")
    p.add_argument("--study-to", type=int, metavar="YEAR")
    p.add_argument("--k-products", type=int, default=8)
    p.add_argument("--k-evidence", type=int, default=5)
    p.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder (faster, less precise)")
    p.add_argument("--show-prompt", action="store_true", help="print the prompt and stop (no LLM call)")
    p.add_argument("--json", action="store_true", help="print the full answer as JSON")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from petfood.retrieval.cli import make_retriever

    pet = PetProfile(species=args.species, life_stage=args.life_stage, food_form=args.food_form,
                     exclude_allergens=tuple(args.exclude), conditions=tuple(args.condition))
    papers = PaperFilters(args.pub_from, args.pub_to, args.study_from, args.study_to)
    retriever = make_retriever(rerank=not args.no_rerank)

    if args.show_prompt:
        products, evidence = gather_context(retriever, args.question, pet, papers,
                                            args.k_products, args.k_evidence)
        system, user = build_prompt(args.question, pet, products, evidence)
        print(f"── system ──\n{system}\n\n── user ──\n{user}")
        return

    from petfood.llm.deepseek import DeepSeekClient

    a = answer(args.question, pet, retriever, DeepSeekClient().chat_json, papers,
               args.k_products, args.k_evidence)
    if args.json:
        print(json.dumps(a.to_dict(), indent=2, ensure_ascii=False))
    else:
        _print(a)


if __name__ == "__main__":
    sys.exit(main())
