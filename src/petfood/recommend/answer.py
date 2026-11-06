"""Answer a pet owner's question: retrieve products + paper evidence, generate, then check (D14).

This is the "G" in RAG. Retrieval already exists (retrieval/pipeline.py); this module adds:

1. gather_context()  two filtered searches: candidate products, then supporting paper chunks
2. build_prompt()    labels them [P1..] and [E1..] so the model cites labels, not free text
3. LLM call          any (system, user) -> dict callable; DeepSeek in the CLI, a fake in tests
4. check_answer()    code, not the model, decides what survives: unknown labels, products that
                     break the pet's allergen/species constraints and invented citations are
                     dropped with a warning. The vet note is a rule, never left to the model.

The API (Phase 4) will call answer() directly; `petfood-answer` is the terminal wrapper.
"""
from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field

from petfood.ingest.tagging import detect_topics
from petfood.retrieval.pipeline import Candidate, HybridRetriever
from petfood.retrieval.store import SearchFilters

LLM = Callable[[str, str], dict]   # (system, user) -> parsed JSON

MAX_RECOMMENDATIONS = 3
MAX_PRODUCT_CHARS = 700     # ingredient lists are long; the start carries the useful part
MAX_EVIDENCE_CHARS = 1500

# Topics (from ingest/tagging.py) that make a question medical: the answer always gets a vet note.
MEDICAL_TOPICS = {"obesity", "renal", "urinary", "diabetes", "food_allergy", "gastrointestinal",
                  "cardiac", "hepatic"}
VET_NOTE = ("This is general information, not veterinary advice. Because a health condition is "
            "involved, talk to your veterinarian before changing your pet's diet.")

_LABEL = re.compile(r"\[([PE]\d+)\]")


@dataclass
class PetProfile:
    species: str = "cat"
    life_stage: str | None = None              # kitten | adult | senior (products tagged "all" always match)
    food_form: str | None = None               # dry | wet | raw | treat
    exclude_allergens: Sequence[str] = ()      # hard filter, e.g. ("chicken", "fish")
    conditions: Sequence[str] = ()             # free text, e.g. ("overweight",); always triggers the vet note


@dataclass
class PaperFilters:
    pub_year_from: int | None = None
    pub_year_to: int | None = None
    study_year_from: int | None = None
    study_year_to: int | None = None


@dataclass
class Recommendation:
    label: str                 # P-label from the prompt
    product_id: int
    name: str
    brand: str | None
    reason: str
    evidence: list[str]        # E-labels that survived the check


@dataclass
class Source:
    label: str                 # E-label
    source_id: str             # PMC id
    title: str
    heading: str | None
    pub_year: int | None
    cited: bool


@dataclass
class Answer:
    question: str
    summary: str
    recommendations: list[Recommendation]
    sources: list[Source]
    vet_note: str | None
    warnings: list[str] = field(default_factory=list)   # what the checks removed, and why
    n_products: int = 0        # candidates the model saw
    n_evidence: int = 0
    timings_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ───────────── 1. retrieval ─────────────
def gather_context(retriever: HybridRetriever, question: str, pet: PetProfile,
                   papers: PaperFilters | None = None, k_products: int = 8,
                   k_evidence: int = 5) -> tuple[list[Candidate], list[Candidate]]:
    """Two searches with the same question. Products are filtered by the pet's hard constraints,
    papers by species and optional year filters. One product chunk = one product."""
    pf = papers or PaperFilters()
    products = retriever.retrieve(question, top_k=k_products, max_per_doc=1, filters=SearchFilters(
        doc_type="product", species=pet.species, life_stage=pet.life_stage, food_form=pet.food_form,
        exclude_allergens=tuple(pet.exclude_allergens))).results
    evidence = retriever.retrieve(question, top_k=k_evidence, filters=SearchFilters(
        doc_type="paper", species=pet.species,
        pub_year_from=pf.pub_year_from, pub_year_to=pf.pub_year_to,
        study_year_from=pf.study_year_from, study_year_to=pf.study_year_to)).results
    return products, evidence


# ───────────── 2. prompt ─────────────
SYSTEM_PROMPT = """You help pet owners choose food. Answer ONLY from the products and research \
excerpts provided. You are not a veterinarian and do not diagnose.

Rules:
- Recommend at most 3 products, chosen only from the PRODUCTS list, by their label (P1, P2, ...).
  If none fits the question, recommend none and say why.
- Support claims with research excerpts by label in square brackets, e.g. [E2]. Cite only labels
  from the RESEARCH list, and only when the excerpt actually supports the claim.
- If the research does not address the question, say so plainly. Do not invent studies, numbers
  or product facts that are not in the text.
- Keep the summary to 3-5 sentences in plain language. In the summary and reasons, refer to
  products by name, never by their P label (the owner never sees the labels).

Reply with JSON only:
{"summary": "<answer with [E#] citations>",
 "recommendations": [{"product": "P#", "reason": "<1-2 sentences, may cite [E#]>", "evidence": ["E#"]}]}"""


def _profile_text(pet: PetProfile) -> str:
    parts = [f"species: {pet.species}"]
    if pet.life_stage:
        parts.append(f"life stage: {pet.life_stage}")
    if pet.food_form:
        parts.append(f"wants: {pet.food_form} food")
    if pet.exclude_allergens:
        parts.append("must avoid: " + ", ".join(pet.exclude_allergens))
    if pet.conditions:
        parts.append("conditions: " + ", ".join(pet.conditions))
    return "; ".join(parts)


def _years(p: dict) -> str:
    study = (f"{p['study_year_start']}–{p['study_year_end']}" if p.get("study_year_start")
             else "not stated")
    return f"published {p.get('pub_year') or '?'}, study period {study}"


def build_prompt(question: str, pet: PetProfile, products: list[Candidate],
                 evidence: list[Candidate]) -> tuple[str, str]:
    lines = [f"QUESTION: {question}", f"PET: {_profile_text(pet)}", "", "PRODUCTS:"]
    if not products:
        lines.append("(none matched the pet's constraints)")
    for i, c in enumerate(products, 1):
        lines.append(f"[P{i}] {c.text[:MAX_PRODUCT_CHARS]}")
    lines += ["", "RESEARCH:"]
    if not evidence:
        lines.append("(no relevant research found)")
    for i, c in enumerate(evidence, 1):
        p = c.payload
        body = (p.get("body") or c.text)[:MAX_EVIDENCE_CHARS]
        lines.append(f"[E{i}] \"{p.get('title')}\" — {p.get('heading')} ({_years(p)})\n{body}")
    return SYSTEM_PROMPT, "\n".join(lines)


# ───────────── 3. checks ─────────────
def needs_vet_note(question: str, pet: PetProfile) -> bool:
    return bool(pet.conditions) or bool(MEDICAL_TOPICS & set(detect_topics(question)))


def _clean_citations(text: str, allowed: set[str], warnings: list[str], where: str) -> str:
    """Remove [X#] labels that weren't in the prompt (or aren't evidence), keeping the sentence."""
    removed = False

    def repl(m: re.Match) -> str:
        nonlocal removed
        if m.group(1) in allowed:
            return m.group(0)
        removed = True
        warnings.append(f"{where}: removed unknown citation [{m.group(1)}]")
        return ""
    out = _LABEL.sub(repl, text)
    if removed:   # "fed wet food [E9]." -> "fed wet food."
        out = re.sub(r"\s+([.,;:])", r"\1", out)
    return out.strip()


_P_LABEL = re.compile(r"\[?\bP(\d+)\b\]?")


def _name_products(text: str, p_by_label: dict[str, Candidate]) -> str:
    """Replace leftover product labels ("P4", "[P4]") with the product's name."""
    def repl(m: re.Match) -> str:
        c = p_by_label.get(f"P{m.group(1)}")
        return (c.payload.get("name") or m.group(0)) if c else m.group(0)
    return _P_LABEL.sub(repl, text)


def check_answer(raw: dict, pet: PetProfile, products: list[Candidate],
                 evidence: list[Candidate]) -> tuple[str, list[Recommendation], set[str], list[str]]:
    """Keep only what the retrieved context supports. Returns (summary, recommendations,
    cited E-labels, warnings)."""
    warnings: list[str] = []
    p_by_label = {f"P{i}": c for i, c in enumerate(products, 1)}
    e_labels = {f"E{i}" for i in range(1, len(evidence) + 1)}
    excluded = {a.lower() for a in pet.exclude_allergens}

    summary = _clean_citations(_name_products(str(raw.get("summary") or ""), p_by_label),
                               e_labels, warnings, "summary")
    recs: list[Recommendation] = []
    seen: set[str] = set()
    raw_recs = raw.get("recommendations")
    if not isinstance(raw_recs, list):
        if raw_recs is not None:
            warnings.append(f"recommendations was not a list: {raw_recs!r}"[:200])
        raw_recs = []
    for r in raw_recs:
        if not isinstance(r, dict):
            warnings.append(f"skipped malformed recommendation: {r!r}"[:200])
            continue
        label = str(r.get("product") or "").strip("[] ")
        c = p_by_label.get(label)
        if c is None:
            warnings.append(f"dropped recommendation for unknown product {label!r}")
            continue
        if label in seen:
            continue
        p = c.payload
        # defence in depth: the search filter already removed these, re-check before showing
        clash = excluded & {t.lower() for t in p.get("allergen_tags") or []}
        if clash:
            warnings.append(f"dropped {label} ({p.get('name')}): contains {', '.join(sorted(clash))}")
            continue
        if excluded and not p.get("has_ingredients"):
            warnings.append(f"dropped {label} ({p.get('name')}): no ingredient list to check allergens")
            continue
        if pet.species not in (p.get("species") or []):
            warnings.append(f"dropped {label} ({p.get('name')}): not labelled for {pet.species}s")
            continue
        if len(recs) >= MAX_RECOMMENDATIONS:
            warnings.append(f"dropped {label}: more than {MAX_RECOMMENDATIONS} recommendations")
            continue
        reason = _clean_citations(_name_products(str(r.get("reason") or ""), p_by_label),
                                  e_labels, warnings, label)
        ev = [e for e in dict.fromkeys(str(x).strip("[] ") for x in r.get("evidence") or [])
              if e in e_labels]
        dropped_ev = len(r.get("evidence") or []) - len(ev)
        if dropped_ev > 0:
            warnings.append(f"{label}: removed {dropped_ev} unknown evidence label(s)")
        ev = list(dict.fromkeys(ev + _LABEL.findall(reason)))   # inline cites count too
        recs.append(Recommendation(label, p["doc_id"], p.get("name") or "", p.get("brand"), reason, ev))
        seen.add(label)

    cited = set(_LABEL.findall(summary)) | {e for r in recs for e in r.evidence}
    return summary, recs, cited & e_labels, warnings


# ───────────── 4. the whole flow ─────────────
def answer(question: str, pet: PetProfile, retriever: HybridRetriever, llm: LLM,
           papers: PaperFilters | None = None, k_products: int = 8, k_evidence: int = 5) -> Answer:
    t: dict[str, float] = {}
    t0 = time.perf_counter()
    products, evidence = gather_context(retriever, question, pet, papers, k_products, k_evidence)
    t["retrieve"] = (time.perf_counter() - t0) * 1000
    vet = VET_NOTE if needs_vet_note(question, pet) else None

    if not products and not evidence:   # nothing to ground an answer in: don't ask the model to guess
        return Answer(question, "No matching products or research were found for this question "
                      "and pet profile.", [], [], vet, n_products=0, n_evidence=0, timings_ms=t)

    system, user = build_prompt(question, pet, products, evidence)
    t1 = time.perf_counter()
    raw = llm(system, user)
    t["generate"] = (time.perf_counter() - t1) * 1000
    if not isinstance(raw, dict):
        raw = {}
    summary, recs, cited, warnings = check_answer(raw, pet, products, evidence)

    sources = [Source(f"E{i}", c.payload.get("source_id") or "", c.payload.get("title") or "",
                      c.payload.get("heading"), c.payload.get("pub_year"), f"E{i}" in cited)
               for i, c in enumerate(evidence, 1)]
    t["total"] = (time.perf_counter() - t0) * 1000
    return Answer(question, summary, recs, sources, vet, warnings,
                  n_products=len(products), n_evidence=len(evidence), timings_ms=t)
