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
from dataclasses import asdict, dataclass, field, replace

from petfood.ingest.tagging import detect_topics
from petfood.retrieval.chunking import split_sentences
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


# Words in a question that name a food form or life stage (D17). A filter is set only when the
# question names exactly one, so "is wet better than dry?" filters nothing.
_FORM_WORDS = {
    "wet": r"\b(wet|canned|cans?|pouch(es)?|pat[eé]s?)\b",
    "dry": r"\b(dry|kibble)\b",
    "raw": r"\b(raw|freeze[- ]dried)\b",
    "treat": r"\b(treats?|snacks?)\b",
}
_STAGE_WORDS = {
    "kitten": r"\b(kittens?|\d+[- ]months?[- ]old)\b",
    "senior": r"\b(senior|elderly|geriatric|older|old cat|aging|ageing)\b",
    "adult": r"\badult\b",
}
_AGE_YEARS = re.compile(r"\b(\d{1,2})[- ]?(?:years?|yrs?)[- ]old\b", re.I)
SENIOR_FROM_YEARS = 11   # cats are commonly called senior from about 11 years


def _only_one(found: set[str]) -> str | None:
    return next(iter(found)) if len(found) == 1 else None


def profile_from_question(question: str, pet: "PetProfile") -> "PetProfile":
    """Fill food form and life stage from the question when the caller left them unset.
    Explicit settings (CLI flags, API fields) always win."""
    q = question.lower()
    form = pet.food_form or _only_one({f for f, pat in _FORM_WORDS.items() if re.search(pat, q)})
    # "4 year old cat" must not also count as "old cat": match stage words with ages removed
    no_ages = _AGE_YEARS.sub(" ", q)
    stages = {s for s, pat in _STAGE_WORDS.items() if re.search(pat, no_ages)}
    for years in _AGE_YEARS.findall(q):
        stages.add("senior" if int(years) >= SENIOR_FROM_YEARS else "adult")
    stage = pet.life_stage or _only_one(stages)
    return replace(pet, food_form=form, life_stage=stage)


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
    model: str | None = None   # which LLM wrote it; set by the caller
    profile: dict = field(default_factory=dict)   # the profile actually used, after reading the question
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
- Only state a number about a product (moisture, protein, calories, ...) if it appears in THAT
  product's own entry. "Not listed" means the value is unknown: say nothing about it.
- Every product listed already matches the pet's species, life stage, food form and allergy
  needs. Missing details are NOT a reason to reject a product: recommend the best 1-3 and say
  briefly what the label doesn't list. Prefer products whose ingredients or nutrients are listed.
  Recommend none only if the question asks for something no listed product can provide.
- Keep the summary to 3-5 sentences in plain language. In the summary and reasons, refer to
  products by name, never by their P label (the owner never sees the labels).

Reply with JSON only:
{"summary": "<answer with [E#] citations>",
 "recommendations": [{"product": "P#", "reason": "<1-2 sentences, may cite [E#]>", "evidence": ["E#"]}]}"""


# Exact reply shape. Local models get it enforced by Ollama (format=<schema>); DeepSeek's JSON mode
# can't take a schema, so for it the prompt's example is the only guide and check_answer() the guard.
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "recommendations": {"type": "array", "items": {
            "type": "object",
            "properties": {"product": {"type": "string"}, "reason": {"type": "string"},
                           "evidence": {"type": "array", "items": {"type": "string"}}},
            "required": ["product", "reason", "evidence"],
        }},
    },
    "required": ["summary", "recommendations"],
}


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


# Product facts the prompt reports as missing when the product has no value for them. The search
# index doesn't carry these lines: "moisture: not listed" on every card would make a keyword
# search for "moisture" match every product.
_FACTS = [("protein", "protein_pct"), ("fat", "fat_pct"), ("fiber", "fiber_pct"),
          ("moisture", "moisture_pct"), ("calories", "kcal_per_100g")]


def _missing_facts(p: dict) -> list[str]:
    missing = [name for name, key in _FACTS if p.get(key) is None]
    if not p.get("has_ingredients"):
        missing.append("ingredients")
    return missing


def product_entry(c: Candidate) -> str:
    """A product exactly as the model sees it (also what the number check compares against)."""
    lines = [c.text[:MAX_PRODUCT_CHARS]]
    if not c.payload.get("life_stage"):
        lines.append("Life stage: adult (the label doesn't say, so it's an adult maintenance food)")
    missing = _missing_facts(c.payload)
    if missing:
        lines.append(f"Not listed: {', '.join(missing)}")
    return "\n".join(lines)


def evidence_entry(c: Candidate) -> str:
    """A research excerpt exactly as the model sees it."""
    p = c.payload
    body = (p.get("body") or c.text)[:MAX_EVIDENCE_CHARS]
    return f"\"{p.get('title')}\" — {p.get('heading')} ({_years(p)})\n{body}"


def build_prompt(question: str, pet: PetProfile, products: list[Candidate],
                 evidence: list[Candidate]) -> tuple[str, str]:
    lines = [f"QUESTION: {question}", f"PET: {_profile_text(pet)}", "", "PRODUCTS:"]
    if not products:
        lines.append("(none matched the pet's constraints)")
    for i, c in enumerate(products, 1):
        lines.append(f"[P{i}] {product_entry(c)}")
    lines += ["", "RESEARCH:"]
    if not evidence:
        lines.append("(no relevant research found)")
    for i, c in enumerate(evidence, 1):
        lines.append(f"[E{i}] {evidence_entry(c)}")
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


# A number not glued to a letter or another digit: matches "84.5", "12", "3" in "3–11%",
# but not the "12" in "B12". Labels like [E1] are removed before matching.
_NUMBER = re.compile(r"(?<![A-Za-z\d.])\d+(?:\.\d+)?")


def _numbers(text: str) -> set[float]:
    return {float(n) for n in _NUMBER.findall(_LABEL.sub(" ", text))}


def _drop_unsupported_numbers(text: str, allowed: set[float], warnings: list[str], where: str) -> str:
    """Remove each sentence that states a number found nowhere in the text it may rely on.
    Catches facts copied from the wrong product (attribute binding errors), see D16."""
    kept = []
    for sent in split_sentences(text):
        bad = _numbers(sent) - allowed
        if bad:
            nums = ", ".join(f"{n:g}" for n in sorted(bad))
            warnings.append(f"{where}: removed sentence with unsupported number(s) {nums}: {sent[:90]!r}")
        else:
            kept.append(sent)
    return " ".join(kept)


def check_answer(raw: dict, pet: PetProfile, products: list[Candidate],
                 evidence: list[Candidate], question: str = "",
                 ) -> tuple[str, list[Recommendation], set[str], list[str]]:
    """Keep only what the retrieved context supports. Returns (summary, recommendations,
    cited E-labels, warnings)."""
    warnings: list[str] = []
    p_by_label = {f"P{i}": c for i, c in enumerate(products, 1)}
    e_labels = {f"E{i}" for i in range(1, len(evidence) + 1)}
    excluded = {a.lower() for a in pet.exclude_allergens}
    # numbers each part of the prompt contains, for the number check
    q_nums = _numbers(question)
    p_nums = {f"P{i}": _numbers(product_entry(c)) for i, c in enumerate(products, 1)}
    e_nums = {f"E{i}": _numbers(evidence_entry(c)) for i, c in enumerate(evidence, 1)}
    all_nums = q_nums.union(*p_nums.values(), *e_nums.values())

    summary = _clean_citations(_name_products(str(raw.get("summary") or ""), p_by_label),
                               e_labels, warnings, "summary")
    # the summary may talk about any product or excerpt, so it's checked against all of them
    summary = _drop_unsupported_numbers(summary, all_nums, warnings, "summary")
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
        # a reason may only use numbers from its own product, the excerpts it cites, or the question
        allowed = p_nums[label] | q_nums
        for e in dict.fromkeys(ev + _LABEL.findall(reason)):
            allowed |= e_nums[e]
        reason = _drop_unsupported_numbers(reason, allowed, warnings, label)
        if not reason:
            warnings.append(f"dropped {label} ({p.get('name')}): no supported reason left")
            continue
        ev = list(dict.fromkeys(ev + _LABEL.findall(reason)))   # inline cites in what's left count too
        recs.append(Recommendation(label, p["doc_id"], p.get("name") or "", p.get("brand"), reason, ev))
        seen.add(label)

    cited = set(_LABEL.findall(summary)) | {e for r in recs for e in r.evidence}
    return summary, recs, cited & e_labels, warnings


# ───────────── 4. the whole flow ─────────────
def answer(question: str, pet: PetProfile, retriever: HybridRetriever, llm: LLM,
           papers: PaperFilters | None = None, k_products: int = 8, k_evidence: int = 5) -> Answer:
    t: dict[str, float] = {}
    t0 = time.perf_counter()
    pet = profile_from_question(question, pet)
    products, evidence = gather_context(retriever, question, pet, papers, k_products, k_evidence)
    t["retrieve"] = (time.perf_counter() - t0) * 1000
    vet = VET_NOTE if needs_vet_note(question, pet) else None

    if not products and not evidence:   # nothing to ground an answer in: don't ask the model to guess
        return Answer(question, "No matching products or research were found for this question "
                      "and pet profile.", [], [], vet, n_products=0, n_evidence=0, timings_ms=t,
                      profile=asdict(pet))

    system, user = build_prompt(question, pet, products, evidence)
    t1 = time.perf_counter()
    raw = llm(system, user)
    t["generate"] = (time.perf_counter() - t1) * 1000
    if not isinstance(raw, dict):
        raw = {}
    summary, recs, cited, warnings = check_answer(raw, pet, products, evidence, question)

    sources = [Source(f"E{i}", c.payload.get("source_id") or "", c.payload.get("title") or "",
                      c.payload.get("heading"), c.payload.get("pub_year"), f"E{i}" in cited)
               for i, c in enumerate(evidence, 1)]
    t["total"] = (time.perf_counter() - t0) * 1000
    return Answer(question, summary, recs, sources, vet, warnings,
                  n_products=len(products), n_evidence=len(evidence), timings_ms=t, profile=asdict(pet))
