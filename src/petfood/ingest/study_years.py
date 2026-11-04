"""When was a study's data collected? Extract it from the paper text.

JATS has no field for this, and most papers never state it. Three steps:

1. Regex: keep sentences that still mention a plausible year once citations like
   "(Smith et al., 2014)" or "AAFCO, 2016" are stripped. Most papers have none and stop here.
2. LLM: one call per paper decides whether any sentence dates *this* study (not a cited
   study, a literature search or a guideline) and points at the sentences by id.
3. Code checks the answer: the years must appear in the cited sentences, start <= end, and
   the study must end no later than the received year (else the publication year).

The LLM is passed in as a callable so tests don't hit the API.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from petfood.retrieval.chunking import split_sentences

LLM = Callable[[str, str], dict]   # (system, user) -> parsed JSON

_YEAR = re.compile(r"\b(19[5-9]\d|20\d\d)\b")
_MONTHS = {"january", "february", "march", "april", "may", "june", "july", "august", "september",
           "october", "november", "december", "jan", "feb", "mar", "apr", "jun", "jul", "aug",
           "sep", "sept", "oct", "nov", "dec", "spring", "summer", "autumn", "fall", "winter"}
# "Smith, 2014" · "Smith et al. 2014" · "Dubey and Urban, 1990" · "Smith (2014)"
_CITATION = re.compile(
    r"\b([A-Z][A-Za-z'\-]+)(?:\s+et\s+al\.?|\s+(?:and|&)\s+[A-Z][A-Za-z'\-]+)?,?\s+\(?(?:19|20)\d\d[a-z]?\)?")
# leftovers of citations split mid-sentence ("(2013) showed…") and numbers with units ("2000 ng/g")
_NOT_A_YEAR = re.compile(
    r"^\s*\(?(?:19|20)\d\d[a-z]?\)?|\b(?:19|20)\d\d\s*(?:ng|µg|mg|kg|g|ml|mL|l|kcal|kJ|IU|ppm|ppb|bp|cells|%)\b")
MIN_SENTENCE_WORDS = 5

# Reviews have no data collection of their own; the model sometimes dates them by the literature
# search ("databases were searched on 14 June 2025"), so they never reach it.
_REVIEW_TITLE = re.compile(r"\b(review|meta-analys[ie]s|scoping|consensus|guidelines?)\b", re.I)

_SCAN_SECTIONS = ("methods", "abstract", "other", "results")   # in priority order
MAX_CANDIDATES = 25
MAX_SENTENCE_CHARS = 500
MAX_STUDY_SPAN = 30


@dataclass
class StudyYears:
    status: str                       # found | no_dates | not_stated | rejected
    start: int | None = None          # set only when status == "found"
    end: int | None = None
    evidence: str | None = None
    note: str | None = None


def _strip_citations(sentence: str) -> str:
    def repl(m: re.Match) -> str:
        return m.group(0) if m.group(1).lower() in _MONTHS else " "
    return _NOT_A_YEAR.sub(" ", _CITATION.sub(repl, sentence))


_SHORT_RANGE = re.compile(r"\b((?:19|20)\d\d)\s*[-–/]\s*(\d\d)\b(?!\d)")


def _cited_years(evidence: str) -> set[int]:
    """Years written in the evidence, expanding short ranges: "1997-99" -> 1999, "2000-02" -> 2002."""
    years = set(_years(evidence, None))
    for full, short in _SHORT_RANGE.findall(evidence):
        end = int(full[:2] + short)
        years.add(end if end >= int(full) else end + 100)
    return years


def _years(text: str, upper_bound: int | None) -> list[int]:
    ys = [int(y) for y in _YEAR.findall(text)]
    return [y for y in ys if upper_bound is None or y <= upper_bound]


def candidate_sentences(sections: list[dict], upper_bound: int | None) -> list[str]:
    """Sentences that may date the study, methods first. Table/figure captions are included
    (first line only, so table rows full of numbers don't flood the list)."""
    texts: list[str] = []
    for stype in _SCAN_SECTIONS:
        texts += [s["text"] for s in sections if s["section_type"] == stype]
    texts += [s["text"].split("\n", 1)[0] for s in sections if s["section_type"] in ("table", "figure")]

    out: list[str] = []
    seen: set[str] = set()
    for text in texts:
        for sent in split_sentences(text):
            if (sent in seen or len(sent.split()) < MIN_SENTENCE_WORDS
                    or not _years(_strip_citations(sent), upper_bound)):
                continue
            seen.add(sent)
            out.append(sent[:MAX_SENTENCE_CHARS])
            if len(out) >= MAX_CANDIDATES:
                return out
    return out


SYSTEM_PROMPT = """You read sentences from a veterinary research paper and decide WHEN THIS STUDY'S \
OWN DATA WAS COLLECTED (animals enrolled, samples or records collected, survey run, feeding trial run).

Ignore years that belong to anything else:
- other studies the paper cites, or earlier work by the same authors
- literature or database searches (reviews, meta-analyses)
- guidelines, standards, software or product versions, publication or approval dates

If no sentence states when this study's data was collected, or the paper is a review or
meta-analysis, return nulls. Do not guess from the publication year.

Reply with JSON only:
{"start": <year or null>, "end": <year or null>, "sentence_ids": [<ids of the sentences that \
state it>], "reason": "<one short sentence>"}
Use the same year for start and end when the study ran within one year."""


def build_prompt(title: str, pub_year: int | None, received_year: int | None,
                 article_type: str | None, candidates: list[str]) -> tuple[str, str]:
    lines = [
        f"Title: {title}",
        f"Article type: {article_type or 'unknown'}",
        f"Received: {received_year or 'unknown'}    Published: {pub_year or 'unknown'}",
        "",
        "Sentences:",
        *(f"[{i}] {s}" for i, s in enumerate(candidates, 1)),
    ]
    return SYSTEM_PROMPT, "\n".join(lines)


def _as_year(v) -> int | None:
    if v is None:
        return None
    if isinstance(v, bool):
        raise ValueError(v)
    return int(v)


def validate(answer: dict, candidates: list[str], upper_bound: int | None) -> StudyYears:
    """Turn the model's JSON into a StudyYears, rejecting anything the text doesn't support."""
    reason = str(answer.get("reason") or "")[:300] or None
    try:
        start, end = _as_year(answer.get("start")), _as_year(answer.get("end"))
    except (TypeError, ValueError):
        return StudyYears("rejected", note=f"non-integer years: {answer!r}"[:300])
    if start is None and end is None:
        return StudyYears("not_stated", note=reason)
    start = start if start is not None else end
    end = end if end is not None else start

    ids = answer.get("sentence_ids") or []
    if not ids or not all(isinstance(i, int) and 1 <= i <= len(candidates) for i in ids):
        return StudyYears("rejected", note=f"{start}–{end}: bad sentence_ids {ids!r}")
    evidence = " … ".join(candidates[i - 1] for i in dict.fromkeys(ids))

    def reject(why: str) -> StudyYears:   # years go in the note only, so nothing filters on them
        return StudyYears("rejected", evidence=evidence, note=f"{start}–{end}: {why}")

    cited_years = _cited_years(evidence)
    if start not in cited_years or end not in cited_years:
        return reject("years not in the cited sentences")
    if start > end:
        return reject("start after end")
    if upper_bound is not None and end > upper_bound:
        return reject(f"ends after {upper_bound} (received/published)")
    if upper_bound is not None and start < upper_bound - MAX_STUDY_SPAN:
        return reject(f"starts more than {MAX_STUDY_SPAN} years before {upper_bound}")
    return StudyYears("found", start, end, evidence, reason)


def extract_study_years(paper: dict, sections: list[dict], article_type: str | None,
                        llm: LLM) -> StudyYears:
    """paper: row from `papers` (needs title, pub_year, received_year)."""
    if (article_type or "").startswith("review") or _REVIEW_TITLE.search(paper["title"]):
        return StudyYears("not_stated", note="review article: no own data collection")
    upper_bound = paper.get("received_year") or paper.get("pub_year")
    candidates = candidate_sentences(sections, upper_bound)
    if not candidates:
        return StudyYears("no_dates")
    system, user = build_prompt(paper["title"], paper.get("pub_year"), paper.get("received_year"),
                                article_type, candidates)
    return validate(llm(system, user), candidates, upper_bound)

