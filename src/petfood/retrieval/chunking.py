"""Turn papers (sections) and products into retrievable chunks.

Papers: sentence-aware sliding windows (~300 words, ~50 overlap) inside each section, so a
chunk never mixes e.g. Methods and Discussion. Each chunk gets a short contextual header
(title + section) — this "contextual chunk" trick helps both BM25 and dense retrieval when a
passage alone says "the cats" without saying which study.

Products: one chunk per product, rendered as a compact label card.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field

# Deterministic ids -> re-indexing the same doc overwrites instead of duplicating.
_NS = uuid.UUID("6f1c2a52-6b0e-4b8e-9a77-5d1f1f0c9e01")

# Sections that rarely answer user questions; still indexed but marked so we can down-weight.
LOW_VALUE_SECTIONS = {"methods"}


@dataclass
class Chunk:
    doc_type: str          # paper | product
    doc_id: int
    ord: int
    text: str              # what gets embedded / BM25-indexed (header + body)
    payload: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return str(uuid.uuid5(_NS, f"{self.doc_type}:{self.doc_id}:{self.ord}"))

    @property
    def token_count(self) -> int:
        return len(self.text.split())


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def window_sentences(text: str, max_words: int = 300, overlap_words: int = 50) -> list[str]:
    """Greedy sentence packing into ~max_words windows with ~overlap_words carried over."""
    sents = split_sentences(text)
    windows: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for s in sents:
        n = len(s.split())
        if n > max_words:  # pathological very long "sentence" (tables flattened etc.)
            words = s.split()
            for i in range(0, len(words), max_words - overlap_words):
                piece = " ".join(words[i:i + max_words])
                if cur:
                    windows.append(" ".join(cur))
                    cur, cur_len = [], 0
                windows.append(piece)
            continue
        if cur_len + n > max_words and cur:
            windows.append(" ".join(cur))
            # carry the tail sentences as overlap
            carry, carry_len = [], 0
            for prev in reversed(cur):
                pl = len(prev.split())
                if carry_len + pl > overlap_words:
                    break
                carry.insert(0, prev)
                carry_len += pl
            cur, cur_len = carry, carry_len
        cur.append(s)
        cur_len += n
    if cur:
        windows.append(" ".join(cur))
    return windows


def window_table(text: str, max_words: int = 300) -> list[str]:
    """Split a rendered table by rows; every piece repeats the caption + header rows so a
    chunk like "| Crude protein | 32.5 |" is never separated from what table it belongs to."""
    lines = text.split("\n")
    # header = caption line(s) up to and including the markdown separator, else caption + first row
    sep = next((i for i, l in enumerate(lines) if l.startswith("|---")), None)
    if sep is not None:
        head, rows = lines[: sep + 1], lines[sep + 1:]
    else:
        first_row = next((i for i, l in enumerate(lines) if l.startswith("|")), len(lines))
        head, rows = lines[: first_row + 1], lines[first_row + 1:]
    head_len = sum(len(l.split()) for l in head)
    budget = max(max_words - head_len, 50)
    pieces, cur, cur_len = [], [], 0
    for r in rows:
        n = len(r.split())
        if cur and cur_len + n > budget:
            pieces.append("\n".join(head + cur))
            cur, cur_len = [], 0
        cur.append(r)
        cur_len += n
    if cur or not pieces:
        pieces.append("\n".join(head + cur))
    return pieces


def study_year_latest(paper: dict) -> int | None:
    """Latest year the study could have run: the stated end, else received, else published.
    A study can't end after the manuscript was submitted, so this bound holds even when the
    paper never says when its data was collected."""
    return paper.get("study_year_end") or paper.get("received_year") or paper.get("pub_year")


def chunk_paper(paper: dict, sections: list[dict], max_words: int = 300,
                overlap_words: int = 50, min_words: int = 25) -> list[Chunk]:
    """paper: row from `papers`; sections: rows from `paper_sections` ordered by ord."""
    chunks: list[Chunk] = []
    title = paper["title"]
    base_payload = {
        "doc_type": "paper",
        "doc_id": paper["id"],
        "source_id": paper["source_id"],
        "title": title,
        "species": paper.get("species") or [],
        "topics": paper.get("topics") or [],
        "pub_year": paper.get("pub_year"),
        # study period: start is NULL unless stated; latest falls back to received/published year
        "study_year_start": paper.get("study_year_start"),
        "study_year_end": paper.get("study_year_end"),
        "study_year_latest": study_year_latest(paper),
        "journal": paper.get("journal"),
        "license": paper.get("license"),
        "doi": paper.get("doi"),
    }
    for sec in sections:
        heading = sec.get("heading") or sec["section_type"].title()
        if sec["section_type"] == "table":
            windows = window_table(sec["text"], max_words)
        else:
            windows = window_sentences(sec["text"], max_words, overlap_words)
        for body in windows:
            n = len(body.split())
            # drop tiny trailing windows (mostly overlap), but keep a short section that stands
            # alone, e.g. a two-sentence Conclusion
            if n < 5 or (n < min_words and len(windows) > 1):
                continue
            chunks.append(Chunk(
                doc_type="paper",
                doc_id=paper["id"],
                ord=len(chunks),
                text=f"{title} — {heading}\n{body}",
                payload=base_payload | {
                    "section_type": sec["section_type"],
                    "heading": heading,
                    "low_value": sec["section_type"] in LOW_VALUE_SECTIONS,
                    "body": body,
                },
            ))
    return chunks


def _fmt(v, unit="%") -> str | None:
    return None if v is None else f"{float(v):g}{unit}"


def render_product(p: dict) -> str:
    """Compact, consistently-ordered label card (good for both BM25 and embeddings)."""
    lines = [f"{p['name']}" + (f" by {p['brand']}" if p.get("brand") else "")]
    facts = [
        f"for {p['species']}s" if p.get("species") else None,
        f"life stage: {p['life_stage']}" if p.get("life_stage") else None,
        f"{p['food_form']} food" if p.get("food_form") else None,
        "grain-free" if p.get("grain_free") else None,
    ]
    lines.append(", ".join(f for f in facts if f))
    ga = [f"{label} {_fmt(p.get(key))}" for label, key in
          [("protein", "protein_pct"), ("fat", "fat_pct"), ("fiber", "fiber_pct"), ("moisture", "moisture_pct")]
          if p.get(key) is not None]
    if p.get("kcal_per_100g") is not None:
        ga.append(f"{float(p['kcal_per_100g']):g} kcal/100g")
    if ga:
        lines.append("Guaranteed analysis: " + ", ".join(ga))
    if p.get("protein_sources"):
        lines.append("Protein sources: " + ", ".join(p["protein_sources"]))
    if p.get("ingredients_text"):
        lines.append("Ingredients: " + p["ingredients_text"][:1500])
    if p.get("labels"):
        lines.append("Labels: " + ", ".join(p["labels"][:10]))
    return "\n".join(line for line in lines if line)


def chunk_product(p: dict) -> Chunk:
    num = lambda k: float(p[k]) if p.get(k) is not None else None  # noqa: E731
    return Chunk(
        doc_type="product",
        doc_id=p["id"],
        ord=0,
        text=render_product(p),
        payload={
            "doc_type": "product",
            "doc_id": p["id"],
            "source": p["source"],
            "name": p["name"],
            "brand": p.get("brand"),
            "species": [p["species"]] if p.get("species") else [],
            "life_stage": p.get("life_stage"),
            "food_form": p.get("food_form"),
            "grain_free": p.get("grain_free"),
            "allergen_tags": p.get("allergen_tags") or [],
            "protein_sources": p.get("protein_sources") or [],
            "has_ingredients": bool(p.get("ingredients_text")),
            "has_nutrients": p.get("protein_pct") is not None,
            "protein_pct": num("protein_pct"),
            "fat_pct": num("fat_pct"),
            "fiber_pct": num("fiber_pct"),
            "moisture_pct": num("moisture_pct"),
            "kcal_per_100g": num("kcal_per_100g"),
            "url": p.get("url"),
        },
    )
