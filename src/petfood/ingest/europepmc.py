"""Fetch open-access veterinary nutrition papers from Europe PMC.

API docs: https://europepmc.org/RestfulWebService
- search (resultType=core) gives metadata + abstract + license
- /{PMCID}/fullTextXML gives JATS full text for OA articles
"""
from __future__ import annotations

import html
import logging
import re
import time
from collections.abc import Iterator

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from petfood.ingest.jats import parse_front_meta, parse_jats
from petfood.ingest.tagging import detect_species, detect_topics
from petfood.models import Paper

log = logging.getLogger(__name__)

BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"

# A paper is in scope only if its TITLE names cats AND something diet-related (D13). Matching
# abstracts let in ~2/3 off-topic papers (plants, fungi, human studies) that mention a cat once.
# The same term lists drive the API query and the local in_scope() check.
CAT_TERMS = ("cat", "cats", "feline", "felines", "kitten", "kittens")
# generic titles ("…in pet foods") are kept only if the abstract mentions cats
GENERIC_TERMS = ("companion animal", "companion animals", "pet food", "pet foods", "petfood")
SPECIES_TERMS = CAT_TERMS + GENERIC_TERMS
DIET_TERMS = (
    "diet", "diets", "dietary", "nutrition", "nutritional", "nutrient", "nutrients", "feeding", "fed",
    "feed", "food", "foods", "petfood", "pet food", "kibble", "meat", "treats", "ingredient",
    "ingredients", "supplement", "supplements", "supplementation", "supplemented", "obesity", "obese",
    "weight loss", "digestibility", "digestible", "taurine",
    "fiber", "fibre", "vitamin", "vitamins", "mineral", "minerals", "calorie", "calories", "caloric",
    "intake", "consumption", "probiotic", "probiotics", "prebiotic", "prebiotics", "fatty acid",
    "fatty acids", "carbohydrate", "carbohydrates", "macronutrient", "macronutrients",
    "metabolizable energy", "hypovitaminosis", "overweight", "body condition",
    # not "gut"/"microbiota"/"diarrhea"/"body weight": they pull in GI-drug and microbiology
    # papers with no diet in them; diet-and-microbiome papers still match on "diet"/"fed"
)


def _or_query(terms) -> str:
    return "(" + " OR ".join(f'"{t}"' if " " in t else t for t in terms) + ")"


def _title_re(terms) -> re.Pattern:
    return re.compile(r"\b(" + "|".join(re.escape(t).replace(r"\ ", r"\s+") for t in terms) + r")\b", re.I)


_CAT_TITLE = _title_re(CAT_TERMS)
_GENERIC_TITLE = _title_re(GENERIC_TERMS)
_DIET_TITLE = _title_re(DIET_TERMS)
_NOT_A_PAPER = re.compile(r"^(correction|erratum|corrigendum|retraction|retracted|expression of concern)\b", re.I)

DEFAULT_QUERY = (
    f"TITLE:{_or_query(SPECIES_TERMS)} AND TITLE:{_or_query(DIET_TERMS)} "
    "AND OPEN_ACCESS:y AND IN_EPMC:y AND PUB_YEAR:[2005 TO 2026]"
)


# Europe PMC search ignores case, so "CAT" (the catalase gene) matches "cat"; reject it locally.
_ACRONYM_ONLY = re.compile(r"\bCATS?\b")


def in_scope(title: str, abstract: str | None = None) -> bool:
    """Same rule as DEFAULT_QUERY, checked locally (custom --query runs, pruning old papers)."""
    if _NOT_A_PAPER.match(title) or not _DIET_TITLE.search(title):
        return False
    species_title = _ACRONYM_ONLY.sub(" ", title)
    if _CAT_TITLE.search(species_title):
        return True
    return bool(_GENERIC_TITLE.search(title) and _CAT_TITLE.search(_ACRONYM_ONLY.sub(" ", abstract or "")))


# Only Creative Commons content goes into the corpus (we store and quote full text).
ALLOWED_LICENSE_PREFIX = ("cc by", "cc0", "cc-by")


class _Retryable(Exception):
    pass


class EuropePMCClient:
    def __init__(self, contact_email: str, client: httpx.Client | None = None, delay_s: float = 0.2):
        self.client = client or httpx.Client(
            timeout=30, headers={"User-Agent": f"petfood-rag/0.1 (mailto:{contact_email})"}
        )
        self.delay_s = delay_s

    @retry(retry=retry_if_exception_type((_Retryable, httpx.TransportError)),
           wait=wait_exponential(min=1, max=30), stop=stop_after_attempt(5), reraise=True)
    def _get(self, url: str, params: dict | None = None) -> httpx.Response:
        r = self.client.get(url, params=params)
        if r.status_code == 429 or r.status_code >= 500:
            raise _Retryable(f"{r.status_code} for {url}")
        time.sleep(self.delay_s)
        return r

    def search(self, query: str = DEFAULT_QUERY, limit: int = 200, page_size: int = 100) -> Iterator[dict]:
        cursor, seen = "*", 0
        while seen < limit:
            r = self._get(f"{BASE}/search", params={
                "query": query, "resultType": "core", "format": "json",
                "pageSize": min(page_size, 1000), "cursorMark": cursor,   # default sort = relevance (D13)
            })
            r.raise_for_status()
            data = r.json()
            results = data.get("resultList", {}).get("result", [])
            if not results:
                return
            for rec in results:
                yield rec
                seen += 1
                if seen >= limit:
                    return
            nxt = data.get("nextCursorMark")
            if not nxt or nxt == cursor:
                return
            cursor = nxt

    def full_text_xml(self, pmcid: str) -> bytes | None:
        r = self._get(f"{BASE}/{pmcid}/fullTextXML")
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.content


def license_ok(license: str | None) -> bool:
    return bool(license) and license.lower().strip().startswith(ALLOWED_LICENSE_PREFIX)


def record_to_paper(rec: dict) -> Paper | None:
    """Map a Europe PMC 'core' search record to a Paper (without full text). None = skip."""
    pmcid = rec.get("pmcid")
    # titles arrive with markup, sometimes double-escaped: "&lt;i&gt;Hermetia&lt;/i&gt;"
    title = re.sub(r"<[^>]+>", "", html.unescape(rec.get("title") or "")).strip().rstrip(".")
    if not pmcid or not title or not in_scope(title, rec.get("abstractText")):
        return None
    abstract = rec.get("abstractText")
    probe = f"{title} {abstract or ''}"
    species = detect_species(probe)
    authors = [a.get("fullName") for a in rec.get("authorList", {}).get("author", []) if a.get("fullName")]
    journal = (rec.get("journalInfo") or {}).get("journal", {}).get("title")
    year = rec.get("pubYear")
    return Paper(
        source_id=pmcid,
        pmid=rec.get("pmid"),
        doi=rec.get("doi"),
        title=title,
        abstract=_strip_tags(abstract) if abstract else None,
        authors=authors,
        journal=journal,
        pub_year=int(year) if year and str(year).isdigit() else None,
        license=(rec.get("license") or "").lower() or None,
        species=species,
        topics=detect_topics(probe),
    )


def attach_full_text(paper: Paper, xml: bytes, s3_key: str) -> Paper:
    abstract, sections = parse_jats(xml)
    paper.sections = sections
    paper.fulltext_available = any(s.section_type != "abstract" for s in sections)
    paper.abstract = paper.abstract or abstract
    paper.s3_key = s3_key
    paper.received_year = parse_front_meta(xml)["received_year"]
    # topics from full text are richer than from the abstract alone
    body = " ".join(s.text for s in sections[:6])
    paper.topics = sorted(set(paper.topics) | set(detect_topics(body)))
    return paper


def _strip_tags(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s)).strip()
