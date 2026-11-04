"""Pet food products from Open Pet Food Facts (open database, ODbL license).

Two ways in:
  1. Search API  — fine for a few hundred products (rate limit: ~10 search req/min).
  2. Full dump   — https://static.openpetfoodfacts.org/data/openpetfoodfacts-products.jsonl.gz
                   download once, then `petfood-ingest products --from-file <path>`.
"""
from __future__ import annotations

import gzip
import json
import logging
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from petfood.ingest.tagging import detect_food_form, detect_life_stage, split_ingredients, tag_ingredients
from petfood.models import Product

log = logging.getLogger(__name__)

SEARCH_URL = "https://world.openpetfoodfacts.org/api/v2/search"
FIELDS = ",".join([
    "code", "product_name", "product_name_en", "generic_name_en", "brands", "categories_tags",
    "ingredients_text", "ingredients_text_en", "nutriments", "labels_tags", "image_front_url",
    "quantity", "lang",
])
CATEGORY_BY_SPECIES = {"cat": "en:cat-food", "dog": "en:dog-food"}


class OPFFClient:
    def __init__(self, contact_email: str, client: httpx.Client | None = None, delay_s: float = 6.5):
        self.client = client or httpx.Client(
            timeout=60, headers={"User-Agent": f"petfood-rag/0.1 (mailto:{contact_email})"}
        )
        self.delay_s = delay_s  # stay under ~10 search requests / minute

    @retry(wait=wait_exponential(min=5, max=60), stop=stop_after_attempt(4), reraise=True)
    def _page(self, category: str, page: int, page_size: int, country: str | None = None) -> list[dict]:
        params = {"categories_tags": category, "fields": FIELDS, "page": page,
                  "page_size": page_size, "sort_by": "unique_scans_n"}
        if country:  # e.g. "united-states" -> only products sold in the US
            params["countries_tags"] = f"en:{country}"
        r = self.client.get(SEARCH_URL, params=params)
        r.raise_for_status()
        return r.json().get("products", [])

    def search(self, species: str, limit: int = 500, page_size: int = 100,
               country: str | None = None) -> Iterator[dict]:
        category = CATEGORY_BY_SPECIES[species]
        page, seen = 1, 0
        while seen < limit:
            products = self._page(category, page, page_size, country)
            if not products:
                return
            for p in products:
                yield p
                seen += 1
                if seen >= limit:
                    return
            page += 1
            time.sleep(self.delay_s)


def iter_dump(path: str | Path) -> Iterator[dict]:
    """Stream products from the OPFF JSONL dump (.jsonl or .jsonl.gz)."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _species_from_categories(tags: list[str]) -> str | None:
    joined = " ".join(tags)
    is_cat = "cat-food" in joined or "for-cats" in joined
    is_dog = "dog-food" in joined or "for-dogs" in joined
    if is_cat and not is_dog:
        return "cat"
    if is_dog and not is_cat:
        return "dog"
    return None


def _num(nutr: dict, *keys: str, max_value: float = 100) -> float | None:
    """First parseable value among keys; values outside [0, max_value] are treated as bad data."""
    for k in keys:
        v = nutr.get(k)
        if v in (None, ""):
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if 0 <= f <= max_value:
            return round(f, 2)
    return None


def record_to_product(rec: dict, species_hint: str | None = None) -> Product | None:
    """Map an OPFF record to a Product. None = unusable (no name/barcode or not cat/dog)."""
    code = str(rec.get("code") or "").strip()
    name = (rec.get("product_name_en") or rec.get("product_name") or rec.get("generic_name_en") or "").strip()
    if not code or not name:
        return None
    cats = rec.get("categories_tags") or []
    species = _species_from_categories(cats) or species_hint
    if species not in ("cat", "dog"):
        return None

    ing_text = (rec.get("ingredients_text_en") or rec.get("ingredients_text") or "").strip() or None
    brand = (rec.get("brands") or "").split(",")[0].strip() or None
    descriptor = f"{name} {' '.join(cats)}"
    tags = tag_ingredients(ing_text, extra_text=name)
    nutr = rec.get("nutriments") or {}
    energy_kcal = _num(nutr, "energy-kcal_100g", max_value=1000)
    if energy_kcal is None and (kj := _num(nutr, "energy-kj_100g", "energy_100g", max_value=4200)) is not None:
        energy_kcal = round(kj / 4.184, 2)

    return Product(
        source="openpetfoodfacts",
        source_id=code,
        name=name,
        brand=brand,
        species=species,
        life_stage=detect_life_stage(descriptor),
        food_form=detect_food_form(descriptor.replace("-", " ")),
        ingredients_text=ing_text,
        ingredients=split_ingredients(ing_text),
        protein_pct=_num(nutr, "crude-protein_100g", "proteins_100g", "proteins"),
        fat_pct=_num(nutr, "crude-fat_100g", "fat_100g", "fat"),
        fiber_pct=_num(nutr, "crude-fibre_100g", "crude-fiber_100g", "fiber_100g", "fiber"),
        moisture_pct=_num(nutr, "moisture_100g", "water_100g", "moisture"),
        kcal_per_100g=energy_kcal,
        labels=[t.split(":", 1)[-1] for t in rec.get("labels_tags") or []],
        url=f"https://world.openpetfoodfacts.org/product/{code}",
        image_url=rec.get("image_front_url"),
        raw={k: rec.get(k) for k in ("code", "brands", "categories_tags", "quantity", "lang")},
        **tags,
    )
