"""Load hand-curated products from CSV (e.g. copied from manufacturer pages).

Columns (header row required; only source_id, name, species are mandatory):
source_id,name,brand,species,life_stage,food_form,ingredients_text,
protein_pct,fat_pct,fiber_pct,moisture_pct,kcal_per_100g,labels,url

`labels` is a `|`-separated list, e.g. "aafco-complete|veterinary-diet".
Guaranteed analysis values are as-fed %. kcal_per_100g = (kcal/kg) / 10.
"""
from __future__ import annotations

import csv
from pathlib import Path

from petfood.ingest.tagging import detect_food_form, detect_life_stage, split_ingredients, tag_ingredients
from petfood.models import Product

COLUMNS = ["source_id", "name", "brand", "species", "life_stage", "food_form", "ingredients_text",
           "protein_pct", "fat_pct", "fiber_pct", "moisture_pct", "kcal_per_100g", "labels", "url"]


def _f(v: str | None) -> float | None:
    v = (v or "").strip().rstrip("%")
    return float(v) if v else None


def load_csv(path: str | Path) -> list[Product]:
    out: list[Product] = []
    with open(path, newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f), start=2):
            row = {k: (v or "").strip() for k, v in row.items() if k}
            if not row.get("source_id") or not row.get("name"):
                continue
            species = row.get("species", "").lower()
            if species not in ("cat", "dog"):
                raise ValueError(f"{path}:{i}: species must be 'cat' or 'dog', got {species!r}")
            ing = row.get("ingredients_text") or None
            out.append(Product(
                source="curated",
                source_id=row["source_id"],
                name=row["name"],
                brand=row.get("brand") or None,
                species=species,
                life_stage=row.get("life_stage") or detect_life_stage(row["name"]),
                food_form=row.get("food_form") or detect_food_form(row["name"]),
                ingredients_text=ing,
                ingredients=split_ingredients(ing),
                protein_pct=_f(row.get("protein_pct")),
                fat_pct=_f(row.get("fat_pct")),
                fiber_pct=_f(row.get("fiber_pct")),
                moisture_pct=_f(row.get("moisture_pct")),
                kcal_per_100g=_f(row.get("kcal_per_100g")),
                labels=[x.strip() for x in row.get("labels", "").split("|") if x.strip()],
                url=row.get("url") or None,
                **tag_ingredients(ing, extra_text=row["name"]),
            ))
    return out
