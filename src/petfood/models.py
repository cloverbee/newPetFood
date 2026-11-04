"""Domain models shared by ingestion, retrieval and the API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class Section(BaseModel):
    ord: int
    section_type: str
    heading: str | None = None
    text: str


class Paper(BaseModel):
    source: str = "europepmc"
    source_id: str
    pmid: str | None = None
    doi: str | None = None
    title: str
    abstract: str | None = None
    authors: list[str] = Field(default_factory=list)
    journal: str | None = None
    pub_year: int | None = None
    received_year: int | None = None   # JATS history date; upper bound for the study period
    license: str | None = None
    species: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    s3_key: str | None = None
    fulltext_available: bool = False
    sections: list[Section] = Field(default_factory=list)


class Product(BaseModel):
    source: str
    source_id: str
    name: str
    brand: str | None = None
    species: str | None = None
    life_stage: str | None = None
    food_form: str | None = None
    ingredients_text: str | None = None
    ingredients: list[str] = Field(default_factory=list)
    protein_sources: list[str] = Field(default_factory=list)
    allergen_tags: list[str] = Field(default_factory=list)
    grain_free: bool | None = None
    protein_pct: float | None = None
    fat_pct: float | None = None
    fiber_pct: float | None = None
    moisture_pct: float | None = None
    kcal_per_100g: float | None = None
    labels: list[str] = Field(default_factory=list)
    url: str | None = None
    image_url: str | None = None
    raw: dict | None = None
