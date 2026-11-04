import json
from pathlib import Path

import boto3
import httpx
import pytest
import respx
from moto import mock_aws

from petfood.ingest import europepmc, openpetfoodfacts
from petfood.ingest.curated import load_csv
from petfood.ingest.jats import parse_jats
from petfood.ingest.tagging import (detect_food_form, detect_life_stage, detect_species, detect_topics,
                                    split_ingredients, tag_ingredients)

FIX = Path(__file__).parent / "fixtures"


# ───────────── tagging ─────────────
def test_split_ingredients_handles_parentheses_and_percentages():
    text = "Ingredients: Chicken (24%), rice, animal fat (preserved with tocopherols, rosemary), _egg_."
    assert split_ingredients(text) == [
        "chicken", "rice", "animal fat (preserved with tocopherols, rosemary)", "egg"]


@pytest.mark.parametrize("text,expected_proteins,expected_allergens,grain_free", [
    ("Salmon, rice, chicken fat, fish oil", ["chicken", "fish"], ["chicken", "fish", "grain"], False),
    ("Duck, peas, potato. Grain-free.", ["duck"], ["duck"], True),
    ("Beef, wheat flour, milk powder, soybean meal", ["beef"], ["beef", "dairy", "gluten", "grain", "soy"], False),
    ("Turkey, eggplant, pigment-free colour", ["turkey"], ["turkey"], True),   # no egg / pig false positives
    ("Lamb, chickpeas, buckwheat", ["lamb"], ["lamb"], True),                  # chickpea != chicken
])
def test_tag_ingredients(text, expected_proteins, expected_allergens, grain_free):
    t = tag_ingredients(text)
    assert t["protein_sources"] == expected_proteins
    assert t["allergen_tags"] == expected_allergens
    assert t["grain_free"] is grain_free


def test_tag_ingredients_unknown_without_list_but_scans_name():
    t = tag_ingredients(None, extra_text="Tuna Pate for Cats")
    assert t["protein_sources"] == ["fish"]
    assert t["grain_free"] is None


def test_species_topics_lifestage_form():
    assert detect_species("Obesity in domestic cats and dogs") == ["cat", "dog"]
    assert detect_species("Catalytic converters") == []
    assert {"obesity", "renal"} <= set(detect_topics("Obesity and chronic kidney disease"))
    assert detect_life_stage("Kitten growth formula") == "kitten"
    assert detect_life_stage("Senior 11+ chicken") == "senior"
    assert detect_food_form("Chicken in gravy pouch") == "wet"
    assert detect_food_form("Adult dry kibble") == "dry"


# ───────────── JATS ─────────────
def test_parse_jats_sections():
    abstract, sections = parse_jats((FIX / "sample_jats.xml").read_bytes())
    assert "feline obesity" in abstract
    types = [s.section_type for s in sections]
    assert types == ["abstract", "introduction", "methods", "results", "discussion", "conclusion",
                     "table", "figure"]
    methods = sections[2].text
    assert "Animals:" in methods and "45% protein" in methods
    assert "Crude protein" not in methods                         # table not flattened into prose
    table = sections[6]
    assert table.heading == "Table 1 Diet composition (dry matter basis)"
    assert "| Nutrient | Diet A | Diet B |" in table.text and "|---|---|---|" in table.text
    assert "| Crude protein (%) | 45.2 | 32.1 |" in table.text and "Values are means." in table.text
    assert sections[7].text.startswith("Figure 1 Body weight change")
    assert all("None declared" not in s.text for s in sections)   # COI section dropped
    assert [s.ord for s in sections] == list(range(len(sections)))


# ───────────── Europe PMC ─────────────
def test_record_to_paper_filters_and_maps():
    recs = json.loads((FIX / "epmc_search_page.json").read_text())["resultList"]["result"]
    p1, p2, p3 = (europepmc.record_to_paper(r) for r in recs)
    assert p1.source_id == "PMC0000001" and p1.title.endswith("cats")
    assert p1.species == ["cat"] and "obesity" in p1.topics
    assert p1.authors == ["Doe J", "Roe R"] and p1.pub_year == 2021
    assert "<i>" not in p1.abstract
    assert p2 is None                          # human-only paper dropped
    assert p3 is not None and not europepmc.license_ok(p3.license)  # no license -> excluded


@pytest.mark.parametrize("title, ok", [
    ("Cats on dry kibble diet have significantly different microbiome than those on canned wet food", True),
    ("Carbohydrate source modulates protein and amino acid digestibility patterns in domestic cats", True),
    ("Nutritional Evaluation of Commercial Dog and Cat Foods Based on Key Nutrient Requirements", True),
    ("Overweight in adult cats: a cross-sectional study", True),
    # cats named, nothing about diet
    ("Effect of Body Weight on Echocardiographic Measurements in Pure-Bred Cats", False),
    ("The fecal microbiome in cats with diarrhea", False),
    # dogs only: the corpus is cats only
    ("Weight loss outcomes in obese dogs fed a high-protein diet", False),
    # diet named, no cats (these filled the old corpus)
    ("Effects of manganese-excess on CO2 assimilation in Citrus grandis seedlings", False),
    ("Dairy cows fed rumen-protected fat", False),
    # acronyms: CAT = catalase, PET = imaging / plastic
    ("Catalase (CAT) gene variants in an obese population", False),
    ("PET microplastics affect human gut microbiota during simulated digestion of food", False),
    # notices, not papers
    ("Correction to: Nutritional recommendations for cats with protein-losing enteropathy", False),
])
def test_in_scope_needs_cats_and_diet_in_title(title, ok):
    assert europepmc.in_scope(title) is ok


def test_in_scope_generic_pet_food_title_needs_cats_in_abstract():
    title = "Using gross energy improves metabolizable energy predictive equations for pet foods"
    assert europepmc.in_scope(title, "Foods for cats and dogs were analysed.")
    assert not europepmc.in_scope(title, "Foods for dogs were analysed.")
    assert not europepmc.in_scope(title)


def test_record_to_paper_cleans_escaped_title_markup():
    rec = {"pmcid": "PMC9", "title": "Black soldier fly (&lt;i&gt;Hermetia illucens&lt;/i&gt;) larvae fat in cat diets.",
           "abstractText": "Dogs were fed diets.", "pubYear": "2023", "license": "cc by"}
    assert europepmc.record_to_paper(rec).title == "Black soldier fly (Hermetia illucens) larvae fat in cat diets"


def test_default_query_matches_titles_only():
    q = europepmc.DEFAULT_QUERY
    assert "ABSTRACT:" not in q and q.count("TITLE:") == 2 and '"pet food"' in q


@respx.mock
def test_epmc_client_paginates_and_respects_limit():
    page = json.loads((FIX / "epmc_search_page.json").read_text())
    route = respx.get(f"{europepmc.BASE}/search").mock(return_value=httpx.Response(200, json=page))
    client = europepmc.EuropePMCClient("t@example.com", delay_s=0)
    out = list(client.search(limit=5, page_size=3))
    assert len(out) == 5 and route.call_count == 2


@respx.mock
def test_epmc_full_text_404_returns_none():
    respx.get(f"{europepmc.BASE}/PMC9/fullTextXML").mock(return_value=httpx.Response(404))
    assert europepmc.EuropePMCClient("t@example.com", delay_s=0).full_text_xml("PMC9") is None


def test_attach_full_text_adds_sections_and_topics():
    rec = json.loads((FIX / "epmc_search_page.json").read_text())["resultList"]["result"][0]
    paper = europepmc.record_to_paper(rec)
    europepmc.attach_full_text(paper, (FIX / "sample_jats.xml").read_bytes(), "papers/x.xml")
    assert paper.fulltext_available and paper.s3_key == "papers/x.xml"
    assert len(paper.sections) == 8


# ───────────── Open Pet Food Facts ─────────────
def test_opff_record_to_product():
    rec = json.loads((FIX / "opff_product.json").read_text())
    p = openpetfoodfacts.record_to_product(rec)
    assert p.species == "cat" and p.brand == "SyntheticBrand"
    assert p.food_form == "dry" and p.life_stage == "adult"
    assert p.protein_pct == 32 and p.fat_pct == 14
    assert p.kcal_per_100g == pytest.approx(382.41, abs=0.01)    # from kJ
    assert set(p.allergen_tags) >= {"fish", "chicken", "egg", "grain"}
    assert p.ingredients[0] == "salmon"


def test_opff_rejects_non_pet_and_unnamed():
    assert openpetfoodfacts.record_to_product({"code": "1", "product_name": "x",
                                               "categories_tags": ["en:bird-food"]}) is None
    assert openpetfoodfacts.record_to_product({"code": "1", "categories_tags": ["en:cat-food"]}) is None


def test_opff_dump_reader(tmp_path):
    import gzip
    f = tmp_path / "dump.jsonl.gz"
    with gzip.open(f, "wt") as g:
        g.write((FIX / "opff_product.json").read_text().replace("\n", " ") + "\n\n")
    assert [r["code"] for r in openpetfoodfacts.iter_dump(f)] == ["0000000000001"]


# ───────────── curated CSV ─────────────
def test_curated_csv(tmp_path):
    f = tmp_path / "c.csv"
    f.write_text(
        "source_id,name,brand,species,life_stage,food_form,ingredients_text,protein_pct,fat_pct,"
        "fiber_pct,moisture_pct,kcal_per_100g,labels,url\n"
        'demo-1,Demo Renal Support,DemoCo,cat,,dry,"Rice, chicken fat, egg",28%,15,,,400,veterinary-diet|aafco-complete,\n'
    )
    [p] = load_csv(f)
    assert p.protein_pct == 28 and p.labels == ["veterinary-diet", "aafco-complete"]
    assert p.allergen_tags == ["chicken", "egg", "grain"]
    f.write_text("source_id,name,species\nx,y,horse\n")
    with pytest.raises(ValueError):
        load_csv(f)


# ───────────── object storage ─────────────
@mock_aws
def test_object_store_roundtrip():
    from petfood.config import Settings
    from petfood.storage import ObjectStore
    s = Settings(s3_endpoint_url=None, s3_bucket="test-bucket")
    store = ObjectStore(s, client=boto3.client("s3", region_name="us-east-1"))
    store.ensure_bucket()
    store.put_bytes("papers/a.xml", b"<x/>")
    assert store.exists("papers/a.xml") and store.get_bytes("papers/a.xml") == b"<x/>"
    assert not store.exists("papers/missing.xml")
    store.delete("papers/a.xml")
    assert not store.exists("papers/a.xml")
