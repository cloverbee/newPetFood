"""Retrieval tests: in-memory Qdrant + deterministic fake encoders (no model downloads)."""
import pytest
from qdrant_client import QdrantClient

from petfood.ingest.tagging import detect_topics, tag_ingredients
from petfood.retrieval.chunking import chunk_paper, chunk_product, render_product, window_sentences
from petfood.retrieval.embeddings import HashDense, HashSparse, OverlapReranker
from petfood.retrieval.pipeline import HybridRetriever, dedupe_by_doc, rrf
from petfood.retrieval.store import ChunkStore, SearchFilters

# ───────────── fixtures ─────────────
PAPERS = [
    ({"id": 1, "source_id": "PMC1", "title": "Phosphorus restriction in cats with chronic kidney disease",
      "species": ["cat"], "topics": ["renal"], "pub_year": 2019,
      "received_year": 2018, "study_year_start": 2015, "study_year_end": 2016},
     [{"section_type": "abstract", "heading": "Abstract",
       "text": "Cats with chronic kidney disease (CKD) fed a phosphorus-restricted renal diet survived longer. "
               "Dietary phosphate restriction slowed progression of azotemia in feline CKD patients over two years. "
               "Protein was moderately restricted in the renal diet compared with maintenance diets."},
      {"section_type": "methods", "heading": "Methods",
       "text": "Fifty client-owned cats with stable CKD were enrolled. Serum creatinine and phosphate were measured "
               "monthly. Kidney disease stage was assigned according to IRIS guidelines for every cat enrolled."}]),
    ({"id": 2, "source_id": "PMC2", "title": "Weight loss in obese dogs fed high-protein high-fibre diets",
      "species": ["dog"], "topics": ["obesity"], "pub_year": 2021, "received_year": 2020},
     [{"section_type": "results", "heading": "Results",
       "text": "Obese dogs lost 15% of body weight over 12 weeks on a high-protein, high-fibre diet. "
               "Lean body mass was preserved and satiety improved compared with the control diet in all dogs."}]),
    ({"id": 3, "source_id": "PMC3", "title": "Struvite urolith dissolution with therapeutic urinary diets in cats",
      "species": ["cat"], "topics": ["urinary"], "pub_year": 2015},
     [{"section_type": "abstract", "heading": "Abstract",
       "text": "Struvite uroliths dissolved within 30 days in most cats fed an acidifying urinary diet. "
               "Urine pH and relative supersaturation of struvite decreased with the therapeutic food."}]),
]
PRODUCTS = [
    {"id": 10, "source": "curated", "name": "Renal Care Chicken Dry Cat Food", "brand": "DemoVet",
     "species": "cat", "life_stage": "adult", "food_form": "dry",
     "ingredients_text": "Chicken, rice, chicken fat, egg", "protein_pct": 28, "fat_pct": 15,
     **tag_ingredients("Chicken, rice, chicken fat, egg")},
    {"id": 11, "source": "curated", "name": "Kidney Support Salmon Wet Cat Food", "brand": "DemoVet",
     "species": "cat", "life_stage": "senior", "food_form": "wet",
     "ingredients_text": "Salmon, water, pea protein, fish oil", "protein_pct": 9,
     **tag_ingredients("Salmon, water, pea protein, fish oil")},
    {"id": 12, "source": "curated", "name": "Kidney Support Duck Wet Cat Food", "brand": "DemoVet",
     "species": "cat", "life_stage": "all", "food_form": "wet",
     "ingredients_text": "Duck, water, potato", **tag_ingredients("Duck, water, potato")},
    {"id": 13, "source": "curated", "name": "Light Weight Control Dog Food", "brand": "DemoDog",
     "species": "dog", "life_stage": "adult", "food_form": "dry",
     "ingredients_text": "Turkey, barley, beet pulp", **tag_ingredients("Turkey, barley, beet pulp")},
]


@pytest.fixture(scope="module")
def store():
    s = ChunkStore(QdrantClient(":memory:"), "test", HashDense(dim=256), HashSparse())
    s.ensure_collection()
    chunks = [c for p, secs in PAPERS for c in chunk_paper(p, secs, min_words=5)]
    chunks += [chunk_product(p) for p in PRODUCTS]
    s.upsert(chunks)
    return s


# ───────────── chunking ─────────────
def test_window_sentences_respects_size_and_overlap():
    text = " ".join(f"Sentence number {i} has exactly seven words." for i in range(100))
    wins = window_sentences(text, max_words=70, overlap_words=14)
    assert all(len(w.split()) <= 70 for w in wins)
    assert len(wins) > 10
    # consecutive windows share the overlap sentences
    assert wins[0].split(". ")[-1].rstrip(".") in wins[1]


def test_chunk_paper_has_context_header_and_stable_ids():
    p, secs = PAPERS[0]
    a = chunk_paper(p, secs, min_words=5)
    b = chunk_paper(p, secs, min_words=5)
    assert [c.id for c in a] == [c.id for c in b]                 # deterministic -> idempotent upserts
    assert a[0].text.startswith("Phosphorus restriction in cats") and "Abstract" in a[0].text.split("\n")[0]
    assert a[1].payload["low_value"] is True                       # methods section flagged
    assert len({c.id for c in a}) == len(a)


def test_render_product_card():
    card = render_product(PRODUCTS[0])
    assert "Renal Care Chicken Dry Cat Food by DemoVet" in card
    assert "protein 28%" in card and "Protein sources: chicken, egg" in card
    assert chunk_product(PRODUCTS[0]).payload["allergen_tags"] == ["chicken", "egg", "grain"]


# ───────────── fusion ─────────────
def test_rrf_rewards_agreement():
    fused = dict(rrf({"dense": ["a", "b", "c"], "sparse": ["b", "d", "a"]}, k=60))
    assert fused["b"] > fused["a"] > fused["c"]          # b: ranks 2+1, a: 1+3
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert set(fused) == {"a", "b", "c", "d"}


def test_rrf_weights():
    fused = rrf({"dense": ["a"], "sparse": ["b"]}, weights={"sparse": 2.0})
    assert fused[0][0] == "b"


# ───────────── store + filters ─────────────
def test_sparse_search_finds_exact_term(store):
    hits = store.search_sparse("struvite", 5)
    assert hits and hits[0].payload["source_id"] == "PMC3"


def test_species_and_type_filters(store):
    hits = store.search_dense("diet", 20, SearchFilters(doc_type="paper", species="dog"))
    assert hits and all(h.payload["doc_type"] == "paper" and "dog" in h.payload["species"] for h in hits)


def test_allergen_exclusion_is_hard(store):
    f = SearchFilters(doc_type="product", species="cat", exclude_allergens=["chicken", "fish"])
    names = {h.payload["name"] for h in store.search_dense("kidney cat food", 20, f)}
    names |= {h.payload["name"] for h in store.search_sparse("kidney cat food chicken salmon", 20, f)}
    assert names == {"Kidney Support Duck Wet Cat Food"}


@pytest.fixture(scope="module")
def stage_store():
    """One cat food per label: kitten, adult, senior, all life stages, and no life stage stated."""
    s = ChunkStore(QdrantClient(":memory:"), "stages", HashDense(dim=64), HashSparse())
    s.ensure_collection()
    s.upsert([chunk_product({"id": i, "source": "curated", "name": name, "species": "cat", "life_stage": stage})
              for i, (name, stage) in enumerate([("K", "kitten"), ("A", "adult"), ("S", "senior"),
                                                 ("ALL", "all"), ("NONE", None)], 1)])
    return s


@pytest.mark.parametrize("stage,expected", [
    ("kitten", {"K", "ALL"}),                       # kittens need growth food: unstated labels excluded
    ("adult", {"A", "ALL", "NONE"}),                # unstated label = adult maintenance food (D17)
    ("senior", {"S", "A", "ALL", "NONE"}),          # adult maintenance suits healthy seniors
])
def test_life_stage_rules(stage_store, stage, expected):
    f = SearchFilters(doc_type="product", species="cat", life_stage=stage)
    assert {h.payload["name"] for h in stage_store.search_dense("cat food", 20, f)} == expected


def _paper_ids(store, f):
    return {h.payload["source_id"] for h in store.search_dense("diet", 50, f)}


def test_pub_year_range(store):
    assert _paper_ids(store, SearchFilters(doc_type="paper", pub_year_from=2016)) == {"PMC1", "PMC2"}
    assert _paper_ids(store, SearchFilters(doc_type="paper", pub_year_to=2019)) == {"PMC1", "PMC3"}
    assert _paper_ids(store, SearchFilters(doc_type="paper", pub_year_from=2016, pub_year_to=2019)) == {"PMC1"}


def test_study_year_from_uses_stated_end_else_received_else_published(store):
    # PMC1 ran 2015–2016; PMC2 unstated but received 2020; PMC3 unstated, published 2015
    assert _paper_ids(store, SearchFilters(doc_type="paper", study_year_from=2017)) == {"PMC2"}
    assert _paper_ids(store, SearchFilters(doc_type="paper", study_year_from=2016)) == {"PMC1", "PMC2"}


def test_study_year_to_keeps_papers_with_unknown_start(store):
    assert _paper_ids(store, SearchFilters(doc_type="paper", study_year_to=2014)) == {"PMC2", "PMC3"}
    assert _paper_ids(store, SearchFilters(doc_type="paper", study_year_to=2015)) == {"PMC1", "PMC2", "PMC3"}


def test_year_filters_do_not_drop_products(store):
    hits = store.search_dense("food", 50, SearchFilters(pub_year_from=2020, study_year_from=2019))
    by_type = {}
    for h in hits:
        by_type.setdefault(h.payload["doc_type"], set()).add(h.payload.get("source_id") or h.payload["name"])
    assert by_type["paper"] == {"PMC2"} and len(by_type["product"]) == len(PRODUCTS)


def test_reindex_is_idempotent(store):
    before = store.count()
    p, secs = PAPERS[1]
    store.upsert(chunk_paper(p, secs, min_words=5))
    assert store.count() == before


# ───────────── pipeline ─────────────
def test_hybrid_rerank_pipeline(store):
    r = HybridRetriever(store, OverlapReranker(), k_dense=10, k_sparse=10, k_rerank=10)
    res = r.retrieve("phosphorus restricted renal diet for cats with kidney disease", top_k=3,
                     filters=SearchFilters(doc_type="paper", species="cat"))
    assert res.results[0].payload["source_id"] == "PMC1"
    assert [c.final_rank for c in res.results] == list(range(1, len(res.results) + 1))
    assert all(c.rrf_rank is not None for c in res.candidates)
    assert {"dense", "sparse", "rerank", "total"} <= set(res.timings_ms)


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid", "hybrid_rerank"])
def test_all_modes_run(store, mode):
    r = HybridRetriever(store, OverlapReranker())
    res = r.retrieve("struvite urinary diet", top_k=2, mode=mode)
    assert res.results
    if mode == "dense":
        assert all(c.sparse_rank is None for c in res.candidates)
    if mode != "hybrid_rerank":
        assert all(c.rerank_score is None for c in res.results)


def test_methods_chunks_are_down_weighted(store):
    r = HybridRetriever(store, OverlapReranker(), low_value_penalty=0.1)
    res = r.retrieve("cats kidney disease", top_k=5, filters=SearchFilters(doc_type="paper"), max_per_doc=None)
    methods = [c for c in res.results if c.payload.get("section_type") == "methods"]
    abstract = [c for c in res.results if c.payload.get("source_id") == "PMC1" and c.payload["section_type"] == "abstract"]
    assert abstract and (not methods or methods[0].final_rank > abstract[0].final_rank)


def test_dedupe_by_doc():
    from petfood.retrieval.pipeline import Candidate
    cands = [Candidate(str(i), {"doc_type": "paper", "doc_id": d}) for i, d in enumerate([1, 1, 1, 2, 1, 3])]
    assert [c.payload["doc_id"] for c in dedupe_by_doc(cands, 2)] == [1, 1, 2, 3]


# ───────────── tagging regressions found on real data ─────────────
def test_topic_rules_ignore_ingredient_words():
    t = detect_topics("Chicken liver and beef heart as protein sources for kittens")
    assert not {"hepatic", "cardiac", "growth"} & set(t)
    assert {"hepatic", "cardiac"} <= set(detect_topics("hepatic lipidosis and congestive heart failure"))


def test_table_chunks_keep_header_rows():
    from petfood.retrieval.chunking import window_table
    rows = "\n".join(f"| Nutrient {i} | {i}.5 | {i}.9 |" for i in range(200))
    text = "Table 2 Nutrient profile\n| Nutrient | A | B |\n|---|---|---|\n" + rows
    pieces = window_table(text, max_words=120)
    assert len(pieces) > 3
    assert all(p.startswith("Table 2 Nutrient profile\n| Nutrient | A | B |\n|---|---|---|") for p in pieces)
    body_rows = [l for p in pieces for l in p.split("\n")[3:]]
    assert len(body_rows) == 200                                 # every row exactly once


def test_chunk_paper_routes_tables():
    p = {"id": 9, "source_id": "PMC9", "title": "T", "species": ["cat"]}
    secs = [{"section_type": "table", "heading": "Table 1 Diets",
             "text": "Table 1 Diets\n| Nutrient | A |\n|---|---|\n| Crude protein | 45.2 |\n| Crude fat | 14.0 |"}]
    [c] = chunk_paper(p, secs)
    assert "\n| Crude protein | 45.2 |" in c.text and c.payload["section_type"] == "table"


class _RecordingReranker:
    """Remembers the texts it was asked to score."""
    def __init__(self):
        self.seen = []

    def score(self, query, docs):
        self.seen += list(docs)
        return [0.0] * len(docs)


def test_rerank_max_words_shortens_only_what_the_reranker_reads(store):
    rec = _RecordingReranker()
    r = HybridRetriever(store, rec, rerank_max_words=5)
    res = r.retrieve("diet", top_k=3)
    assert rec.seen and all(len(t.split()) <= 5 for t in rec.seen)
    assert any(len(c.text.split()) > 5 for c in res.results)   # the LLM still gets full text


def test_rerank_max_words_none_reads_full_text(store):
    rec = _RecordingReranker()
    HybridRetriever(store, rec).retrieve("diet", top_k=3)
    assert any(len(t.split()) > 5 for t in rec.seen)
