"""Answer generation tests: in-memory Qdrant + fake encoders + a fake LLM (no network)."""
import pytest
from qdrant_client import QdrantClient

from petfood.recommend.answer import (VET_NOTE, PetProfile, answer, build_prompt, check_answer,
                                      gather_context, needs_vet_note)
from petfood.retrieval.chunking import chunk_paper, chunk_product
from petfood.retrieval.embeddings import HashDense, HashSparse, OverlapReranker
from petfood.retrieval.pipeline import Candidate, HybridRetriever
from petfood.retrieval.store import ChunkStore
from tests.test_retrieval import PAPERS, PRODUCTS

Q = "wet food for a senior cat with kidney disease"


@pytest.fixture(scope="module")
def retriever():
    s = ChunkStore(QdrantClient(":memory:"), "answer_test", HashDense(dim=256), HashSparse())
    s.ensure_collection()
    s.upsert([c for p, secs in PAPERS for c in chunk_paper(p, secs, min_words=5)]
             + [chunk_product(p) for p in PRODUCTS])
    return HybridRetriever(s, OverlapReranker())


class FakeLLM:
    """Returns a canned reply and records the prompt it was given."""

    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def __call__(self, system, user):
        self.calls.append((system, user))
        return self.reply


def _prod(name, tags, species=("cat",), doc_id=1, has_ingredients=True):
    return Candidate(id=name, payload={"doc_type": "product", "doc_id": doc_id, "name": name,
                                       "brand": "B", "species": list(species), "allergen_tags": tags,
                                       "has_ingredients": has_ingredients})


def _paper(source_id):
    return Candidate(id=source_id, payload={"doc_type": "paper", "source_id": source_id, "title": "T",
                                            "heading": "Results", "body": "Cats did well."})


# ───────────── retrieval + prompt ─────────────
def test_context_respects_hard_filters(retriever):
    pet = PetProfile(species="cat", exclude_allergens=("chicken",))
    products, evidence = gather_context(retriever, Q, pet)
    names = [c.payload["name"] for c in products]
    assert names and not any("Chicken" in n for n in names)          # allergen filtered before the LLM
    assert not any("Dog" in n for n in names)                          # species filtered
    assert evidence and all(c.payload["doc_type"] == "paper" for c in evidence)
    assert all("cat" in c.payload["species"] for c in evidence)


def test_prompt_labels_products_and_evidence(retriever):
    pet = PetProfile(species="cat", life_stage="senior", exclude_allergens=("chicken",))
    products, evidence = gather_context(retriever, Q, pet)
    _system, user = build_prompt(Q, pet, products, evidence)
    assert "[P1]" in user and "[E1]" in user
    assert f"[P{len(products) + 1}]" not in user
    assert "must avoid: chicken" in user and "life stage: senior" in user
    assert "study period 2015–2016" in user                           # PMC1's stated years reach the model


def test_prompt_says_when_nothing_matched():
    _s, user = build_prompt("q", PetProfile(), [], [])
    assert "(none matched" in user and "(no relevant research found)" in user


# ───────────── checks ─────────────
def test_check_keeps_valid_answer():
    products, evidence = [_prod("Salmon Wet", ["fish"])], [_paper("PMC1"), _paper("PMC3")]
    raw = {"summary": "Renal diets help [E1].",
           "recommendations": [{"product": "P1", "reason": "Wet and moderate protein [E2].", "evidence": ["E1"]}]}
    summary, recs, cited, warnings = check_answer(raw, PetProfile(), products, evidence)
    assert summary == "Renal diets help [E1]."
    assert [r.name for r in recs] == ["Salmon Wet"] and recs[0].evidence == ["E1", "E2"]
    assert cited == {"E1", "E2"} and warnings == []


def test_check_drops_invented_products_and_citations():
    products, evidence = [_prod("Salmon Wet", ["fish"])], [_paper("PMC1")]
    raw = {"summary": "Studies show benefits [E1] [E7].",
           "recommendations": [{"product": "P9", "reason": "x"},
                               {"product": "P1", "reason": "Good [E4].", "evidence": ["E1", "E5"]}]}
    summary, recs, cited, warnings = check_answer(raw, PetProfile(), products, evidence)
    assert summary == "Studies show benefits [E1]."
    assert [r.label for r in recs] == ["P1"] and recs[0].reason == "Good." and recs[0].evidence == ["E1"]
    assert cited == {"E1"}
    joined = " ".join(warnings)
    assert "unknown product 'P9'" in joined and "[E7]" in joined and "[E4]" in joined and "1 unknown evidence" in joined


def test_check_rechecks_allergens_and_species():
    products = [_prod("Chicken Pate", ["chicken"], doc_id=1), _prod("Dog Stew", [], species=("dog",), doc_id=2),
                _prod("Duck Wet", ["duck"], doc_id=3)]
    raw = {"summary": "", "recommendations": [{"product": f"P{i}", "reason": "r"} for i in (1, 2, 3)]}
    _s, recs, _c, warnings = check_answer(raw, PetProfile(exclude_allergens=("Chicken",)), products, [])
    assert [r.name for r in recs] == ["Duck Wet"]
    assert any("contains chicken" in w for w in warnings) and any("not labelled for cats" in w for w in warnings)


def test_check_caps_recommendations_and_tolerates_junk():
    products = [_prod(f"Food {i}", [], doc_id=i) for i in range(1, 6)]
    raw = {"summary": None, "recommendations": ["P1", {"product": "[P1]", "reason": "a"}, {"product": "P1"},
                                                *({"product": f"P{i}", "reason": "b"} for i in range(2, 6))]}
    _s, recs, _c, warnings = check_answer(raw, PetProfile(), products, [])
    assert [r.label for r in recs] == ["P1", "P2", "P3"]               # "[P1]" accepted, duplicate ignored
    assert any("malformed" in w for w in warnings) and sum("more than 3" in w for w in warnings) == 2
    assert check_answer({"recommendations": "P1"}, PetProfile(), products, [])[3]   # not a list -> warning


def test_check_drops_products_without_ingredients_when_allergens_matter():
    products = [_prod("Mystery Pate", [], has_ingredients=False)]
    raw = {"summary": "", "recommendations": [{"product": "P1", "reason": "r"}]}
    assert check_answer(raw, PetProfile(exclude_allergens=("chicken",)), products, [])[1] == []
    assert len(check_answer(raw, PetProfile(), products, [])[1]) == 1   # no allergy: nothing to verify


def test_check_replaces_product_labels_with_names():
    products = [_prod("Salmon Wet", ["fish"], doc_id=1), _prod("Duck Wet", ["duck"], doc_id=2)]
    raw = {"summary": "P1 and [P2] are wet foods; P9 is unknown.",
           "recommendations": [{"product": "P2", "reason": "Unlike P1, it has duck."}]}
    summary, recs, _c, warnings = check_answer(raw, PetProfile(), products, [])
    assert summary == "Salmon Wet and Duck Wet are wet foods; P9 is unknown."
    assert recs[0].reason == "Unlike Salmon Wet, it has duck." and warnings == []


def test_search_drops_products_without_ingredients_when_excluding_allergens():
    s = ChunkStore(QdrantClient(":memory:"), "no_ingredients", HashDense(dim=64), HashSparse())
    s.ensure_collection()
    unknown = {"id": 20, "source": "curated", "name": "Mystery Wet Cat Food", "species": "cat"}
    paper, secs = PAPERS[2]
    s.upsert([chunk_product(p) for p in (*PRODUCTS, unknown)] + chunk_paper(paper, secs, min_words=5))
    from petfood.retrieval.store import SearchFilters
    names = lambda f: {h.payload.get("name") or h.payload["title"]  # noqa: E731
                       for h in s.search_dense("wet cat food", 20, f)}
    assert "Mystery Wet Cat Food" in names(SearchFilters(doc_type="product"))
    excl = names(SearchFilters(doc_type="product", exclude_allergens=("chicken",)))
    assert "Mystery Wet Cat Food" not in excl and "Kidney Support Duck Wet Cat Food" in excl
    assert paper["title"] in names(SearchFilters(exclude_allergens=("chicken",)))   # papers unaffected


@pytest.mark.parametrize("question,conditions,expected", [
    ("best wet food for a healthy adult cat", (), False),
    ("diet for a cat with chronic kidney disease", (), True),
    ("my cat is overweight", (), True),
    ("best wet food", ("diabetes",), True),
])
def test_vet_note_rule(question, conditions, expected):
    assert needs_vet_note(question, PetProfile(conditions=conditions)) is expected


# ───────────── whole flow ─────────────
def test_answer_end_to_end(retriever):
    pet = PetProfile(species="cat", exclude_allergens=("chicken",), conditions=("ckd",))
    llm = FakeLLM({"summary": "Phosphorus-restricted diets helped cats with CKD [E1].",
                   "recommendations": [{"product": "P1", "reason": "Kidney support formula.", "evidence": ["E1"]}]})
    a = answer(Q, pet, retriever, llm)
    assert len(llm.calls) == 1
    assert a.recommendations and "Chicken" not in a.recommendations[0].name
    assert a.sources[0].cited and a.sources[0].source_id.startswith("PMC")
    assert a.vet_note == VET_NOTE and a.warnings == []
    assert {"retrieve", "generate", "total"} <= set(a.timings_ms)
    assert a.to_dict()["recommendations"][0]["label"] == "P1"


def test_answer_without_context_skips_llm(retriever):
    llm = FakeLLM({"summary": "should not be called"})
    a = answer("anything", PetProfile(species="cat", food_form="raw"), retriever, llm,
               k_products=5, k_evidence=0)
    assert llm.calls == [] and a.recommendations == [] and "No matching" in a.summary
