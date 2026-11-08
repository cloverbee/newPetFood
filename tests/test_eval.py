"""Eval scoring: pure functions, no models, no database."""
import pytest

from petfood.eval.run import Question, load_questions, product_problems, score


def _answer(recs=(), sources=(), vet=None, warnings=(), **profile):
    return {"profile": {"species": "cat", "life_stage": None, "food_form": None,
                        "exclude_allergens": (), "conditions": ()} | profile,
            "recommendations": [{"product_id": i, "name": f"P{i}"} for i in recs],
            "sources": list(sources), "vet_note": vet, "warnings": list(warnings),
            "timings_ms": {"total": 1000}}


def test_question_file_loads_and_ids_are_unique():
    qs = load_questions()
    assert len(qs) >= 20
    assert all(q.id and q.question for q in qs)


@pytest.mark.parametrize("product, profile, problem", [
    ({"food_form": "dry", "life_stage": "adult"}, {"food_form": "wet"}, "form dry ≠ wet"),
    ({"food_form": "wet", "life_stage": None}, {"life_stage": "kitten"}, "life stage unstated ≠ kitten"),
    ({"food_form": "wet", "life_stage": "kitten"}, {"life_stage": "adult"}, "life stage kitten ≠ adult"),
    ({"allergen_tags": ["chicken"], "ingredients": ["chicken"]}, {"exclude_allergens": ["chicken"]},
     "contains chicken"),
    ({"allergen_tags": [], "ingredients": None}, {"exclude_allergens": ["fish"]},
     "no ingredient list to check allergens"),
])
def test_product_problems(product, profile, problem):
    assert problem in product_problems(product, profile)


@pytest.mark.parametrize("product, profile", [
    ({"food_form": "wet", "life_stage": None}, {"food_form": "wet", "life_stage": "adult"}),
    ({"food_form": "wet", "life_stage": None}, {"life_stage": "senior"}),
    ({"food_form": "wet", "life_stage": "all"}, {"life_stage": "kitten"}),
    ({"allergen_tags": ["fish"], "ingredients": ["tuna"]}, {"exclude_allergens": ["chicken"]}),
])
def test_product_fits(product, profile):
    assert product_problems(product, profile) == []


def test_score_all_pass():
    q = Question("x", "wet food for adult cat?", expect={
        "life_stage": "adult", "food_form": "wet", "min_recs": 1, "vet_note": False,
        "papers": ["overweight"]})
    a = _answer(recs=[1], life_stage="adult", food_form="wet",
                sources=[{"title": "Overweight in adult cats", "cited": True}])
    checks, notes = score(q, a, {1: {"food_form": "wet", "life_stage": None}})
    assert all(v is True for v in checks.values()) and notes == []


def test_score_failures_explain_themselves():
    q = Question("x", "?", expect={"food_form": "wet", "min_recs": 2, "vet_note": True,
                                   "papers": ["raw diets"]})
    a = _answer(recs=[1], food_form="wet", warnings=["dropped a sentence"],
                sources=[{"title": "Raw diets for dogs and cats", "cited": False}])
    checks, notes = score(q, a, {1: {"food_form": "dry"}})
    assert checks == {"filters": True, "recs": False, "recs_fit": False, "vet_note": False,
                      "paper_found": True, "paper_cited": False, "no_removals": False}
    assert any("1 recommendations < 2" in n for n in notes)
    assert any("form dry ≠ wet" in n for n in notes)
    assert "vet note missing" in notes


def test_none_means_no_filter_and_missing_expectations_are_skipped():
    q = Question("x", "?", expect={"food_form": "none"})
    checks, _ = score(q, _answer(), {})
    assert checks["filters"] is True
    assert checks["recs"] is None and checks["recs_fit"] is None and checks["paper_found"] is None
