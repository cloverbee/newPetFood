"""Study-year extraction: regex prefilter, answer validation, DeepSeek client (all offline).

The tricky sentences are taken from real papers in the corpus.
"""
from pathlib import Path

import httpx
import pytest
import respx

from petfood.config import Settings
from petfood.ingest.jats import parse_front_meta
from petfood.ingest.study_years import candidate_sentences, extract_study_years, validate
from petfood.llm.deepseek import DeepSeekClient

FIX = Path(__file__).parent / "fixtures"

REAL_DATES = "Fecal samples were collected from December 2011 to March 2012 from healthy dogs (n = 20)."
CITED_STUDY = ("A total diet study measuring contaminants in ready to eat food was conducted in 2000 "
               "by the French National Institute for Agricultural Research.")
LIT_SEARCH = "A literature search was conducted through 2 March 2020, using PubMed."
CITATION_ONLY = ("Faecal output was collected over four days (Association of American Feed Control "
                 "Officials, 2016) and compared with earlier work (Dubey and Urban, 1990).")
NO_YEAR = "Forty neutered adult cats were enrolled."


def _sections(*texts, stype="methods"):
    return [{"section_type": stype, "text": t} for t in texts]


# ───────────── JATS front matter ─────────────
def test_parse_front_meta_reads_received_year_and_article_type():
    meta = parse_front_meta((FIX / "sample_jats.xml").read_bytes())
    assert meta == {"article_type": "research-article", "received_year": 2020}


def test_parse_front_meta_without_history():
    meta = parse_front_meta(b"<article><front><article-meta/></front></article>")
    assert meta == {"article_type": None, "received_year": None}


# ───────────── regex prefilter ─────────────
def test_candidates_drop_citation_only_and_yearless_sentences():
    secs = _sections(" ".join([NO_YEAR, CITATION_ONLY, REAL_DATES, CITED_STUDY]))
    assert candidate_sentences(secs, upper_bound=2013) == [REAL_DATES, CITED_STUDY]


def test_candidates_drop_citation_fragments_and_measurements():
    secs = _sections("(2013) showed that cross-reactivity varies widely between antibodies. "
                     "(2003). Progesterone was below the 2000 ng/g threshold in one sample.")
    assert candidate_sentences(secs, upper_bound=2017) == []


def test_candidates_drop_years_after_the_upper_bound():
    assert candidate_sentences(_sections(LIT_SEARCH), upper_bound=2019) == []
    assert candidate_sentences(_sections(LIT_SEARCH), upper_bound=2020) == [LIT_SEARCH]


def test_candidates_put_methods_first_and_use_caption_line_only():
    secs = (_sections("Cats were followed from 2014 to 2016.", stype="results")
            + _sections("Cats were enrolled in 2014.")
            + _sections("Table 1 Cats recruited 2014\n| 2015 | 2016 |", stype="table"))
    assert candidate_sentences(secs, 2018) == [
        "Cats were enrolled in 2014.", "Cats were followed from 2014 to 2016.", "Table 1 Cats recruited 2014"]


# ───────────── validation ─────────────
CANDS = [REAL_DATES, CITED_STUDY]


def test_validate_accepts_supported_range():
    sy = validate({"start": 2011, "end": 2012, "sentence_ids": [1], "reason": "sample collection"}, CANDS, 2013)
    assert (sy.status, sy.start, sy.end, sy.evidence) == ("found", 2011, 2012, REAL_DATES)


def test_validate_single_year_fills_both_ends():
    sy = validate({"start": None, "end": 2012, "sentence_ids": [1]}, CANDS, 2013)
    assert (sy.status, sy.start, sy.end) == ("found", 2012, 2012)


def test_validate_null_answer_is_not_stated():
    sy = validate({"start": None, "end": None, "sentence_ids": [], "reason": "review"}, CANDS, 2013)
    assert (sy.status, sy.start, sy.end, sy.note) == ("not_stated", None, None, "review")


@pytest.mark.parametrize("answer, why", [
    ({"start": 2010, "end": 2012, "sentence_ids": [1]}, "years not in the cited sentences"),
    ({"start": 2011, "end": 2012, "sentence_ids": [2]}, "years not in the cited sentences"),
    ({"start": 2011, "end": 2012, "sentence_ids": [7]}, "bad sentence_ids"),
    ({"start": 2011, "end": 2012, "sentence_ids": []}, "bad sentence_ids"),
    ({"start": 2012, "end": 2011, "sentence_ids": [1]}, "start after end"),
    ({"start": "soon", "end": 2012, "sentence_ids": [1]}, "non-integer"),
])
def test_validate_rejects_unsupported_answers(answer, why):
    sy = validate(answer, CANDS, 2013)
    assert sy.status == "rejected" and why in sy.note
    assert sy.start is None and sy.end is None      # rejected years are never filterable


def test_validate_expands_two_digit_range_ends():
    cands = ["A questionnaire was mailed in the second year of the study (1997-99).",
             "Follow-up visits were held in the fifth year (2000-02) for all participants."]
    sy = validate({"start": 1997, "end": 2002, "sentence_ids": [1, 2]}, cands, 2009)
    assert (sy.status, sy.start, sy.end) == ("found", 1997, 2002)


def test_validate_rejects_study_ending_after_submission():
    sy = validate({"start": 2011, "end": 2012, "sentence_ids": [1]}, CANDS, upper_bound=2011)
    assert sy.status == "rejected" and "ends after 2011" in sy.note


# ───────────── end to end with a fake LLM ─────────────
def test_extract_skips_llm_when_no_dates():
    def llm(_s, _u):
        raise AssertionError("should not be called")
    sy = extract_study_years({"title": "T", "pub_year": 2020}, _sections(NO_YEAR, CITATION_ONLY), None, llm)
    assert sy.status == "no_dates"


@pytest.mark.parametrize("title, article_type", [
    ("How Nutritionally Sound Is Vegan Pet Food? A Systematic Review", "research-article"),
    ("Probiotics in pet food: a meta-analysis", None),
    ("Fibre in feline diets", "review-article"),
])
def test_extract_skips_reviews_without_calling_llm(title, article_type):
    def llm(_s, _u):
        raise AssertionError("should not be called")
    paper = {"title": title, "pub_year": 2026}
    sy = extract_study_years(paper, _sections("These databases were searched on 14th June 2025."), article_type, llm)
    assert sy.status == "not_stated" and "review" in sy.note


def test_extract_prompt_and_bound_use_received_year():
    seen = {}

    def llm(system, user):
        seen["user"] = user
        return {"start": 2011, "end": 2012, "sentence_ids": [1]}

    paper = {"title": "Faecal microbiota of dogs and cats", "pub_year": 2013, "received_year": 2012}
    sy = extract_study_years(paper, _sections(REAL_DATES, CITED_STUDY), "research-article", llm)
    assert sy.status == "found" and (sy.start, sy.end) == (2011, 2012)
    assert "Received: 2012" in seen["user"] and "Article type: research-article" in seen["user"]
    assert f"[1] {REAL_DATES}" in seen["user"] and f"[2] {CITED_STUDY}" in seen["user"]


# ───────────── DeepSeek client ─────────────
def test_deepseek_requires_api_key():
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        DeepSeekClient(Settings(deepseek_api_key=""))


@respx.mock
def test_deepseek_chat_json_sends_json_mode_and_parses_reply():
    route = respx.post("https://api.deepseek.com/chat/completions").mock(return_value=httpx.Response(200, json={
        "choices": [{"message": {"content": '{"start": 2011, "end": 2012}'}}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 9},
    }))
    client = DeepSeekClient(Settings(deepseek_api_key="sk-test", deepseek_model="deepseek-chat"))
    assert client.chat_json("sys", "user") == {"start": 2011, "end": 2012}
    body = route.calls.last.request.read()
    assert b'"json_object"' in body and b'"temperature":0' in body
    assert route.calls.last.request.headers["Authorization"] == "Bearer sk-test"
