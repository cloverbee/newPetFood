"""Latency breakdown: grouping raw stage times and DeepSeek retry counting."""
import httpx
import respx

from petfood.config import Settings
from petfood.llm.deepseek import DeepSeekClient
from petfood.recommend.latency import format_many, format_one, stages

T = {"products.dense_embed": 10, "products.sparse_embed": 2, "products.dense_qdrant": 30,
     "products.sparse_qdrant": 8, "products.rerank": 4000, "products.rerank_n": 20,
     "papers.dense_embed": 10, "papers.sparse_embed": 2, "papers.dense_qdrant": 40,
     "papers.sparse_qdrant": 10, "papers.rerank": 15000, "papers.rerank_n": 30,
     "prompt": 1, "prompt_chars": 12000, "generate": 3000, "check": 2, "total": 22200}


def test_stages_group_and_add_other():
    got = dict(stages(T))
    assert got["products: embed query"] == 12 and got["products: Qdrant search"] == 38
    assert got["papers: rerank"] == 15000 and got["LLM writes answer"] == 3000
    assert round(got["other"]) == 22200 - sum(v for k, v in got.items() if k != "other")


def test_counts_are_not_counted_as_time():
    assert "rerank_n" not in " ".join(label for label, _ in stages(T))
    assert sum(ms for _, ms in stages(T)) == T["total"]


def test_no_llm_call_leaves_stage_out():
    t = {k: v for k, v in T.items() if k not in ("prompt", "generate", "check")}
    assert "LLM writes answer" not in dict(stages(t))


def test_format_one_shows_shares_and_llm_line():
    out = format_one(T, {"tokens_in": 3800, "tokens_out": 250, "attempts": 2}, startup_ms=4000)
    assert "papers: rerank" in out and "68%" in out          # 15000 / 22200
    assert "start-up (load models)" in out and "4.00 s" in out
    assert "2 attempts (retried" in out
    assert "reranked 20 product + 30 paper texts" in out


def test_format_many_reports_median_and_retries():
    out = format_many([T, T | {"generate": 60000, "total": 79200}],
                      [{"tokens_in": 3800, "tokens_out": 250, "attempts": 1},
                       {"tokens_in": 3800, "tokens_out": 250, "attempts": 3}])
    assert "over 2 answers" in out and "LLM writes answer" in out
    assert "1 call(s) retried" in out


@respx.mock
def test_deepseek_counts_attempts_and_tokens():
    url = "https://api.deepseek.com/chat/completions"
    ok = httpx.Response(200, json={"choices": [{"message": {"content": '{"a": 1}'}}],
                                   "usage": {"prompt_tokens": 3800, "completion_tokens": 250,
                                             "prompt_cache_hit_tokens": 1024}})
    respx.post(url).mock(side_effect=[httpx.Response(503), ok])
    c = DeepSeekClient(Settings(deepseek_api_key="test"))
    assert c.chat_json("s", "u") == {"a": 1}
    assert c.last_stats == {"tokens_in": 3800, "tokens_out": 250, "tokens_cached": 1024, "attempts": 2}
