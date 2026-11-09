"""Latency breakdown: where an answer's time goes (D20).

answer() records raw stage times in Answer.timings_ms ("products.dense_embed", "papers.rerank",
"generate", ...). This module groups them into the stages worth reading, formats one answer as
a bar chart, and summarises many answers (eval) as average / median / p90 / slowest per stage.
Counts ("*.rerank_n", "prompt_chars") live in the same dict but aren't times.
"""
from __future__ import annotations

import statistics

# (label, keys summed). Order = order things happen in.
STAGES: list[tuple[str, tuple[str, ...]]] = [
    ("products: embed query", ("products.dense_embed", "products.sparse_embed")),
    ("products: Qdrant search", ("products.dense_qdrant", "products.sparse_qdrant")),
    ("products: rerank", ("products.rerank",)),
    ("papers: embed query", ("papers.dense_embed", "papers.sparse_embed")),
    ("papers: Qdrant search", ("papers.dense_qdrant", "papers.sparse_qdrant")),
    ("papers: rerank", ("papers.rerank",)),
    ("build prompt", ("prompt",)),
    ("LLM writes answer", ("generate",)),
    ("check answer", ("check",)),
]


def stages(t: dict) -> list[tuple[str, float]]:
    """Grouped stage times in ms for one answer, plus "other" = total minus the stages (fusion,
    filtering, Python overhead). Stages with no data (e.g. no LLM call) are left out."""
    out = [(label, sum(t.get(k, 0.0) for k in keys)) for label, keys in STAGES
           if any(k in t for k in keys)]
    total = t.get("total", 0.0)
    other = total - sum(ms for _, ms in out)
    if total and other > 0.5:
        out.append(("other", other))
    return out


def _bar(share: float, width: int = 30) -> str:
    return "█" * round(share * width)


def format_one(t: dict, llm_stats: dict | None = None, startup_ms: float | None = None) -> str:
    """Bar chart for one answer."""
    total = t.get("total", 0.0) or 1.0
    rows = stages(t)
    w = max(len(label) for label, _ in rows)
    lines = ["Latency breakdown:"]
    if startup_ms is not None:
        lines.append(f"  {'start-up (load models)':<{w}}  {startup_ms / 1000:6.2f} s   (not in total; "
                     "paid once per command)")
    for label, ms in rows:
        lines.append(f"  {label:<{w}}  {ms / 1000:6.2f} s  {ms / total:4.0%}  {_bar(ms / total)}")
    lines.append(f"  {'total':<{w}}  {total / 1000:6.2f} s")
    extras = []
    if "products.rerank_n" in t:
        extras.append(f"reranked {t['products.rerank_n']:.0f} product + {t.get('papers.rerank_n', 0):.0f} paper texts")
    if "prompt_chars" in t:
        extras.append(f"prompt {t['prompt_chars']:,.0f} chars")
    if extras:
        lines.append("  " + " · ".join(extras))
    if llm_stats:
        lines.append("  LLM: " + _llm_line(llm_stats))
    return "\n".join(lines)


def _llm_line(s: dict) -> str:
    parts = [f"{s.get('tokens_in') or '?'} tokens in, {s.get('tokens_out') or '?'} out"]
    if s.get("tokens_cached"):
        parts.append(f"{s['tokens_cached']} cached")
    if "attempts" in s:
        parts.append(f"{s['attempts']} attempt{'s' if s['attempts'] != 1 else ''}"
                     + (" (retried: slow network/server)" if s["attempts"] > 1 else ""))
    if "prompt_ms" in s:
        parts.append(f"load {s.get('load_ms', 0) / 1000:.1f} s, read prompt {s['prompt_ms'] / 1000:.1f} s, "
                     f"write {s.get('output_ms', 0) / 1000:.1f} s")
        if s.get("output_ms") and s.get("tokens_out"):
            parts.append(f"{s['tokens_out'] / (s['output_ms'] / 1000):.0f} tokens/s")
    return " · ".join(parts)


def _pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(p * (len(xs) - 1)))]


def format_many(timings: list[dict], llm_stats: list[dict] | None = None) -> str:
    """Per-stage average / median / p90 / slowest over many answers, and each stage's share of
    the average total."""
    if not timings:
        return ""
    labels = [label for label, _ in STAGES] + ["other"]
    per = {label: [] for label in labels}
    for t in timings:
        got = dict(stages(t))
        for label in labels:
            if label in got or label != "other":
                per[label].append(got.get(label, 0.0))
    totals = [t.get("total", 0.0) for t in timings]
    mean_total = statistics.mean(totals) or 1.0
    w = max(len(label) for label in labels)
    lines = [f"Latency breakdown over {len(timings)} answers (seconds):",
             f"  {'stage':<{w}}   avg  median   p90   max  share"]
    for label in labels:
        xs = per[label]
        if not xs or max(xs) == 0:
            continue
        avg = statistics.mean(xs)
        lines.append(f"  {label:<{w}}  {avg / 1000:5.2f}  {statistics.median(xs) / 1000:5.2f}  "
                     f"{_pct(xs, .9) / 1000:5.2f} {max(xs) / 1000:5.2f}  {avg / mean_total:4.0%}  "
                     f"{_bar(avg / mean_total, 20)}")
    lines.append(f"  {'total':<{w}}  {mean_total / 1000:5.2f}  {statistics.median(totals) / 1000:5.2f}  "
                 f"{_pct(totals, .9) / 1000:5.2f} {max(totals) / 1000:5.2f}")
    stats = [s for s in (llm_stats or []) if s]
    if stats:
        tin = [s["tokens_in"] for s in stats if s.get("tokens_in")]
        tout = [s["tokens_out"] for s in stats if s.get("tokens_out")]
        retried = sum(1 for s in stats if s.get("attempts", 1) > 1)
        if tin and tout:
            lines.append(f"  LLM tokens: {statistics.mean(tin):.0f} in, {statistics.mean(tout):.0f} out on average"
                         + (f"; {retried} call(s) retried" if retried else ""))
    return "\n".join(lines)
