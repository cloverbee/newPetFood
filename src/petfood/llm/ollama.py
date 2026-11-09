"""Local LLM client for Ollama (Qwen3 by default). Same chat_json(system, user) shape as
DeepSeekClient, so callers can swap models without other changes (D15).

API docs: https://github.com/ollama/ollama/blob/main/docs/api.md (POST /api/chat)

Two settings matter for small local models:
- think=False: Qwen3 otherwise writes a long reasoning trace first (~4x slower on a laptop).
- format=<JSON schema>: Ollama constrains decoding to the schema. With thinking off, plain
  "json" mode gave valid JSON with invented keys; a schema fixes the shape.
- num_ctx=8192 (LOCAL_NUM_CTX): Ollama's default 4096-token window was nearly full (3,789
  prompt tokens); an overflow silently drops the start of the prompt.
"""
from __future__ import annotations

import json
import logging

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from petfood.config import Settings, get_settings

log = logging.getLogger(__name__)


class _Retryable(Exception):
    pass


class OllamaClient:
    def __init__(self, settings: Settings | None = None, client: httpx.Client | None = None,
                 model: str | None = None):
        s = settings or get_settings()
        self.model = model or s.local_model
        self.think = s.local_think
        self.num_ctx = s.local_num_ctx
        self.url = f"{s.ollama_base_url.rstrip('/')}/api/chat"
        # first call loads the model into memory (a few seconds); generation on a laptop is slow
        self.client = client or httpx.Client(timeout=300)
        self.last_stats: dict = {}   # Ollama's own timings for the latest call (latency breakdown)

    @retry(retry=retry_if_exception_type((_Retryable, httpx.TransportError)),
           wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3), reraise=True)
    def _post(self, body: dict) -> dict:
        try:
            r = self.client.post(self.url, json=body)
        except httpx.ConnectError as e:
            raise RuntimeError(f"Ollama is not running at {self.url} (start it with `ollama serve`)") from e
        if r.status_code == 404:
            raise RuntimeError(f"Ollama model {self.model!r} not found (run `ollama pull {self.model}`)")
        if r.status_code >= 500:
            raise _Retryable(f"{r.status_code} from Ollama: {r.text[:200]}")
        r.raise_for_status()
        return r.json()

    def chat_json(self, system: str, user: str, schema: dict | None = None) -> dict:
        """One deterministic call; returns the parsed JSON object. `schema` (a JSON schema)
        makes Ollama enforce the exact shape; without it, any JSON object is allowed."""
        data = self._post({
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "think": self.think,
            "format": schema or "json",
            "options": {"temperature": 0, "num_ctx": self.num_ctx},
        })
        content = (data.get("message") or {}).get("content") or ""
        ms = lambda k: round((data.get(k) or 0) / 1e6)   # Ollama reports nanoseconds
        # load = reading the model into memory (0 when already loaded); prompt = reading the
        # prompt (prefill); output = writing the answer token by token (decode)
        self.last_stats = {"tokens_in": data.get("prompt_eval_count"), "tokens_out": data.get("eval_count"),
                           "load_ms": ms("load_duration"), "prompt_ms": ms("prompt_eval_duration"),
                           "output_ms": ms("eval_duration")}
        log.debug("ollama %s tokens in=%s out=%s total=%.1fs", self.model, data.get("prompt_eval_count"),
                  data.get("eval_count"), (data.get("total_duration") or 0) / 1e9)
        try:
            out = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"{self.model} returned invalid JSON: {content[:200]!r}") from e
        if not isinstance(out, dict):
            raise ValueError(f"{self.model} returned JSON that is not an object: {content[:200]!r}")
        return out
