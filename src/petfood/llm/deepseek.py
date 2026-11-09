"""DeepSeek chat client (OpenAI-compatible API). Minimal for now; Phase 3 adds the router.

API docs: https://api-docs.deepseek.com
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


class DeepSeekClient:
    def __init__(self, settings: Settings | None = None, client: httpx.Client | None = None):
        s = settings or get_settings()
        if not s.deepseek_api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set (see .env.example)")
        self.model = s.deepseek_model
        self.url = f"{s.deepseek_base_url.rstrip('/')}/chat/completions"
        self.client = client or httpx.Client(
            timeout=60, headers={"Authorization": f"Bearer {s.deepseek_api_key}"})
        self.last_stats: dict = {}   # tokens and attempts of the latest call (latency breakdown)
        self._attempts = 0

    @retry(retry=retry_if_exception_type((_Retryable, httpx.TransportError)),
           wait=wait_exponential(min=1, max=30), stop=stop_after_attempt(5), reraise=True)
    def _post(self, body: dict) -> dict:
        self._attempts += 1
        r = self.client.post(self.url, json=body)
        if r.status_code == 429 or r.status_code >= 500:
            raise _Retryable(f"{r.status_code} from DeepSeek")
        r.raise_for_status()
        return r.json()

    def chat_json(self, system: str, user: str) -> dict:
        """One deterministic call in JSON mode; returns the parsed JSON object."""
        self._attempts = 0
        data = self._post({
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        })
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage") or {}
        log.debug("deepseek tokens in=%s out=%s", usage.get("prompt_tokens"), usage.get("completion_tokens"))
        # attempts > 1 means a retry (timeout, 429 or 5xx) happened: the usual cause of a slow call
        self.last_stats = {"tokens_in": usage.get("prompt_tokens"), "tokens_out": usage.get("completion_tokens"),
                           "tokens_cached": usage.get("prompt_cache_hit_tokens"), "attempts": self._attempts}
        return json.loads(content)
