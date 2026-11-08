"""Ollama client tests: HTTP mocked with respx (no server needed)."""
import json

import httpx
import pytest
import respx

from petfood.config import Settings
from petfood.llm.ollama import OllamaClient

URL = "http://localhost:11434/api/chat"
SETTINGS = Settings(ollama_base_url="http://localhost:11434", local_model="qwen3:4b", local_think=False,
                    local_num_ctx=8192)


def _reply(content: str) -> httpx.Response:
    return httpx.Response(200, json={"message": {"role": "assistant", "content": content},
                                     "eval_count": 12, "total_duration": 1_500_000_000})


@respx.mock
def test_chat_json_sends_schema_no_think_and_temperature_zero():
    route = respx.post(URL).mock(return_value=_reply('{"answer": "hi"}'))
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    assert OllamaClient(SETTINGS).chat_json("sys", "user", schema=schema) == {"answer": "hi"}
    body = json.loads(route.calls.last.request.read())
    assert body["model"] == "qwen3:4b" and body["stream"] is False and body["think"] is False
    assert body["format"] == schema and body["options"]["temperature"] == 0
    assert body["options"]["num_ctx"] == 8192
    assert [m["role"] for m in body["messages"]] == ["system", "user"]


@respx.mock
def test_chat_json_without_schema_uses_plain_json_mode():
    route = respx.post(URL).mock(return_value=_reply('{"a": 1}'))
    OllamaClient(SETTINGS).chat_json("s", "u")
    assert json.loads(route.calls.last.request.read())["format"] == "json"


@respx.mock
@pytest.mark.parametrize("content", ["not json", "[1, 2]"])
def test_chat_json_rejects_non_object_replies(content):
    respx.post(URL).mock(return_value=_reply(content))
    with pytest.raises(ValueError, match="qwen3:4b"):
        OllamaClient(SETTINGS).chat_json("s", "u")


@respx.mock
def test_missing_model_gives_pull_hint():
    respx.post(URL).mock(return_value=httpx.Response(404, json={"error": "model not found"}))
    with pytest.raises(RuntimeError, match="ollama pull qwen3:4b"):
        OllamaClient(SETTINGS).chat_json("s", "u")


@respx.mock
def test_server_down_gives_start_hint():
    respx.post(URL).mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(RuntimeError, match="ollama serve"):
        OllamaClient(SETTINGS).chat_json("s", "u")


@respx.mock
def test_server_errors_are_retried():
    route = respx.post(URL).mock(side_effect=[httpx.Response(500, text="boom"), _reply('{"ok": true}')])
    assert OllamaClient(SETTINGS).chat_json("s", "u") == {"ok": True}
    assert route.call_count == 2
