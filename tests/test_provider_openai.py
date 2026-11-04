from __future__ import annotations

import json

import httpx
import pytest

from app.providers.openai_compat import OpenAICompatProvider


def _provider(handler, **overrides):
    cfg = {
        "name": "P",
        "kind": "openai",
        "base_url": "https://gateway.test",
        "api_key": "sk-test",
        "admin_key": "sk-admin",
        "auth_scheme": "bearer",
        **overrides,
    }
    return OpenAICompatProvider(cfg, transport=httpx.MockTransport(handler))


async def test_complete_non_stream_normalizes_message_and_usage():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer sk-test"
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        assert body["model"] == "m1"
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": "OK"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        })

    provider = _provider(handler)
    try:
        result = await provider.complete("m1", [{"role": "user", "content": "hi"}], max_tokens=8)
    finally:
        await provider.aclose()

    assert result.status_code == 200
    assert result.message["content"] == "OK"
    assert result.usage["prompt_tokens"] == 3
    assert result.usage["completion_tokens"] == 2
    assert result.error is None
    assert result.latency_ms is not None


async def test_complete_stream_accumulates_content_tools_and_ttft():
    chunks = [
        {"choices": [{"delta": {"content": "O"}}]},
        {"choices": [{"delta": {"content": "K"}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1",
                                                "function": {"name": "read_file", "arguments": "{\"path\":"}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "\"a.txt\"}"}}]}}]},
        {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 4}},
    ]
    payload = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=payload.encode())

    provider = _provider(handler)
    try:
        result = await provider.complete("m1", [{"role": "user", "content": "hi"}], stream=True)
    finally:
        await provider.aclose()

    assert result.message["content"] == "OK"
    assert result.message["tool_calls"][0]["function"]["name"] == "read_file"
    assert json.loads(result.message["tool_calls"][0]["function"]["arguments"]) == {"path": "a.txt"}
    assert result.usage["completion_tokens"] == 4
    assert result.ttft_ms is not None
    assert result.chunks == 4
    assert result.done is True


async def test_list_models_prefers_admin_metadata():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer sk-admin"
        if request.url.path == "/v2/model/info":
            return httpx.Response(200, json={
                "data": [{
                    "model_name": "mock/good",
                    "litellm_params": {"input_cost_per_token": 1e-6, "output_cost_per_token": 3e-6},
                    "model_info": {
                        "mode": "chat",
                        "supports_function_calling": True,
                        "supports_response_schema": True,
                        "max_input_tokens": 128000,
                    },
                }],
                "total_pages": 1,
            })
        return httpx.Response(404)

    provider = _provider(handler)
    try:
        models = await provider.list_models()
    finally:
        await provider.aclose()

    assert len(models) == 1
    model = models[0]
    assert model.name == "mock/good"
    assert model.mode == "chat"
    assert model.supports_tools is True
    assert model.supports_json is True
    assert model.context_window == 128000
    assert model.input_cost_per_token == 1e-6


async def test_error_response_keeps_status_and_text():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "rate limited"}})

    provider = _provider(handler)
    try:
        result = await provider.complete("m1", [{"role": "user", "content": "hi"}])
    finally:
        await provider.aclose()

    assert result.status_code == 429
    assert "rate limited" in result.text