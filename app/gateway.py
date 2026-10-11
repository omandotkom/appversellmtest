"""Thin HTTP client for the gateway (OpenAI-compatible API + LiteLLM admin endpoints)."""
from __future__ import annotations

import json
import time
from typing import Any, Optional

import httpx

from . import config

CONNECT_TIMEOUT = 10.0
ADMIN_LIST_ENDPOINTS = ("/v2/model/info", "/model/info")
PAGE_SIZE = 100
MAX_PAGES = 30


class GatewayNotConfigured(RuntimeError):
    pass


def configured_url() -> str:
    """The gateway URL as configured (scheme-normalized, trailing slash stripped)."""
    url = (config.get("GATEWAY_URL") or "http://localhost:4000").strip()
    if not url:
        url = "http://localhost:4000"
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url.rstrip("/")


def base_url() -> str:
    """Transport root for requests.

    OpenAI-compatible clients point at a base URL that may already include the
    ``/v1`` version prefix (e.g. ``https://gateway.example/v1``). The app always
    issues ``/v1/...`` paths, so strip a trailing ``/v1`` here to avoid a doubled
    ``/v1/v1`` prefix.
    """
    url = configured_url()
    if url.endswith("/v1"):
        url = url[:-3]
    return url


def timeout(total_s: float) -> httpx.Timeout:
    return httpx.Timeout(total_s, connect=CONNECT_TIMEOUT)


def admin_key() -> Optional[str]:
    return config.get("GATEWAY_ADMIN_KEY") or None


def test_key() -> Optional[str]:
    return config.get("GATEWAY_TEST_KEY") or admin_key()


def test_headers() -> dict[str, str]:
    key = test_key()
    if not key:
        raise GatewayNotConfigured("GATEWAY_TEST_KEY belum diisi. Buka tab Settings dan isi virtual key.")
    return {"Authorization": f"Bearer {key}"}


def error_text(response: httpx.Response) -> str:
    try:
        body = response.json()
    except Exception:
        return (response.text or "")[:500]
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            return str(err.get("message") or err)[:500]
        if isinstance(err, str):
            return err[:500]
    return str(body)[:500]


def chat_payload(
    *,
    model: str,
    messages: list,
    tools: Optional[list] = None,
    tool_choice: Optional[Any] = None,
    response_format: Optional[dict] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    stream: bool = False,
    stream_options: Optional[dict] = None,
) -> dict:
    payload: dict[str, Any] = {"model": model, "messages": messages}
    if tools:
        payload["tools"] = tools
    if tool_choice:
        payload["tool_choice"] = tool_choice
    if response_format:
        payload["response_format"] = response_format
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if temperature is not None:
        payload["temperature"] = temperature
    if stream:
        payload["stream"] = True
    if stream_options:
        payload["stream_options"] = stream_options
    return payload


async def stream_chat(client: httpx.AsyncClient, payload: dict, timeout_s: float) -> dict:
    """Stream one chat completion; returns raw observations.

    Result keys: status_code, text, ttft_ms, chunks, done, usage, error.
    """
    started = time.perf_counter()
    out: dict[str, Any] = {
        "status_code": None, "text": None, "ttft_ms": None,
        "chunks": 0, "done": False, "usage": {}, "error": None,
    }
    try:
        async with client.stream("POST", "/v1/chat/completions", json=payload,
                                 headers=test_headers(), timeout=timeout(timeout_s)) as response:
            out["status_code"] = response.status_code
            if response.status_code != 200:
                raw = await response.aread()
                out["text"] = raw.decode("utf-8", "replace")[:500]
                return out
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    out["done"] = True
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    out["usage"] = obj["usage"]
                try:
                    delta = obj["choices"][0].get("delta") or {}
                except (KeyError, IndexError, TypeError):
                    continue
                if delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls"):
                    out["chunks"] += 1
                    if out["ttft_ms"] is None:
                        out["ttft_ms"] = int((time.perf_counter() - started) * 1000)
    except httpx.HTTPError as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def _first(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def normalize_admin_model(item: dict) -> dict:
    """Flatten a LiteLLM /model/info entry into the shape the UI consumes."""
    info = item.get("model_info") or {}
    params = item.get("litellm_params") or {}
    supported = info.get("supported_openai_params") or []
    supports_json = info.get("supports_response_schema")
    if supports_json is None and supported:
        supports_json = "response_format" in supported
    return {
        "name": item.get("model_name"),
        "mode": info.get("mode"),
        "blocked": bool(info.get("blocked", False)),
        "provider": info.get("litellm_provider") or params.get("custom_llm_provider"),
        "input_cost_per_token": _first(params.get("input_cost_per_token"), info.get("input_cost_per_token")),
        "output_cost_per_token": _first(params.get("output_cost_per_token"), info.get("output_cost_per_token")),
        "supports_tools": info.get("supports_function_calling"),
        "supports_json": supports_json,
    }


def is_chat_candidate(model: dict) -> bool:
    return model.get("mode") in (None, "chat")


async def _fetch_admin_models(client: httpx.AsyncClient, key: str) -> list[dict]:
    last_error: Optional[Exception] = None
    headers = {"Authorization": f"Bearer {key}"}
    for endpoint in ADMIN_LIST_ENDPOINTS:
        try:
            models: list[dict] = []
            page = 1
            while page <= MAX_PAGES:
                response = await client.get(endpoint, params={"page": page, "size": PAGE_SIZE}, headers=headers)
                if response.status_code in (401, 403, 404):
                    raise PermissionError(f"{endpoint}: HTTP {response.status_code}")
                response.raise_for_status()
                body = response.json()
                if isinstance(body, list):
                    data, total_pages = body, None
                else:
                    data = body.get("data") or []
                    total_pages = body.get("total_pages")
                for item in data:
                    normalized = normalize_admin_model(item)
                    if normalized.get("name"):
                        models.append(normalized)
                if not data or len(data) < PAGE_SIZE:
                    break
                if total_pages and page >= int(total_pages):
                    break
                page += 1
            return models
        except PermissionError as exc:
            last_error = exc
            continue
    if last_error:
        raise last_error
    return []


def _dedupe(models: list[dict]) -> list[dict]:
    seen: dict[str, dict] = {}
    for model in models:
        name = model["name"]
        if name not in seen:
            seen[name] = model
            continue
        existing = seen[name]
        for field in ("mode", "provider", "input_cost_per_token", "output_cost_per_token",
                      "supports_tools", "supports_json"):
            if existing.get(field) is None and model.get(field) is not None:
                existing[field] = model[field]
        existing["blocked"] = bool(existing.get("blocked")) or bool(model.get("blocked"))
    return sorted(seen.values(), key=lambda m: m["name"])


async def list_models() -> list[dict]:
    """List deployments. Prefers admin metadata, falls back to /v1/models."""
    async with httpx.AsyncClient(base_url=base_url(), timeout=timeout(30)) as client:
        key = config.get("GATEWAY_ADMIN_KEY")
        if key:
            try:
                models = await _fetch_admin_models(client, key)
                if models:
                    return _dedupe(models)
            except (PermissionError, httpx.HTTPStatusError):
                pass  # fall through to the key-scoped public list
        headers: dict[str, str] = {}
        fallback_key = test_key()
        if fallback_key:
            headers["Authorization"] = f"Bearer {fallback_key}"
        response = await client.get("/v1/models", headers=headers)
        response.raise_for_status()
        body = response.json()
        data = body.get("data") if isinstance(body, dict) else body
        models = []
        for item in data or []:
            name = item.get("id") if isinstance(item, dict) else str(item)
            if not name:
                continue
            models.append({
                "name": name,
                "mode": None,
                "blocked": False,
                "provider": None,
                "input_cost_per_token": None,
                "output_cost_per_token": None,
                "supports_tools": None,
                "supports_json": None,
            })
        return _dedupe(models)
