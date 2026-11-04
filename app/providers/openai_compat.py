"""OpenAI-compatible adapter (LiteLLM, OpenRouter, vLLM, Ollama, ...)."""
from __future__ import annotations

import json
import time
from typing import Any, Optional

import httpx

from .base import BaseProvider, ChatResult, ModelInfo

ADMIN_LIST_PATHS = ("/v2/model/info", "/model/info")
PAGE_SIZE = 100
MAX_PAGES = 30


def _first(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _param_fallback(error: str) -> bool:
    lowered = (error or "").lower()
    return any(key in lowered for key in ("max_tokens", "max_completion_tokens", "stream_options"))


class OpenAICompatProvider(BaseProvider):
    kind = "openai"

    @property
    def chat_path(self) -> str:
        return self.cfg.get("chat_path") or "/v1/chat/completions"

    @property
    def models_path(self) -> str:
        return self.cfg.get("models_path") or "/v1/models"

    @property
    def admin_paths(self) -> list[str]:
        configured = self.cfg.get("admin_models_paths")
        if isinstance(configured, list) and configured:
            return [str(item) for item in configured]
        if isinstance(configured, str) and configured.strip():
            try:
                parsed = json.loads(configured)
                if isinstance(parsed, list) and parsed:
                    return [str(item) for item in parsed]
            except json.JSONDecodeError:
                pass
        return list(ADMIN_LIST_PATHS)

    def build_payload(
        self,
        model: str,
        messages: list,
        *,
        tools: Optional[list] = None,
        tool_choice: Any = None,
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
            payload["stream_options"] = stream_options or {"include_usage": True}
        return payload

    async def complete(
        self,
        model: str,
        messages: list,
        *,
        tools: Optional[list] = None,
        tool_choice: Any = None,
        response_format: Optional[dict] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
        timeout_s: Optional[float] = None,
    ) -> ChatResult:
        payload = self.build_payload(
            model, messages, tools=tools, tool_choice=tool_choice,
            response_format=response_format, max_tokens=max_tokens,
            temperature=temperature, stream=stream,
        )
        started = time.perf_counter()
        result: Optional[ChatResult] = None
        for attempt in range(2):
            if stream:
                result = await self._stream(payload, timeout_s, started)
            else:
                result = await self._post(payload, timeout_s)
            if result.status_code == 400 and attempt == 0 and _param_fallback(result.text or ""):
                removed = False
                for key in ("max_tokens", "stream_options"):
                    if key in payload:
                        payload.pop(key)
                        removed = True
                if removed:
                    continue
            break
        assert result is not None
        result.latency_ms = int((time.perf_counter() - started) * 1000)
        return result

    async def _post(self, payload: dict, timeout_s: Optional[float]) -> ChatResult:
        try:
            response = await self.client.post(
                self.chat_path,
                json=payload,
                headers=self.headers(),
                params=self.auth_params(),
                timeout=self._timeout(timeout_s),
            )
        except httpx.HTTPError as exc:
            return ChatResult(error=f"{type(exc).__name__}: {exc}")
        if response.status_code != 200:
            return ChatResult(status_code=response.status_code, text=self.error_text(response))
        try:
            body = response.json()
            message = body["choices"][0]["message"] or {}
        except (KeyError, IndexError, TypeError, json.JSONDecodeError):
            return ChatResult(status_code=response.status_code, text="respons tidak memiliki choices")
        return ChatResult(
            status_code=response.status_code,
            message=message,
            usage=body.get("usage") or {},
            done=True,
            raw=body,
        )

    async def _stream(self, payload: dict, timeout_s: Optional[float],
                      started: Optional[float] = None) -> ChatResult:
        out = ChatResult()
        started = started if started is not None else time.perf_counter()
        try:
            async with self.client.stream(
                "POST",
                self.chat_path,
                json=payload,
                headers=self.headers(),
                params=self.auth_params(),
                timeout=self._timeout(timeout_s),
            ) as response:
                out.status_code = response.status_code
                if response.status_code != 200:
                    raw = await response.aread()
                    out.text = self._error_text_from_bytes(raw)
                    return out
                content_parts: list[str] = []
                reasoning_parts: list[str] = []
                calls: dict[int, dict] = {}
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        out.done = True
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("usage"):
                        out.usage = obj["usage"]
                    try:
                        delta = obj["choices"][0].get("delta") or {}
                    except (KeyError, IndexError, TypeError):
                        continue
                    if delta.get("content"):
                        content_parts.append(delta["content"])
                    if delta.get("reasoning_content"):
                        reasoning_parts.append(delta["reasoning_content"])
                    for tc in delta.get("tool_calls") or []:
                        index = tc.get("index", 0)
                        slot = calls.setdefault(index, {"id": "", "type": "function",
                                                        "function": {"name": "", "arguments": ""}})
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            slot["function"]["name"] += fn["name"]
                        if fn.get("arguments"):
                            slot["function"]["arguments"] += fn["arguments"]
                    if delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls"):
                        out.chunks += 1
                        if out.ttft_ms is None:
                            out.ttft_ms = max(1, int((time.perf_counter() - started) * 1000))
        except httpx.HTTPError as exc:
            out.error = f"{type(exc).__name__}: {exc}"
            return out

        out.message = {"content": "".join(content_parts)}
        if reasoning_parts:
            out.message["reasoning_content"] = "".join(reasoning_parts)
        if calls:
            out.message["tool_calls"] = [
                {
                    "id": calls[index]["id"] or f"call_{index}",
                    "type": "function",
                    "function": {
                        "name": calls[index]["function"]["name"],
                        "arguments": calls[index]["function"]["arguments"],
                    },
                }
                for index in sorted(calls)
            ]
        return out

    @staticmethod
    def _error_text_from_bytes(raw: bytes) -> str:
        text = raw.decode("utf-8", "replace")
        try:
            body = json.loads(text)
        except json.JSONDecodeError:
            return text[:500]
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict):
                return str(err.get("message") or err)[:500]
            if isinstance(err, str):
                return err[:500]
        return text[:500]

    # --- discovery ---------------------------------------------------------

    async def list_models(self) -> list[ModelInfo]:
        if self.cfg.get("admin_key"):
            for path in self.admin_paths:
                try:
                    models = await self._fetch_admin(path)
                    if models:
                        return models
                except (PermissionError, httpx.HTTPStatusError):
                    continue
        return await self._fetch_public()

    async def _fetch_admin(self, path: str) -> list[ModelInfo]:
        models: list[ModelInfo] = []
        page = 1
        while page <= MAX_PAGES:
            params = {"page": page, "size": PAGE_SIZE, **self.auth_params()}
            response = await self.client.get(
                path, params=params, headers=self.headers(admin=True),
                timeout=self._timeout(),
            )
            if response.status_code in (401, 403, 404):
                raise PermissionError(f"{path}: HTTP {response.status_code}")
            response.raise_for_status()
            body = response.json()
            if isinstance(body, list):
                data, total_pages = body, None
            else:
                data = body.get("data") or []
                total_pages = body.get("total_pages")
            for item in data:
                info = self.normalize_admin_model(item)
                if info.name:
                    models.append(info)
            if not data or len(data) < PAGE_SIZE:
                break
            if total_pages and page >= int(total_pages):
                break
            page += 1
        return _dedupe(models)

    async def _fetch_public(self) -> list[ModelInfo]:
        response = await self.client.get(
            self.models_path, headers=self.headers(), params=self.auth_params(),
            timeout=self._timeout(),
        )
        response.raise_for_status()
        body = response.json()
        data = body.get("data") if isinstance(body, dict) else body
        models: list[ModelInfo] = []
        for item in data or []:
            name = item.get("id") if isinstance(item, dict) else str(item)
            if not name:
                continue
            models.append(ModelInfo(name=str(name), meta=item if isinstance(item, dict) else {}))
        return _dedupe(models)

    @staticmethod
    def normalize_admin_model(item: dict) -> ModelInfo:
        info = item.get("model_info") or {}
        params = item.get("litellm_params") or {}
        supported = info.get("supported_openai_params") or []
        supports_json = info.get("supports_response_schema")
        if supports_json is None and supported:
            supports_json = "response_format" in supported
        return ModelInfo(
            name=str(item.get("model_name") or ""),
            display_name=item.get("model_name"),
            mode=info.get("mode"),
            blocked=bool(info.get("blocked", False)),
            supports_tools=info.get("supports_function_calling"),
            supports_json=supports_json,
            input_cost_per_token=_first(params.get("input_cost_per_token"),
                                        info.get("input_cost_per_token")),
            output_cost_per_token=_first(params.get("output_cost_per_token"),
                                         info.get("output_cost_per_token")),
            context_window=_first(info.get("max_input_tokens"), info.get("max_tokens"),
                                  info.get("context_window")),
            meta=item,
        )


def _dedupe(models: list[ModelInfo]) -> list[ModelInfo]:
    seen: dict[str, ModelInfo] = {}
    for model in models:
        if model.name not in seen:
            seen[model.name] = model
            continue
        existing = seen[model.name]
        for field in ("mode", "supports_tools", "supports_json", "input_cost_per_token",
                      "output_cost_per_token", "context_window"):
            if getattr(existing, field) is None and getattr(model, field) is not None:
                setattr(existing, field, getattr(model, field))
        existing.blocked = existing.blocked or model.blocked
    return sorted(seen.values(), key=lambda model: model.name)