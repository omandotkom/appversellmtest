"""Provider abstraction: normalized model info and chat results.

Every adapter turns its native protocol into :class:`ModelInfo` and
:class:`ChatResult`, so runners never need to know which provider they hit.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

CONNECT_TIMEOUT = 10.0


@dataclass
class ModelInfo:
    name: str
    display_name: Optional[str] = None
    mode: Optional[str] = None
    blocked: bool = False
    supports_tools: Optional[bool] = None
    supports_json: Optional[bool] = None
    input_cost_per_token: Optional[float] = None
    output_cost_per_token: Optional[float] = None
    context_window: Optional[int] = None
    meta: dict = field(default_factory=dict)


@dataclass
class ChatResult:
    status_code: Optional[int] = None
    message: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)
    ttft_ms: Optional[int] = None
    chunks: int = 0
    done: bool = False
    error: Optional[str] = None
    text: Optional[str] = None
    raw: Optional[dict] = None
    latency_ms: Optional[int] = None


class ProviderError(RuntimeError):
    pass


def _json_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _json_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


def normalize_base_url(url: str) -> str:
    """Normalize a gateway base URL, removing a trailing ``/v1``.

    Paths are configured with their version prefix (``/v1/chat/completions``),
    so a base URL that already ends in ``/v1`` would otherwise double it.
    """
    cleaned = (url or "").strip().rstrip("/")
    if cleaned.lower().endswith("/v1"):
        cleaned = cleaned[:-3]
    return cleaned


class BaseProvider:
    kind = "base"

    def __init__(self, cfg: dict, transport: Optional[httpx.BaseTransport] = None) -> None:
        self.cfg = cfg
        self.base_url = normalize_base_url(cfg.get("base_url") or "")
        self.transport = transport
        self._client: Optional[httpx.AsyncClient] = None

    # --- HTTP plumbing -----------------------------------------------------

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                transport=self.transport,
                follow_redirects=True,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "BaseProvider":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

    def _timeout(self, timeout_s: Optional[float] = None) -> httpx.Timeout:
        total = float(timeout_s or self.cfg.get("timeout_s") or 60)
        return httpx.Timeout(total, connect=CONNECT_TIMEOUT)

    def headers(self, *, admin: bool = False) -> dict[str, str]:
        scheme = (self.cfg.get("auth_scheme") or "bearer").lower()
        key = self.cfg.get("admin_key") if admin and self.cfg.get("admin_key") else self.cfg.get("api_key")
        headers = _json_dict(self.cfg.get("extra_headers"))
        headers = {str(k): str(v) for k, v in headers.items()}
        if key and scheme == "bearer":
            headers["Authorization"] = f"Bearer {key}"
        elif key and scheme == "x-api-key":
            headers["x-api-key"] = str(key)
        return headers

    def auth_params(self) -> dict[str, str]:
        if (self.cfg.get("auth_scheme") or "").lower() == "query" and self.cfg.get("api_key"):
            return {"api_key": str(self.cfg["api_key"])}
        return {}

    # --- contract ----------------------------------------------------------

    async def list_models(self) -> list[ModelInfo]:
        raise NotImplementedError

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
        raise NotImplementedError

    @staticmethod
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