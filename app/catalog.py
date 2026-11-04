"""Provider catalog: CRUD, connectivity tests and model discovery."""
from __future__ import annotations

import json
import sqlite3
import time
from typing import Any, Optional

from . import db
from .providers import KINDS, ModelInfo, build_provider

DEFAULT_CHAT_PATH = "/v1/chat/completions"
DEFAULT_MODELS_PATH = "/v1/models"
DEFAULT_ADMIN_PATHS = ["/v2/model/info", "/model/info"]


def _clean_paths(value: Any) -> Any:
    if value in (None, ""):
        return None
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            pass
        return [part.strip() for part in text.split(",") if part.strip()]
    raise ValueError("admin_models_paths harus berupa list atau JSON array")


def _clean_headers(value: Any) -> Any:
    if value in (None, ""):
        return None
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items()}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"extra_headers bukan JSON valid: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("extra_headers harus berupa object JSON")
        return {str(k): str(v) for k, v in parsed.items()}
    raise ValueError("extra_headers harus berupa object JSON")


def _clean_provider(body: dict, *, partial: bool = False) -> dict:
    out: dict[str, Any] = {}
    if "name" in body or not partial:
        name = str(body.get("name") or "").strip()
        if not name:
            raise ValueError("name provider wajib diisi")
        out["name"] = name
    if "base_url" in body or not partial:
        base_url = str(body.get("base_url") or "").strip()
        if not base_url:
            raise ValueError("base_url wajib diisi")
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("base_url harus diawali http:// atau https://")
        out["base_url"] = base_url
    if "kind" in body or not partial:
        kind = str(body.get("kind") or "openai").strip().lower()
        if kind not in KINDS:
            raise ValueError(f"kind '{kind}' belum didukung (pilihan: {', '.join(KINDS)})")
        out["kind"] = kind
    if "chat_path" in body:
        out["chat_path"] = str(body.get("chat_path") or DEFAULT_CHAT_PATH).strip() or DEFAULT_CHAT_PATH
    if "models_path" in body:
        out["models_path"] = str(body.get("models_path") or DEFAULT_MODELS_PATH).strip() or DEFAULT_MODELS_PATH
    if "admin_models_paths" in body:
        out["admin_models_paths"] = _clean_paths(body.get("admin_models_paths"))
    if "auth_scheme" in body:
        scheme = str(body.get("auth_scheme") or "bearer").strip().lower()
        if scheme not in ("bearer", "x-api-key", "query", "none"):
            raise ValueError("auth_scheme harus bearer, x-api-key, query, atau none")
        out["auth_scheme"] = scheme
    if "extra_headers" in body:
        out["extra_headers"] = _clean_headers(body.get("extra_headers"))
    for key in ("timeout_s", "max_concurrency"):
        if key in body:
            raw = body.get(key)
            out[key] = int(raw) if raw not in (None, "") else None
    if "enabled" in body:
        out["enabled"] = int(bool(body["enabled"]))
    if "api_key" in body:
        out["api_key"] = str(body["api_key"]).strip() if body["api_key"] else None
    if "admin_key" in body:
        out["admin_key"] = str(body["admin_key"]).strip() if body["admin_key"] else None
    return out


def provider_public(row: dict, *, model_count: Optional[int] = None) -> dict:
    out = {key: value for key, value in row.items() if key not in ("api_key", "admin_key")}
    out["has_api_key"] = bool(row.get("api_key"))
    out["has_admin_key"] = bool(row.get("admin_key"))
    if model_count is not None:
        out["model_count"] = model_count
    return out


def list_providers() -> list[dict]:
    out = []
    for row in db.list_providers():
        count = len(db.get_provider_models(row["id"]))
        out.append(provider_public(row, model_count=count))
    return out


def get_provider(provider_id: int) -> Optional[dict]:
    row = db.get_provider(provider_id)
    if not row:
        return None
    return provider_public(row, model_count=len(db.get_provider_models(provider_id)))


def get_provider_secrets(provider_id: int) -> dict:
    row = db.get_provider(provider_id)
    if not row:
        return {}
    return {"api_key": row.get("api_key"), "admin_key": row.get("admin_key")}


def create_provider(body: dict) -> int:
    fields = _clean_provider(body)
    fields.setdefault("chat_path", DEFAULT_CHAT_PATH)
    fields.setdefault("models_path", DEFAULT_MODELS_PATH)
    fields.setdefault("admin_models_paths", DEFAULT_ADMIN_PATHS)
    fields.setdefault("auth_scheme", "bearer")
    try:
        return db.create_provider(fields)
    except sqlite3.IntegrityError as exc:
        raise ValueError("nama provider sudah dipakai") from exc


def update_provider(provider_id: int, body: dict) -> None:
    if not db.get_provider(provider_id):
        raise ValueError("provider tidak ditemukan")
    fields = _clean_provider(body, partial=True)
    try:
        db.update_provider(provider_id, **fields)
    except sqlite3.IntegrityError as exc:
        raise ValueError("nama provider sudah dipakai") from exc


def set_secrets(provider_id: int, body: dict) -> None:
    if not db.get_provider(provider_id):
        raise ValueError("provider tidak ditemukan")
    api_key = db.UNSET if "api_key" not in body else (str(body["api_key"]).strip() or None)
    admin_key = db.UNSET if "admin_key" not in body else (str(body["admin_key"]).strip() or None)
    db.set_provider_secrets(provider_id, api_key=api_key, admin_key=admin_key)


async def test_provider(provider_id: int) -> dict:
    """Check discovery, then verify the customer path with a small chat call."""
    row = db.get_provider(provider_id)
    if not row:
        raise ValueError("provider tidak ditemukan")
    provider = build_provider(row)
    started = time.perf_counter()
    try:
        models = await provider.list_models()
        latency_ms = int((time.perf_counter() - started) * 1000)
        candidate = next(
            (model for model in models
             if model.mode in (None, "chat") and not model.blocked),
            None,
        )
        chat = None
        auth_ok = True
        if candidate is not None:
            chat_started = time.perf_counter()
            result = await provider.complete(
                candidate.name,
                [{"role": "user", "content": "Reply with exactly: OK"}],
                max_tokens=16,
                temperature=0,
                timeout_s=min(20.0, float(row.get("timeout_s") or 60)),
            )
            chat = {
                "model": candidate.name,
                "ok": result.status_code == 200 and not result.error,
                "latency_ms": int((time.perf_counter() - chat_started) * 1000),
                "http_status": result.status_code,
                "error": result.error or (
                    result.text if result.status_code not in (None, 200) else None
                ),
            }
            auth_ok = result.status_code not in (401, 403)
        out = {
            "ok": auth_ok,
            "latency_ms": latency_ms,
            "model_count": len(models),
            "sample": [model.name for model in models[:10]],
            "chat": chat,
        }
        if not auth_ok and chat:
            out["error"] = f"key ditolak provider (HTTP {chat['http_status']}): {chat['error']}"
        return out
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        await provider.aclose()


async def refresh_models(provider_id: int) -> dict:
    row = db.get_provider(provider_id)
    if not row:
        raise ValueError("provider tidak ditemukan")
    provider = build_provider(row)
    try:
        models = await provider.list_models()
    finally:
        await provider.aclose()

    existing = {model["model_name"] for model in db.get_provider_models(provider_id)}
    seen = []
    added = 0
    for model in models:
        _upsert_model(provider_id, model)
        seen.append(model.name)
        if model.name not in existing:
            added += 1
    db.disable_missing_provider_models(provider_id, seen)
    return {
        "ok": True,
        "total": len(models),
        "added": added,
        "updated": len(models) - added,
        "models": [model.name for model in models],
    }


def _upsert_model(provider_id: int, model: ModelInfo) -> None:
    db.upsert_provider_model(provider_id, {
        "model_name": model.name,
        "display_name": model.display_name,
        "mode": model.mode,
        "blocked": model.blocked,
        "supports_tools": model.supports_tools,
        "supports_json": model.supports_json,
        "input_cost_per_token": model.input_cost_per_token,
        "output_cost_per_token": model.output_cost_per_token,
        "context_window": model.context_window,
        "meta": model.meta,
    })


def catalog() -> dict:
    providers = []
    for row in db.list_providers():
        models = db.get_provider_models(row["id"])
        public = provider_public(row, model_count=len(models))
        public["models"] = models
        providers.append(public)
    return {"providers": providers}


def provider_models(provider_id: int) -> list[dict]:
    return db.get_provider_models(provider_id)