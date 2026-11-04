"""Resolve user selections into concrete provider + model targets."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from . import db


@dataclass
class Target:
    provider: dict
    model: dict
    run_model_id: Optional[int] = None

    @property
    def provider_id(self) -> int:
        return int(self.provider["id"])

    @property
    def provider_name(self) -> str:
        return str(self.provider.get("name") or f"provider-{self.provider_id}")

    @property
    def model_name(self) -> str:
        return str(self.model["model_name"])

    def key(self) -> str:
        return f"p{self.provider_id}:{self.model_name}"

    def as_dict(self) -> dict:
        return {
            "provider_id": self.provider_id,
            "provider_name": self.provider_name,
            "model": self.model_name,
        }


def _is_chat(model: dict) -> bool:
    return model.get("mode") in (None, "chat")


def _blocked(model: dict) -> bool:
    return bool(model.get("blocked"))


def _find_model(provider_id: int, model_name: str) -> Optional[dict]:
    for model in db.get_provider_models(provider_id):
        if model["model_name"] == model_name:
            return model
    return None


def _target(provider: dict, model: dict, include_blocked: bool) -> Target:
    if not provider.get("enabled"):
        raise ValueError(f"provider '{provider.get('name')}' sedang disabled")
    if not model.get("enabled"):
        raise ValueError(f"model '{model['model_name']}' sedang disabled di provider '{provider.get('name')}'")
    if not _is_chat(model):
        raise ValueError(f"model '{model['model_name']}' bukan model chat")
    if _blocked(model) and not include_blocked:
        raise ValueError(f"model '{model['model_name']}' diblokir (aktifkan 'ikutkan model blocked')")
    return Target(provider=provider, model=model)


def resolve_targets(
    raw_targets: Optional[list] = None,
    legacy_models: Optional[list] = None,
    *,
    include_blocked: bool = False,
) -> list[Target]:
    """Resolve explicit targets, legacy model-name selections, or defaults."""
    resolved: list[Target] = []
    seen: set[tuple[int, str]] = set()

    if raw_targets:
        for item in raw_targets:
            if isinstance(item, str):
                provider_part, _, model_name = item.partition(":")
                try:
                    provider_id = int(provider_part.lstrip("pP"))
                except ValueError as exc:
                    raise ValueError(f"target '{item}' tidak valid (format: provider_id:model)") from exc
            elif isinstance(item, dict):
                try:
                    provider_id = int(item.get("provider_id"))
                except (TypeError, ValueError) as exc:
                    raise ValueError("setiap target wajib punya provider_id") from exc
                model_name = str(item.get("model") or "").strip()
            else:
                raise ValueError("target harus berupa object {provider_id, model}")
            if not model_name:
                raise ValueError("setiap target wajib punya model")
            provider = db.get_provider(provider_id)
            if not provider:
                raise ValueError(f"provider {provider_id} tidak ditemukan")
            model = _find_model(provider_id, model_name)
            if not model:
                raise ValueError(f"model '{model_name}' tidak ditemukan di provider '{provider['name']}'")
            key = (provider_id, model_name)
            if key in seen:
                continue
            seen.add(key)
            resolved.append(_target(provider, model, include_blocked))
        return resolved

    if legacy_models:
        for name in legacy_models:
            matches = [row for row in db.list_models_by_name(str(name))
                       if row.get("provider_enabled") and row.get("enabled")]
            if not matches:
                raise ValueError(f"model '{name}' tidak ditemukan di provider yang aktif")
            if len(matches) > 1:
                providers = ", ".join(f"{row['provider_name']} (id {row['provider_id']})" for row in matches)
                raise ValueError(
                    f"model '{name}' ambiguous di beberapa provider: {providers}. "
                    "Pilih target eksplisit {provider_id, model}."
                )
            row = matches[0]
            provider = db.get_provider(row["provider_id"])
            if not provider:
                raise ValueError(f"provider {row['provider_id']} tidak ditemukan")
            key = (row["provider_id"], row["model_name"])
            if key in seen:
                continue
            seen.add(key)
            resolved.append(_target(provider, row, include_blocked))
        return resolved

    for provider in db.list_providers(enabled_only=True):
        for model in db.get_provider_models(provider["id"], enabled_only=True):
            if not _is_chat(model):
                continue
            if _blocked(model) and not include_blocked:
                continue
            key = (provider["id"], model["model_name"])
            if key in seen:
                continue
            seen.add(key)
            resolved.append(Target(provider=provider, model=model))

    resolved.sort(key=lambda target: (target.provider_name.lower(), target.model_name.lower()))
    return resolved