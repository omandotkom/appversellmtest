"""Provider registry."""
from __future__ import annotations

from .base import BaseProvider, ChatResult, ModelInfo
from .openai_compat import OpenAICompatProvider

_PROVIDERS: dict[str, type[BaseProvider]] = {
    "openai": OpenAICompatProvider,
}

KINDS = tuple(sorted(_PROVIDERS))


def build_provider(cfg: dict, transport=None) -> BaseProvider:
    kind = (cfg.get("kind") or "openai").lower()
    cls = _PROVIDERS.get(kind)
    if cls is None:
        raise ValueError(f"provider kind '{kind}' belum didukung (pilihan: {', '.join(KINDS)})")
    return cls(cfg, transport=transport)


__all__ = ["BaseProvider", "ChatResult", "ModelInfo", "KINDS", "build_provider"]