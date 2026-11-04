from __future__ import annotations

import pytest

from app import db, targets


def _seed():
    a = db.create_provider({"name": "A", "base_url": "http://a.test"})
    b = db.create_provider({"name": "B", "base_url": "http://b.test"})
    db.upsert_provider_model(a, {"model_name": "shared", "mode": "chat"})
    db.upsert_provider_model(b, {"model_name": "shared", "mode": "chat"})
    db.upsert_provider_model(b, {"model_name": "only-b", "mode": "chat"})
    return a, b


def test_explicit_targets_resolve(temp_db):
    a, b = _seed()
    resolved = targets.resolve_targets([
        {"provider_id": a, "model": "shared"},
        {"provider_id": b, "model": "only-b"},
    ])
    assert [(t.provider_id, t.model_name) for t in resolved] == [(a, "shared"), (b, "only-b")]


def test_ambiguous_legacy_name_is_rejected(temp_db):
    _seed()
    with pytest.raises(ValueError, match="ambiguous"):
        targets.resolve_targets(None, ["shared"])


def test_legacy_name_with_single_match_resolves(temp_db):
    a, _ = _seed()
    resolved = targets.resolve_targets(None, ["only-b"])
    assert len(resolved) == 1
    assert resolved[0].model_name == "only-b"


def test_disabled_provider_is_rejected(temp_db):
    a, _ = _seed()
    db.update_provider(a, enabled=0)
    with pytest.raises(ValueError, match="disabled"):
        targets.resolve_targets([{"provider_id": a, "model": "shared"}])


def test_default_selection_is_all_chat_models(temp_db):
    _seed()
    resolved = targets.resolve_targets(None, None)
    assert len(resolved) == 3