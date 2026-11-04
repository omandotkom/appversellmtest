from __future__ import annotations

from app import catalog, db


def test_provider_secrets_are_stored_but_never_public(temp_db):
    provider_id = catalog.create_provider({
        "name": "Secret Provider",
        "base_url": "http://localhost:4010",
        "api_key": "sk-customer",
        "admin_key": "sk-admin",
    })

    public = catalog.get_provider(provider_id)
    assert "api_key" not in public
    assert "admin_key" not in public
    assert public["has_api_key"] is True
    assert public["has_admin_key"] is True

    listed = catalog.list_providers()[0]
    assert "api_key" not in listed
    assert "admin_key" not in listed

    stored = db.get_provider(provider_id)
    assert stored["api_key"] == "sk-customer"
    assert stored["admin_key"] == "sk-admin"


def test_set_secrets_only_updates_supplied_keys(temp_db):
    provider_id = catalog.create_provider({
        "name": "P",
        "base_url": "http://localhost:4010",
        "api_key": "old-api",
        "admin_key": "old-admin",
    })

    catalog.set_secrets(provider_id, {"api_key": "new-api"})
    stored = db.get_provider(provider_id)
    assert stored["api_key"] == "new-api"
    assert stored["admin_key"] == "old-admin"