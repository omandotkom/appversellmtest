from __future__ import annotations

import sqlite3

from app import db


def test_init_backs_up_and_recreates_old_schema(tmp_path, monkeypatch):
    from app import config

    path = tmp_path / "health.db"
    monkeypatch.setattr(config, "db_path", lambda: path)

    old = sqlite3.connect(path)
    old.execute("PRAGMA user_version = 1")
    old.execute("CREATE TABLE runs (id INTEGER PRIMARY KEY, kind TEXT)")
    old.execute("INSERT INTO runs (kind) VALUES ('functional')")
    old.commit()
    old.close()

    db.init()

    backups = list(tmp_path.glob("health.db.bak.*"))
    assert len(backups) == 1

    conn = sqlite3.connect(path)
    try:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()
    assert version == 2
    assert {"providers", "provider_models", "runs", "run_models", "steps"} <= tables
    assert db.list_runs() == []


def test_provider_crud_and_model_upsert(temp_db):
    pid = db.create_provider({
        "name": "OpenRouter",
        "kind": "openai",
        "base_url": "http://localhost:4010",
        "api_key": "sk-test",
    })
    provider = db.get_provider(pid)
    assert provider["name"] == "OpenRouter"
    assert provider["api_key"] == "sk-test"

    db.upsert_provider_model(pid, {
        "model_name": "mock/good",
        "mode": "chat",
        "supports_tools": True,
        "supports_json": True,
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 3e-6,
    })
    models = db.get_provider_models(pid)
    assert len(models) == 1
    assert models[0]["model_name"] == "mock/good"

    # discovery must not overwrite a manual price override
    db.update_provider_model(models[0]["id"], input_cost_per_token=2e-6)
    db.upsert_provider_model(pid, {"model_name": "mock/good", "mode": "chat"})
    assert db.get_provider_model(models[0]["id"])["input_cost_per_token"] == 2e-6


def test_model_history_includes_functional_pass_rows(temp_db):
    pid = db.create_provider({"name": "P", "base_url": "http://localhost:4010"})
    run_id = db.create_run("functional", {})
    rm_id = db.create_run_model(run_id, pid, "P", "mock-good")
    db.update_run_model(rm_id, status="pass", metrics={"avg_latency_ms": 123})
    db.finish_run(run_id, "done", {})

    history = db.model_history(pid, "mock-good", "functional", None, 5)
    assert len(history) == 1
    assert history[0]["run_id"] == run_id
    assert history[0]["metrics"]["avg_latency_ms"] == 123


def test_discovery_disables_missing_models(temp_db):
    pid = db.create_provider({"name": "P", "base_url": "http://localhost:4010"})
    db.upsert_provider_model(pid, {"model_name": "a"})
    db.upsert_provider_model(pid, {"model_name": "b"})
    db.disable_missing_provider_models(pid, ["a"])
    rows = {row["model_name"]: row for row in db.get_provider_models(pid)}
    assert rows["a"]["enabled"] == 1
    assert rows["b"]["enabled"] == 0