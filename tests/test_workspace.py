from __future__ import annotations

from app import db, main, targets


def test_bench_workspace_is_isolated_per_provider(temp_db):
    provider_a = db.create_provider({"name": "A", "base_url": "http://localhost:4010"})
    provider_b = db.create_provider({"name": "B", "base_url": "http://localhost:4010"})
    db.upsert_provider_model(provider_a, {"model_name": "same-model"})
    db.upsert_provider_model(provider_b, {"model_name": "same-model"})

    resolved = targets.resolve_targets([
        {"provider_id": provider_a, "model": "same-model"},
        {"provider_id": provider_b, "model": "same-model"},
    ])
    run_id = db.create_run("bench", {})
    main._attach_run_models(run_id, resolved, bench=True)

    rows = db.get_run_models(run_id)
    workspaces = {row["workspace"] for row in rows}
    assert len(workspaces) == 2
    assert all(f"p{provider_a}" in workspace or f"p{provider_b}" in workspace
               for workspace in workspaces)
    assert workspaces != {None}