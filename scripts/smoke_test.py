"""End-to-end smoke test against a running dashboard (backed by the mock gateway).

Usage:
  1. python mock/mock_gateway.py          (terminal 1)
  2. python run.py                        (terminal 2)
  3. python scripts/smoke_test.py
"""
from __future__ import annotations

import os
import time

import httpx

APP = os.environ.get("APP_URL", "http://127.0.0.1:8000")
MOCK = os.environ.get("MOCK_URL", "http://127.0.0.1:4010")


def wait_run(client: httpx.Client, run_id: int, timeout: int = 240) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        data = client.get(f"{APP}/api/runs/{run_id}").json()
        if data["run"]["status"] != "running":
            return data
        time.sleep(1.0)
    raise SystemExit(f"run {run_id} tidak selesai dalam {timeout}s")


def ensure_provider(client: httpx.Client, name: str, base_url: str) -> int:
    providers = client.get(f"{APP}/api/providers").json()["providers"]
    existing = next((p for p in providers if p["name"] == name), None)
    payload = {
        "name": name,
        "kind": "openai",
        "base_url": base_url,
        "api_key": "sk-smoke",
        "admin_key": "sk-smoke-admin",
        "auth_scheme": "bearer",
        "enabled": 1,
    }
    if existing:
        client.patch(f"{APP}/api/providers/{existing['id']}", json=payload).raise_for_status()
        client.put(f"{APP}/api/providers/{existing['id']}/secrets",
                   json={"api_key": "sk-smoke", "admin_key": "sk-smoke-admin"}).raise_for_status()
        provider_id = existing["id"]
    else:
        created = client.post(f"{APP}/api/providers", json=payload)
        created.raise_for_status()
        provider_id = created.json()["provider"]["id"]
    tested = client.post(f"{APP}/api/providers/{provider_id}/test").json()
    assert tested.get("ok"), tested
    refreshed = client.post(f"{APP}/api/providers/{provider_id}/refresh-models").json()
    assert refreshed.get("ok") and refreshed["total"] > 0, refreshed
    return provider_id


def key(provider_id: int, model: str) -> str:
    return f"p{provider_id}:{model}"


def main() -> None:
    client = httpx.Client(timeout=120)
    status = client.get(f"{APP}/api/status").json()
    print("status:", status)

    a = ensure_provider(client, "Mock Provider A", f"{MOCK}/a")
    b = ensure_provider(client, "Mock Provider B", f"{MOCK}/b")
    # Partial secret updates must not wipe the other key.
    client.put(f"{APP}/api/providers/{a}/secrets", json={"api_key": "sk-smoke-rotated"}).raise_for_status()
    provider_a = client.get(f"{APP}/api/providers/{a}").json()["provider"]
    assert provider_a["has_api_key"] and provider_a["has_admin_key"], provider_a
    print(f"providers: A={a} B={b}")

    catalog = client.get(f"{APP}/api/catalog").json()["providers"]
    by_name = {provider["name"]: provider for provider in catalog}
    a_models = {model["model_name"] for model in by_name["Mock Provider A"]["models"]}
    b_models = {model["model_name"] for model in by_name["Mock Provider B"]["models"]}
    assert "mock-good" in a_models and "mock-good" in b_models
    assert "mock-no-tools" in a_models
    assert "mock-rate-limited" in b_models
    print(f"catalog: A={len(a_models)} model, B={len(b_models)} model")

    # --- functional with repeat + warmup, explicit cross-provider targets ---
    run = client.post(f"{APP}/api/runs/functional", json={
        "targets": [
            {"provider_id": a, "model": "mock-good"},
            {"provider_id": b, "model": "mock-good"},
            {"provider_id": a, "model": "mock-no-tools"},
        ],
        "repeat": 2, "warmup": True,
    }).json()
    data = wait_run(client, run["run_id"])
    assert data["run"]["status"] == "done", data["run"]
    by_key = {key(row["provider_id"], row["model"]): row for row in data["models"]}
    good_a = by_key[key(a, "mock-good")]
    good_b = by_key[key(b, "mock-good")]
    no_tools = by_key[key(a, "mock-no-tools")]
    assert good_a["provider_name"] == "Mock Provider A"
    assert good_b["provider_name"] == "Mock Provider B"
    assert len(good_a["checks"]["basic"]["samples"]) == 2, good_a["checks"]["basic"]
    assert good_a["checks"]["streaming"]["ttft_ms"] is not None
    assert good_a["checks"]["streaming"].get("tok_per_s") is not None
    assert good_a["metrics"]["p95_latency_ms"] is not None
    assert no_tools["checks"]["tools"]["status"] == "na", no_tools["checks"]["tools"]
    print(f"functional: A good p95={good_a['metrics']['p95_latency_ms']} tok/s={good_a['checks']['streaming'].get('tok_per_s')} "
          f"B good status={good_b['status']} tools A no-tools={no_tools['checks']['tools']['status']}")

    # --- comparison run: full 4 checks + repeat + warmup ---
    run = client.post(f"{APP}/api/runs/comparison", json={
        "targets": [
            {"provider_id": a, "model": "mock-good"},
            {"provider_id": b, "model": "mock-good"},
            {"provider_id": b, "model": "mock-flaky"},
            {"provider_id": a, "model": "mock-slow"},
        ],
        "repeat": 2,
    }).json()
    cmp_run = wait_run(client, run["run_id"], timeout=240)
    assert cmp_run["run"]["status"] == "done", cmp_run["run"]
    assert len(cmp_run["models"]) == 4
    cmp = client.get(f"{APP}/api/comparison", params={"run_id": run["run_id"]}).json()
    assert len(cmp["models"]) == 4, cmp
    assert len(cmp["providers"]) == 2, cmp["providers"]
    names = {(model["provider_name"], model["model"]) for model in cmp["models"]}
    assert ("Mock Provider A", "mock-good") in names
    assert ("Mock Provider B", "mock-good") in names
    top = cmp["models"][0]
    assert top["score"] is not None and 0 <= top["score"] <= 100, top
    assert top["metrics"]["functional_tok_per_s"]["value"] is not None, top
    scores = [model["score"] for model in cmp["models"] if model["score"] is not None]
    assert scores == sorted(scores, reverse=True), scores
    print(f"comparison: top={top['provider_name']}/{top['model']} score={top['score']} "
          f"tok/s={top['metrics']['functional_tok_per_s']['value']} providers={len(cmp['providers'])}")

    # --- load probe ---
    run = client.post(f"{APP}/api/runs/load", json={
        "targets": [
            {"provider_id": a, "model": "mock-good"},
            {"provider_id": b, "model": "mock-rate-limited"},
        ],
        "levels": [2], "requests_per_level": 4,
    }).json()
    data = wait_run(client, run["run_id"])
    rl = next(row for row in data["models"] if row["model"] == "mock-rate-limited")
    good_load = next(row for row in data["models"] if row["model"] == "mock-good")
    assert rl["metrics"]["total_429"] >= 4, rl["metrics"]
    assert good_load["metrics"]["max_throughput_rps"] is not None
    print(f"load: A good throughput={good_load['metrics']['max_throughput_rps']} "
          f"B rate-limited 429={rl['metrics']['total_429']}")

    # --- context scaling ---
    run = client.post(f"{APP}/api/runs/context", json={
        "targets": [
            {"provider_id": a, "model": "mock-good"},
            {"provider_id": b, "model": "mock-small-context"},
        ],
        "sizes": [1000, 32000],
    }).json()
    data = wait_run(client, run["run_id"])
    small = next(row for row in data["models"] if row["model"] == "mock-small-context")
    good_ctx = next(row for row in data["models"] if row["model"] == "mock-good")
    statuses = [level["status"] for level in small["metrics"]["levels"]]
    assert "context_exceeded" in statuses, statuses
    assert good_ctx["metrics"]["max_prefill_tok_per_s"] is not None
    print(f"context: B small statuses={statuses} A good prefill={good_ctx['metrics']['max_prefill_tok_per_s']}")

    # --- benchmark (streaming turns), two providers with duplicate model name ---
    run = client.post(f"{APP}/api/runs/bench", json={
        "task_id": "landing-page",
        "targets": [
            {"provider_id": a, "model": "mock-good"},
            {"provider_id": a, "model": "mock-flaky"},
            {"provider_id": a, "model": "mock-lazy"},
            {"provider_id": a, "model": "mock-broken"},
            {"provider_id": b, "model": "mock-good"},
        ],
        "max_turns": 6,
    }).json()
    data = wait_run(client, run["run_id"])
    outcomes = {}
    for row in data["models"]:
        metrics = row.get("metrics") or {}
        outcomes[key(row["provider_id"], row["model"])] = row["outcome"]
        print(f"bench {row['provider_name']}/{row['model']}: outcome={row['outcome']} "
              f"error_rate={metrics.get('error_rate')} ttft/turn={metrics.get('avg_turn_ttft_ms')} "
              f"tok/s={metrics.get('gen_tok_per_s')}")

    assert outcomes.get(key(a, "mock-good")) == "completed", outcomes
    assert outcomes.get(key(b, "mock-good")) == "completed", outcomes
    assert outcomes.get(key(a, "mock-flaky")) == "completed", outcomes
    assert outcomes.get(key(a, "mock-lazy")) == "completed_incomplete", outcomes
    assert outcomes.get(key(a, "mock-broken")) == "max_turns", outcomes

    rm_good = next(row for row in data["models"]
                   if row["provider_id"] == a and row["model"] == "mock-good")
    assert rm_good["metrics"]["avg_turn_ttft_ms"] is not None
    detail = client.get(f"{APP}/api/runs/{run['run_id']}/models/{rm_good['id']}").json()
    assistant_steps = [step for step in detail["steps"] if step["type"] == "assistant"]
    assert any(step.get("ttft_ms") is not None for step in assistant_steps)

    files = client.get(f"{APP}/api/bench/{rm_good['id']}/files").json()["files"]
    assert any(file["path"] == "index.html" for file in files), files
    html = client.get(f"{APP}/api/bench/{rm_good['id']}/file", params={"path": "index.html"})
    assert html.status_code == 200 and len(html.text) > 800, html.status_code

    # --- provider-aware history + historical comparison ---
    hist = client.get(f"{APP}/api/models/history",
                      params={"provider_id": a, "model": "mock-good", "kind": "functional"}).json()
    assert "points" in hist and len(hist["points"]) >= 1, hist

    hist_cmp = client.get(f"{APP}/api/comparison", params={
        "mode": "median", "window": 5,
        "provider_ids": f"{a},{b}",
    }).json()
    assert len(hist_cmp["providers"]) == 2, hist_cmp["providers"]
    assert len(hist_cmp["models"]) >= 2
    print(f"history comparison: {len(hist_cmp['models'])} target, {len(hist_cmp['providers'])} provider")

    # --- soft delete / re-enable ---
    client.delete(f"{APP}/api/providers/{b}").raise_for_status()
    providers = client.get(f"{APP}/api/providers").json()["providers"]
    disabled = next(provider for provider in providers if provider["id"] == b)
    assert disabled["enabled"] == 0, disabled
    client.patch(f"{APP}/api/providers/{b}", json={"enabled": 1}).raise_for_status()

    print("SMOKE OK")


if __name__ == "__main__":
    main()