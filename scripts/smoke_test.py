"""End-to-end smoke test against a running dashboard (backed by the mock gateway).

Usage:
  1. python mock/mock_gateway.py          (terminal 1)
  2. python run.py                        (terminal 2)
  3. python scripts/smoke_test.py
"""
from __future__ import annotations

import json
import os
import time

import httpx

APP = os.environ.get("APP_URL", "http://127.0.0.1:8000")


def wait_run(client: httpx.Client, run_id: int, timeout: int = 120) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        data = client.get(f"{APP}/api/runs/{run_id}").json()
        if data["run"]["status"] != "running":
            return data
        time.sleep(1.0)
    raise SystemExit(f"run {run_id} tidak selesai dalam {timeout}s")


def main() -> None:
    client = httpx.Client(timeout=60)
    status = client.get(f"{APP}/api/status").json()
    print("status:", status)

    models = client.get(f"{APP}/api/models?refresh=1").json()["models"]
    print("models:", [m["name"] for m in models])

    # --- functional with repeat + warmup + deep latency ---
    run = client.post(f"{APP}/api/runs/functional", json={
        "models": ["mock-good", "mock-no-tools"],
        "repeat": 2, "warmup": True,
    }).json()
    data = wait_run(client, run["run_id"])
    assert data["run"]["status"] == "done", data["run"]
    by_name = {r["model"]: r for r in data["models"]}
    good = by_name["mock-good"]
    assert len(good["checks"]["basic"]["samples"]) == 2, good["checks"]["basic"]
    assert good["checks"]["streaming"]["ttft_ms"] is not None
    assert good["checks"]["streaming"].get("tok_per_s") is not None
    assert good["metrics"]["p95_latency_ms"] is not None
    assert by_name["mock-no-tools"]["checks"]["tools"]["status"] == "na"
    print(f"functional mock-good: status={good['status']} p95={good['metrics']['p95_latency_ms']} "
          f"tok/s={good['checks']['streaming'].get('tok_per_s')} samples={len(good['checks']['basic']['samples'])}")

    # --- load probe ---
    run = client.post(f"{APP}/api/runs/load", json={
        "models": ["mock-good", "mock-rate-limited"],
        "levels": [2], "requests_per_level": 4,
    }).json()
    data = wait_run(client, run["run_id"])
    rl = next(r for r in data["models"] if r["model"] == "mock-rate-limited")
    good_load = next(r for r in data["models"] if r["model"] == "mock-good")
    assert rl["metrics"]["total_429"] >= 4, rl["metrics"]
    assert good_load["metrics"]["max_throughput_rps"] is not None
    print(f"load mock-good throughput={good_load['metrics']['max_throughput_rps']} "
          f"mock-rate-limited 429={rl['metrics']['total_429']}")

    # --- context scaling ---
    run = client.post(f"{APP}/api/runs/context", json={
        "models": ["mock-good", "mock-small-context"],
        "sizes": [1000, 32000],
    }).json()
    data = wait_run(client, run["run_id"])
    sc = next(r for r in data["models"] if r["model"] == "mock-small-context")
    good_ctx = next(r for r in data["models"] if r["model"] == "mock-good")
    sc_statuses = [l["status"] for l in sc["metrics"]["levels"]]
    assert "context_exceeded" in sc_statuses, sc_statuses
    assert good_ctx["metrics"]["max_prefill_tok_per_s"] is not None
    print(f"context mock-small-context statuses={sc_statuses} mock-good prefill={good_ctx['metrics']['max_prefill_tok_per_s']}")

    # --- benchmark (streaming turns) ---
    run = client.post(f"{APP}/api/runs/bench", json={
        "task_id": "landing-page",
        "models": ["mock-good", "mock-flaky", "mock-lazy", "mock-broken"],
        "max_turns": 6,
    }).json()
    data = wait_run(client, run["run_id"])
    outcomes = {}
    for row in data["models"]:
        m = row.get("metrics") or {}
        outcomes[row["model"]] = row["outcome"]
        print(f"bench {row['model']}: outcome={row['outcome']} error_rate={m.get('error_rate')} "
              f"ttft/turn={m.get('avg_turn_ttft_ms')} tok/s={m.get('gen_tok_per_s')}")

    assert outcomes.get("mock-good") == "completed", outcomes
    assert outcomes.get("mock-flaky") == "completed", outcomes
    assert outcomes.get("mock-lazy") == "completed_incomplete", outcomes
    assert outcomes.get("mock-broken") == "max_turns", outcomes

    rm_good = next(r for r in data["models"] if r["model"] == "mock-good")
    assert rm_good["metrics"]["avg_turn_ttft_ms"] is not None
    detail = client.get(f"{APP}/api/runs/{run['run_id']}/models/{rm_good['id']}").json()
    assistant_steps = [s for s in detail["steps"] if s["type"] == "assistant"]
    assert any(s.get("ttft_ms") is not None for s in assistant_steps)

    # artifact fetch + preview
    files = client.get(f"{APP}/api/bench/{rm_good['id']}/files").json()["files"]
    assert any(f["path"] == "index.html" for f in files), files
    html = client.get(f"{APP}/api/bench/{rm_good['id']}/file", params={"path": "index.html"})
    assert html.status_code == 200 and len(html.text) > 800, html.status_code

    # history endpoint
    hist = client.get(f"{APP}/api/models/history", params={"model": "mock-good", "kind": "functional"}).json()
    assert "points" in hist and len(hist["points"]) >= 1, hist

    # --- speed comparison ---
    speed = client.get(f"{APP}/api/speed-comparison").json()
    names = {m["model"]: m for m in speed["models"]}
    assert "mock-good" in names, names
    good_speed = names["mock-good"]
    assert good_speed["metrics"]["functional_tok_per_s"]["value"] is not None
    assert good_speed["metrics"]["load_throughput_rps"]["value"] is not None
    assert good_speed["metrics"]["prefill_tok_per_s"]["value"] is not None
    assert good_speed["metrics"]["bench_tok_per_s"]["value"] is not None
    assert good_speed["score"] is not None and 0 <= good_speed["score"] <= 100
    sorted_scores = [m["score"] for m in speed["models"] if m["score"] is not None]
    assert sorted_scores == sorted(sorted_scores, reverse=True), sorted_scores
    print(f"speed mock-good score={good_speed['score']} "
          f"fn tok/s={good_speed['metrics']['functional_tok_per_s']['value']} "
          f"load={good_speed['metrics']['load_throughput_rps']['value']} "
          f"bench tok/s={good_speed['metrics']['bench_tok_per_s']['value']}")
    print("SMOKE OK")


if __name__ == "__main__":
    main()
