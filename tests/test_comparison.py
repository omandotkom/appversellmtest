from __future__ import annotations

from app.comparison import build_comparison

WEIGHTS = {
    "functional_tok_per_s": 0.30,
    "ttft_ms": 0.20,
    "pass_rate": 0.20,
    "itl_ms": 0.10,
    "avg_latency_ms": 0.10,
    "cost_usd": 0.10,
}


def _row(provider_id, provider_name, model, *, tok_s, ttft, itl, latency, cost, tools="pass"):
    return {
        "provider_id": provider_id,
        "provider_name": provider_name,
        "model": model,
        "status": "pass",
        "checks": {
            "basic": {"status": "pass", "latency_ms": latency},
            "streaming": {"status": "pass", "latency_ms": latency, "tok_per_s": tok_s,
                          "p50_ttft_ms": ttft, "ttft_ms": ttft, "itl_ms": itl},
            "tools": {"status": tools, "latency_ms": latency},
            "json": {"status": "pass", "latency_ms": latency},
        },
        "metrics": {"avg_latency_ms": latency, "cost_usd": cost},
    }


def test_build_comparison_scores_faster_model_higher():
    rows = [
        _row(1, "A", "fast", tok_s=100, ttft=100, itl=10, latency=150, cost=0.01),
        _row(2, "B", "slow", tok_s=50, ttft=200, itl=20, latency=300, cost=0.02),
    ]
    result = build_comparison(rows, weights=WEIGHTS)

    models = {m["model"]: m for m in result["models"]}
    assert models["fast"]["score"] > models["slow"]["score"]
    assert models["fast"]["metrics"]["functional_tok_per_s"]["score"] == 100.0
    assert models["slow"]["metrics"]["functional_tok_per_s"]["score"] == 50.0


def test_build_comparison_pass_rate_and_provider_rollup():
    rows = [
        _row(1, "A", "m1", tok_s=100, ttft=100, itl=10, latency=150, cost=0.01),
        _row(1, "A", "m2", tok_s=80, ttft=120, itl=12, latency=180, cost=0.02, tools="warn"),
        _row(2, "B", "m3", tok_s=60, ttft=150, itl=15, latency=220, cost=0.03),
    ]
    result = build_comparison(rows, weights=WEIGHTS)

    models = {m["model"]: m for m in result["models"]}
    assert models["m2"]["metrics"]["pass_rate"]["value"] == 0.75
    assert models["m1"]["metrics"]["pass_rate"]["value"] == 1.0

    providers = {p["provider_id"]: p for p in result["providers"]}
    assert providers[1]["models"] == 2
    assert providers[2]["models"] == 1
    assert providers[1]["best_model"] in {"m1", "m2"}


def test_missing_metric_is_excluded_from_score():
    rows = [
        _row(1, "A", "m1", tok_s=100, ttft=100, itl=10, latency=150, cost=None),
        _row(2, "B", "m2", tok_s=50, ttft=200, itl=20, latency=300, cost=0.02),
    ]
    result = build_comparison(rows, weights=WEIGHTS)
    m1 = next(m for m in result["models"] if m["model"] == "m1")
    assert m1["metrics"]["cost_usd"]["value"] is None
    assert m1["score"] is not None


def test_zero_cost_is_treated_as_best():
    rows = [
        _row(1, "A", "free", tok_s=100, ttft=100, itl=10, latency=150, cost=0.0),
        _row(2, "B", "paid", tok_s=100, ttft=100, itl=10, latency=150, cost=0.10),
    ]
    result = build_comparison(rows, weights=WEIGHTS)
    models = {m["model"]: m for m in result["models"]}
    assert models["free"]["metrics"]["cost_usd"]["score"] == 100.0
    assert models["paid"]["metrics"]["cost_usd"]["score"] < 100.0