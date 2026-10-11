"""Speed comparison across models: aggregate historical run metrics into
per-model speed profiles plus a weighted overall score (0-100)."""
from __future__ import annotations

import statistics
from typing import Any, Optional

from . import db

# Metric weights for the overall score (higher-is-better after normalization).
_SCORE_WEIGHTS = {
    "functional_tok_per_s": 0.30,
    "ttft_ms": 0.20,
    "load_throughput_rps": 0.15,
    "prefill_tok_per_s": 0.10,
    "bench_tok_per_s": 0.15,
    "itl_ms": 0.10,
    "bench_ttft_ms": 0.00,  # covered by functional ttft
}


def _median_or_none(values: list[float]) -> Optional[float]:
    clean = [v for v in values if v is not None and v > 0]
    return round(statistics.median(clean), 1) if clean else None


def _extract(row: dict, kind: str) -> dict[str, Any]:
    """Pull comparable speed metrics out of one run_models row."""
    checks = row.get("checks") or {}
    metrics = row.get("metrics") or {}
    out: dict[str, Any] = {}
    if kind == "functional":
        streaming = checks.get("streaming") or {}
        basic = checks.get("basic") or {}
        # Streaming tok/s is measured during generation only — the honest figure.
        out["functional_tok_per_s"] = streaming.get("tok_per_s") or basic.get("tok_per_s")
        out["ttft_ms"] = streaming.get("p50_ttft_ms") or streaming.get("ttft_ms")
        out["itl_ms"] = streaming.get("itl_ms")
    elif kind == "load":
        out["load_throughput_rps"] = max(
            (l["throughput_rps"] for l in metrics.get("levels") or []
             if l.get("throughput_rps") is not None), default=None)
        p95s = [l["p95_ms"] for l in metrics.get("levels") or [] if l.get("p95_ms") is not None]
        out["load_p95_ms"] = max(p95s) if p95s else None
        out["load_total_429"] = metrics.get("total_429")
    elif kind == "context":
        out["prefill_levels"] = [
            {"target_tokens": l.get("target_tokens"), "prefill_tok_per_s": l.get("prefill_tok_per_s")}
            for l in metrics.get("levels") or []
            if l.get("prefill_tok_per_s") is not None
        ]
        out["prefill_tok_per_s"] = metrics.get("max_prefill_tok_per_s")
    elif kind == "bench":
        out["bench_tok_per_s"] = metrics.get("gen_tok_per_s")
        out["bench_ttft_ms"] = metrics.get("avg_turn_ttft_ms")
    return out


def _normalize(values: dict[str, Optional[float]], higher_better: bool) -> dict[str, Optional[float]]:
    """Scale each metric to 0-100 across models, relative to the best model."""
    clean = {m: v for m, v in values.items() if v is not None and v > 0}
    if not clean:
        return {m: None for m in values}
    best = max(clean.values()) if higher_better else min(clean.values())
    out: dict[str, Optional[float]] = {}
    for m, v in values.items():
        if v is None or v <= 0:
            out[m] = None
        elif higher_better:
            out[m] = round(100.0 * v / best, 1)
        else:
            out[m] = round(100.0 * best / v, 1)
    return out


def comparison(kinds: list[str] | None = None) -> dict:
    kinds = kinds or ["functional", "bench", "load", "context"]
    rows = db.all_run_models_with_runs(kinds)
    per_model: dict[str, dict] = {}
    for row in rows:
        model = row["model"]
        if row["status"] in ("pending", "running", "skipped", "error"):
            continue
        rkind = row["kind"]
        prof = per_model.setdefault(model, {"runs": 0})
        prof["runs"] += 1
        extracted = _extract(row, rkind)
        for key, value in extracted.items():
            if isinstance(value, list) and value:
                prof.setdefault(key, []).extend(value)
                continue
            if value is None:
                continue
            prof.setdefault(key, []).append(value)
    # collapse collected lists of numbers into medians
    for prof in per_model.values():
        for key in list(prof):
            if isinstance(prof[key], list) and key != "prefill_levels":
                prof[key] = _median_or_none(prof[key])

    models = []
    metric_specs = [  # (key, higher is better)
        ("functional_tok_per_s", True),
        ("ttft_ms", False),
        ("itl_ms", False),
        ("load_throughput_rps", True),
        ("prefill_tok_per_s", True),
        ("bench_tok_per_s", True),
        ("bench_ttft_ms", False),
    ]
    normalized: dict[str, dict[str, Optional[float]]] = {}
    present = {key for prof in per_model.values() for key, _ in metric_specs if prof.get(key) is not None}
    for key, higher_better in metric_specs:
        if key not in present:
            continue
        normalized[key] = _normalize({m: p.get(key) for m, p in per_model.items()}, higher_better)

    # ttft score combines functional ttft and bench ttft when both exist
    def model_score(name: str) -> Optional[float]:
        total, weight_sum = 0.0, 0.0
        for key in present:
            value = normalized[key].get(name)
            if value is None:
                continue
            weight = _SCORE_WEIGHTS.get(key, 0.0)
            if weight <= 0:
                continue
            total += value * weight
            weight_sum += weight
        return round(total / weight_sum, 1) if weight_sum else None

    for name, prof in per_model.items():
        prof["chart_key"] = name
        prof["score"] = model_score(name)
        profiles: dict[str, Any] = {}
        for key in present:
            profiles[key] = {"value": prof.get(key), "score": normalized[key].get(name)}
        models.append({"model": name, "run_count": prof.get("runs", 0),
                       "score": prof["score"], "metrics": profiles,
                       "prefill_levels": prof.get("prefill_levels") or []})
    models.sort(key=lambda m: (m["score"] is None, -(m["score"] or 0), m["model"]))
    return {"models": models, "weights": {k: v for k, v in _SCORE_WEIGHTS.items() if v > 0}}
