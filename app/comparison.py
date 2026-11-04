"""Cross-provider comparison built from full functional runs."""
from __future__ import annotations

import statistics
from typing import Any, Optional

from . import config

METRIC_SPECS: list[tuple[str, bool]] = [
    ("functional_tok_per_s", True),
    ("ttft_ms", False),
    ("pass_rate", True),
    ("itl_ms", False),
    ("avg_latency_ms", False),
    ("cost_usd", False),
]

CHECK_NAMES = ("basic", "streaming", "tools", "json")


def _num(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_row(row: dict) -> dict[str, Optional[float]]:
    checks = row.get("checks") or {}
    metrics = row.get("metrics") or {}
    streaming = checks.get("streaming") or {}
    basic = checks.get("basic") or {}

    statuses = []
    for name in CHECK_NAMES:
        check = checks.get(name)
        if isinstance(check, dict) and check.get("status") in ("pass", "warn", "fail"):
            statuses.append(check["status"])
    pass_rate = (statuses.count("pass") / len(statuses)) if statuses else None

    return {
        "functional_tok_per_s": _num(streaming.get("tok_per_s") or basic.get("tok_per_s")),
        "ttft_ms": _num(streaming.get("p50_ttft_ms") or streaming.get("ttft_ms")),
        "itl_ms": _num(streaming.get("itl_ms")),
        "avg_latency_ms": _num(metrics.get("avg_latency_ms") or metrics.get("p50_latency_ms")),
        "pass_rate": pass_rate,
        "cost_usd": _num(metrics.get("cost_usd")),
    }


def _normalize(values: dict[str, Optional[float]], higher_better: bool) -> dict[str, Optional[float]]:
    clean = {key: value for key, value in values.items() if value is not None}
    if not clean:
        return {key: None for key in values}

    if higher_better:
        best = max(clean.values())
        if best <= 0:
            return {key: (0.0 if value is not None else None) for key, value in values.items()}
        return {
            key: (round(100.0 * value / best, 1) if value is not None else None)
            for key, value in values.items()
        }

    best = min(clean.values())
    if best < 0:
        return {key: None for key in values}
    if best == 0:
        return {
            key: (100.0 if value == 0 else 0.0) if value is not None else None
            for key, value in values.items()
        }
    return {
        key: (round(100.0 * best / value, 1) if value not in (None, 0) else None)
        for key, value in values.items()
    }


def _score(normalized: dict[str, dict[str, Optional[float]]], weights: dict[str, float], name: str) -> Optional[float]:
    total = 0.0
    weight_sum = 0.0
    for key, weight in weights.items():
        if weight <= 0 or key not in normalized:
            continue
        value = normalized[key].get(name)
        if value is None:
            continue
        total += value * weight
        weight_sum += weight
    return round(total / weight_sum, 1) if weight_sum else None


def _provider_rollup(models: list[dict]) -> list[dict]:
    groups: dict[int, list[dict]] = {}
    for model in models:
        provider_id = model.get("provider_id")
        groups.setdefault(provider_id if provider_id is not None else 0, []).append(model)

    rollups = []
    for provider_id, members in groups.items():
        scores = [member["score"] for member in members if member.get("score") is not None]
        median: dict[str, Optional[float]] = {}
        for key, _ in METRIC_SPECS:
            values = [member["metrics"][key]["value"] for member in members
                      if member["metrics"].get(key) and member["metrics"][key]["value"] is not None]
            median[key] = round(statistics.median(values), 4) if values else None
        best = max(members, key=lambda member: member.get("score") or -1)
        rollups.append({
            "provider_id": provider_id,
            "name": members[0].get("provider_name") or "legacy",
            "models": len(members),
            "score": round(statistics.median(scores), 1) if scores else None,
            "median": median,
            "best_model": best.get("model") if best.get("score") is not None else None,
        })
    rollups.sort(key=lambda item: (item["score"] is None, -(item["score"] or 0), item["name"].lower()))
    return rollups


def build_comparison(
    rows: list[dict],
    *,
    weights: Optional[dict[str, float]] = None,
    run_id: Optional[int] = None,
    scope: str = "run",
    top_n: Optional[int] = None,
) -> dict:
    weights = {key: value for key, value in (weights or config.get_weights()).items() if value > 0}

    extracted: dict[str, dict[str, Optional[float]]] = {}
    names: dict[str, dict] = {}
    for row in rows:
        if row.get("status") in ("pending", "running", "skipped", "error"):
            continue
        provider_id = row.get("provider_id")
        name = f"p{provider_id if provider_id is not None else 0}:{row.get('model')}"
        if name in names:
            continue
        names[name] = row
        extracted[name] = row.get("_extracted") or extract_row(row)

    normalized: dict[str, dict[str, Optional[float]]] = {}
    for key, higher_better in METRIC_SPECS:
        normalized[key] = _normalize({name: values.get(key) for name, values in extracted.items()},
                                     higher_better)

    models = []
    for name, row in names.items():
        provider_id = row.get("provider_id")
        models.append({
            "provider_id": provider_id,
            "provider_name": row.get("provider_name") or "legacy",
            "model": row.get("model"),
            "status": row.get("status"),
            "checks": {check: (row.get("checks") or {}).get(check, {}).get("status")
                       for check in CHECK_NAMES},
            "score": _score(normalized, weights, name),
            "metrics": {
                key: {"value": extracted[name].get(key), "score": normalized[key].get(name)}
                for key, _ in METRIC_SPECS
            },
        })

    models.sort(key=lambda item: (
        item["score"] is None,
        -(item["score"] or 0),
        (item.get("provider_name") or "").lower(),
        (item.get("model") or "").lower(),
    ))
    providers = _provider_rollup(models)
    total = len(models)
    if top_n:
        models = models[:max(1, int(top_n))]

    return {
        "run_id": run_id,
        "scope": scope,
        "models": models,
        "providers": providers,
        "weights": weights,
        "total": total,
    }


def build_from_history(
    rows: list[dict],
    *,
    mode: str = "latest",
    window: int = 5,
    top_n: Optional[int] = None,
) -> dict:
    """Aggregate historical functional rows into one synthetic row per target."""
    grouped: dict[tuple[Optional[int], str], list[dict]] = {}
    for row in rows:
        if row.get("status") in ("pending", "running", "skipped", "error"):
            continue
        key = (row.get("provider_id"), row.get("model"))
        grouped.setdefault(key, []).append(row)

    synthetic: list[dict] = []
    for (provider_id, model), members in grouped.items():
        members.sort(key=lambda row: row.get("run_id") or 0, reverse=True)
        latest = members[0]
        if mode != "median":
            synthetic.append(latest)
            continue

        sample = members[:max(1, int(window))]
        values = [extract_row(row) for row in sample]
        aggregated: dict[str, Optional[float]] = {}
        for key, _ in METRIC_SPECS:
            present = [value[key] for value in values if value.get(key) is not None]
            aggregated[key] = round(statistics.median(present), 4) if present else None
        synthetic.append({
            "provider_id": provider_id,
            "provider_name": latest.get("provider_name") or "legacy",
            "model": model,
            "status": latest.get("status"),
            "checks": latest.get("checks") or {},
            "metrics": latest.get("metrics") or {},
            "_extracted": aggregated,
        })

    return build_comparison(synthetic, run_id=None, scope=f"history:{mode}", top_n=top_n)