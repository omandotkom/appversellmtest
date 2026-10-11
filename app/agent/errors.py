"""Aggregate per-tool-call metrics from agent trace steps."""
from __future__ import annotations

from collections import Counter
from typing import Optional

ERROR_CATEGORIES = [
    "invalid_json_args",
    "unknown_tool",
    "schema_violation",
    "path_escape",
    "file_not_found",
    "edit_not_found",
    "edit_ambiguous",
    "tool_execution_error",
    "command_not_found",
    "command_timeout",
    "gateway_error",
    "http_error",
]


def summarize_steps(steps: list[dict], costs: Optional[dict] = None) -> dict:
    tool_results = [s for s in steps if s.get("type") == "tool_result"]
    total = len(tool_results)
    errors = [s for s in tool_results if s.get("ok") == 0]

    by_category: Counter = Counter()
    for s in errors:
        by_category[s.get("category") or "unknown"] += 1

    by_tool: dict[str, dict] = {}
    for s in tool_results:
        name = s.get("tool") or "unknown"
        entry = by_tool.setdefault(name, {"calls": 0, "errors": 0})
        entry["calls"] += 1
        if s.get("ok") == 0:
            entry["errors"] += 1

    # An error call "recovers" if any later tool call in the same run succeeds.
    recovered = 0
    for index, s in enumerate(tool_results):
        if s.get("ok") == 0:
            if any(later.get("ok") == 1 for later in tool_results[index + 1:]):
                recovered += 1

    streak = 0
    max_streak = 0
    for s in tool_results:
        if s.get("ok") == 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    repeated = sum(1 for s in tool_results if "repeated_loop" in (s.get("flags") or ""))
    turns = sum(1 for s in steps if s.get("type") == "assistant")
    tokens_in = sum(int(s.get("tokens_in") or 0) for s in steps)
    tokens_out = sum(int(s.get("tokens_out") or 0) for s in steps)

    cost = None
    if costs:
        ic, oc = costs.get("input"), costs.get("output")
        if ic is not None and oc is not None:
            cost = tokens_in * ic + tokens_out * oc

    return {
        "tool_calls": total,
        "tool_errors": len(errors),
        "error_rate": (len(errors) / total) if total else 0.0,
        "by_category": dict(by_category),
        "by_tool": by_tool,
        "recovery_rate": (recovered / len(errors)) if errors else None,
        "max_error_streak": max_streak,
        "repeated_loop": repeated,
        "turns": turns,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_usd": cost,
    }
