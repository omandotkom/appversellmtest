"""Functional checks: basic chat, streaming, tool calling, JSON mode.

Adds deep latency fields (tok/s, ITL, generation time), multi-trial sampling
with percentiles, and an optional warmup request. Each check receives a
provider adapter, so the same code runs against any registered protocol.
"""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from typing import Any, Optional

from . import db
from .providers import build_provider
from .providers.base import BaseProvider
from .targets import Target

CHECK_NAMES = ("basic", "streaming", "tools", "json")

OK_PROMPT = "Reply with exactly: OK"
TOOLS_PROMPT = "Call the get_weather tool to find the weather in Jakarta. Use the tool instead of answering from memory."
JSON_PROMPT = 'Return a JSON object with exactly one key "status" and value "ok".'
WEATHER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
            },
        },
    }
]


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _elapsed(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _usage(usage: dict) -> tuple[int, int]:
    return int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)


def _usage_details(usage: dict) -> tuple[Optional[int], Optional[int]]:
    """Return (cached_tokens, reasoning_tokens) when the provider reports them."""
    cached: Optional[int] = None
    reasoning: Optional[int] = None
    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    if isinstance(prompt_details, dict) and prompt_details.get("cached_tokens") is not None:
        cached = int(prompt_details["cached_tokens"])
    if isinstance(completion_details, dict) and completion_details.get("reasoning_tokens") is not None:
        reasoning = int(completion_details["reasoning_tokens"])
    return cached, reasoning


def _result(status: str, latency_ms: int, *, http: Optional[int] = None,
            tokens_in: int = 0, tokens_out: int = 0, **extra: Any) -> dict:
    out: dict[str, Any] = {"status": status, "latency_ms": latency_ms,
                           "tokens_in": tokens_in, "tokens_out": tokens_out}
    if http is not None:
        out["http_status"] = http
    for key, value in extra.items():
        if value is not None:
            out[key] = value
    return out


def _percentile(sorted_values: list[int], p: float) -> Optional[int]:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * (p / 100.0)
    lower = int(k)
    upper = min(lower + 1, len(sorted_values) - 1)
    return int(round(sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (k - lower)))


def _looks_non_chat(error: str) -> bool:
    lowered = (error or "").lower()
    return "not a chat model" in lowered or "embedding" in lowered or "completion model" in lowered


def _reject_or_fail(supported: Optional[bool], text: str, latency_ms: int,
                    http_status: int, feature: str) -> dict:
    if http_status == 400 and supported is not True:
        return _result("na", latency_ms, http=http_status, error=text,
                       note=f"provider menolak {feature}; metadata tidak menjamin dukungan")
    return _result("fail", latency_ms, http=http_status, error=text)


def _with_perf(usage: dict, latency_ms: int) -> dict:
    """Attach cached/reasoning tokens and end-to-end tok/s."""
    tokens_in, tokens_out = _usage(usage)
    cached, reasoning = _usage_details(usage)
    tok_per_s = (tokens_out / (latency_ms / 1000.0)) if latency_ms and tokens_out else None
    return {"cached_tokens": cached, "reasoning_tokens": reasoning,
            "tok_per_s": round(tok_per_s, 1) if tok_per_s is not None else None}


async def check_basic(provider: BaseProvider, model: str, timeout_s: float, meta: dict) -> dict:
    result = await provider.complete(
        model, [{"role": "user", "content": OK_PROMPT}],
        max_tokens=16, temperature=0, timeout_s=timeout_s,
    )
    latency_ms = result.latency_ms or 0
    if result.error:
        return _result("fail", latency_ms, error=result.error)
    if result.status_code != 200:
        text = result.text or "request failed"
        if _looks_non_chat(text):
            return _result("na", latency_ms, http=result.status_code, error=text, note="bukan model chat")
        return _result("fail", latency_ms, http=result.status_code, error=text)

    message = result.message or {}
    content = (message.get("content") or "").strip()
    reasoning = message.get("reasoning_content")
    reasoning = reasoning.strip() if isinstance(reasoning, str) else ""
    tokens_in, tokens_out = _usage(result.usage)
    perf = _with_perf(result.usage, latency_ms)
    if content:
        return _result("pass", latency_ms, http=200, tokens_in=tokens_in,
                       tokens_out=tokens_out, **perf)
    if reasoning:
        return _result("warn", latency_ms, http=200, tokens_in=tokens_in, tokens_out=tokens_out,
                       note="konten kosong, hanya reasoning (jawaban mungkin tidak selesai)", **perf)
    return _result("warn", latency_ms, http=200, tokens_in=tokens_in, tokens_out=tokens_out,
                   note="respons kosong", **perf)


async def check_streaming(provider: BaseProvider, model: str, timeout_s: float, meta: dict) -> dict:
    result = await provider.complete(
        model, [{"role": "user", "content": OK_PROMPT}],
        max_tokens=16, temperature=0, stream=True, timeout_s=timeout_s,
    )
    latency = result.latency_ms or 0
    status_code = result.status_code
    if status_code != 200:
        text = result.text or result.error or "stream error"
        if _looks_non_chat(text):
            return _result("na", latency, http=status_code, error=text, note="bukan model chat")
        return _result("fail", latency, http=status_code, error=text)
    if result.error and result.chunks == 0:
        return _result("fail", latency, error=result.error)

    tokens_in, tokens_out = _usage(result.usage)
    estimated = tokens_out == 0 and result.chunks > 0
    if estimated:
        tokens_out = result.chunks
    cached, reasoning = _usage_details(result.usage)

    ttft = result.ttft_ms
    generation = max(0, latency - ttft) if ttft is not None else None
    itl = (generation / max(1, tokens_out - 1)) if generation is not None and tokens_out > 1 else None
    tok_per_s = (tokens_out / (generation / 1000.0)) if generation and tokens_out else None

    extras = {
        "ttft_ms": ttft,
        "generation_ms": generation,
        "itl_ms": round(itl, 1) if itl is not None else None,
        "tok_per_s": round(tok_per_s, 1) if tok_per_s is not None else None,
        "cached_tokens": cached,
        "reasoning_tokens": reasoning,
    }
    note = "tokens_out diestimasi dari jumlah chunk" if estimated else None
    if result.chunks and result.done:
        return _result("pass", latency, http=status_code or 200, tokens_in=tokens_in,
                       tokens_out=tokens_out, note=note, **extras)
    if result.chunks:
        return _result("warn", latency, http=status_code or 200, tokens_in=tokens_in,
                       tokens_out=tokens_out,
                       note=note or "stream berakhir tanpa penanda [DONE]", **extras)
    return _result("fail", latency, http=status_code or 200, note="tidak ada chunk konten yang diterima")


async def check_tools(provider: BaseProvider, model: str, timeout_s: float, meta: dict) -> dict:
    result = await provider.complete(
        model, [{"role": "user", "content": TOOLS_PROMPT}],
        tools=WEATHER_TOOLS, max_tokens=256, temperature=0, timeout_s=timeout_s,
    )
    latency_ms = result.latency_ms or 0
    if result.error:
        return _result("fail", latency_ms, error=result.error)
    if result.status_code != 200:
        text = result.text or "request failed"
        return _reject_or_fail(meta.get("supports_tools"), text, latency_ms,
                               result.status_code, "tool calling")

    message = result.message or {}
    tokens_in, tokens_out = _usage(result.usage)
    perf = _with_perf(result.usage, latency_ms)
    tool_calls = message.get("tool_calls") or []
    if not tool_calls:
        content = (message.get("content") or "")[:160]
        return _result("warn", latency_ms, http=200, tokens_in=tokens_in, tokens_out=tokens_out,
                       note=f"model menjawab tanpa tool call: {content!r}", **perf)
    args = (tool_calls[0].get("function") or {}).get("arguments") or "{}"
    try:
        json.loads(args)
    except (json.JSONDecodeError, TypeError):
        return _result("warn", latency_ms, http=200, tokens_in=tokens_in, tokens_out=tokens_out,
                       note="tool call ada tetapi arguments bukan JSON valid", **perf)
    return _result("pass", latency_ms, http=200, tokens_in=tokens_in, tokens_out=tokens_out, **perf)


async def check_json(provider: BaseProvider, model: str, timeout_s: float, meta: dict) -> dict:
    result = await provider.complete(
        model, [{"role": "user", "content": JSON_PROMPT}],
        response_format={"type": "json_object"}, max_tokens=100, temperature=0,
        timeout_s=timeout_s,
    )
    latency_ms = result.latency_ms or 0
    if result.error:
        return _result("fail", latency_ms, error=result.error)
    if result.status_code != 200:
        text = result.text or "request failed"
        return _reject_or_fail(meta.get("supports_json"), text, latency_ms,
                               result.status_code, "JSON mode")

    message = result.message or {}
    tokens_in, tokens_out = _usage(result.usage)
    perf = _with_perf(result.usage, latency_ms)
    content = (message.get("content") or "").strip()
    if not content:
        return _result("warn", latency_ms, http=200, tokens_in=tokens_in, tokens_out=tokens_out,
                       note="konten kosong", **perf)
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return _result("fail", latency_ms, http=200, tokens_in=tokens_in, tokens_out=tokens_out,
                       error=f"respons bukan JSON valid: {content[:160]!r}")
    if not isinstance(parsed, (dict, list)):
        return _result("warn", latency_ms, http=200, tokens_in=tokens_in, tokens_out=tokens_out,
                       note="JSON valid tetapi bukan object/array", **perf)
    return _result("pass", latency_ms, http=200, tokens_in=tokens_in, tokens_out=tokens_out, **perf)


_CHECKS = {
    "basic": check_basic,
    "streaming": check_streaming,
    "tools": check_tools,
    "json": check_json,
}

_STATUS_ORDER = {"fail": 0, "warn": 1, "pass": 2, "na": 3, "skipped": 4}


def _summarize(results: dict[str, dict]) -> str:
    statuses = [result["status"] for result in results.values()]
    if not statuses:
        return "skipped"
    if "fail" in statuses:
        return "fail"
    if "warn" in statuses:
        return "warn"
    if all(status == "na" for status in statuses):
        return "na"
    return "pass"


def _aggregate_samples(samples: list[dict]) -> dict:
    if not samples:
        return _result("skipped", 0)
    status = _summarize(dict(enumerate(samples)))
    representative = min(samples, key=lambda sample: _STATUS_ORDER.get(sample["status"], 5))
    out = representative.copy()
    out["status"] = status
    out["tokens_in"] = sum(int(sample.get("tokens_in") or 0) for sample in samples)
    out["tokens_out"] = sum(int(sample.get("tokens_out") or 0) for sample in samples)
    cached = sum(int(sample.get("cached_tokens") or 0) for sample in samples)
    reasoning = sum(int(sample.get("reasoning_tokens") or 0) for sample in samples)
    if cached:
        out["cached_tokens"] = cached
    if reasoning:
        out["reasoning_tokens"] = reasoning
    latencies = sorted(sample["latency_ms"] for sample in samples
                       if sample.get("latency_ms") is not None)
    ttfts = sorted(sample["ttft_ms"] for sample in samples
                   if sample.get("ttft_ms") is not None)
    if latencies:
        out["latency_ms"] = int(sum(latencies) / len(latencies))
    if len(samples) > 1:
        out["samples"] = samples
        if latencies:
            out["p50_latency_ms"] = _percentile(latencies, 50)
            out["p95_latency_ms"] = _percentile(latencies, 95)
        if ttfts:
            out["p50_ttft_ms"] = _percentile(ttfts, 50)
            out["p95_ttft_ms"] = _percentile(ttfts, 95)
    return out


def _metrics(results: dict[str, dict], meta: dict) -> dict:
    latencies = [result["latency_ms"] for result in results.values()
                 if result.get("latency_ms") is not None and result["status"] != "na"]
    all_lats: list[int] = []
    for result in results.values():
        for sample in result.get("samples") or [result]:
            if sample.get("latency_ms") is not None and sample.get("status") != "na":
                all_lats.append(sample["latency_ms"])
    tokens_in = sum(int(result.get("tokens_in") or 0) for result in results.values())
    tokens_out = sum(int(result.get("tokens_out") or 0) for result in results.values())
    cached = sum(int(result.get("cached_tokens") or 0) for result in results.values())
    reasoning = sum(int(result.get("reasoning_tokens") or 0) for result in results.values())
    cost = None
    input_cost, output_cost = meta.get("input_cost_per_token"), meta.get("output_cost_per_token")
    if input_cost is not None and output_cost is not None:
        cost = tokens_in * input_cost + tokens_out * output_cost
    ordered = sorted(all_lats)
    return {
        "avg_latency_ms": int(sum(latencies) / len(latencies)) if latencies else None,
        "p50_latency_ms": _percentile(ordered, 50),
        "p95_latency_ms": _percentile(ordered, 95),
        "sample_count": len(all_lats),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cached_tokens": cached,
        "reasoning_tokens": reasoning,
        "cost_usd": cost,
    }


async def run_functional(
    run_id: int,
    checks: list[str],
    opts: dict,
    abort: asyncio.Event,
    targets: list[Target],
) -> None:
    concurrency = max(1, int(opts.get("concurrency") or 4))
    timeout_s = float(opts.get("timeout_s") or 60)
    repeat = max(1, int(opts.get("repeat") or 1))
    warmup = bool(opts.get("warmup", repeat > 1))
    semaphore = asyncio.Semaphore(concurrency)
    provider_sems: dict[int, asyncio.Semaphore] = {}
    for target in targets:
        limit = int(target.provider.get("max_concurrency") or concurrency)
        provider_sems[target.provider_id] = asyncio.Semaphore(max(1, limit))

    async def one(target: Target) -> None:
        rm_id = target.run_model_id
        if rm_id is None:
            return
        model = target.model_name
        async with provider_sems[target.provider_id], semaphore:
            if abort.is_set():
                db.update_run_model(rm_id, status="skipped", finished_at=_now())
                return
            db.update_run_model(rm_id, status="running", started_at=_now())
            results: dict[str, dict] = {}
            warmup_ms: Optional[int] = None
            provider = build_provider(target.provider)
            try:
                if warmup:
                    wstart = time.perf_counter()
                    try:
                        await provider.complete(
                            model, [{"role": "user", "content": OK_PROMPT}],
                            max_tokens=16, temperature=0, timeout_s=timeout_s,
                        )
                    except Exception:  # noqa: BLE001 - warmup must not fail a run
                        pass
                    warmup_ms = _elapsed(wstart)
                for name in checks:
                    if abort.is_set():
                        break
                    fn = _CHECKS[name]
                    samples: list[dict] = []
                    for _ in range(repeat):
                        if abort.is_set():
                            break
                        try:
                            samples.append(await fn(provider, model, timeout_s, target.model))
                        except Exception as exc:  # noqa: BLE001 - one check must not kill the run
                            samples.append(_result("fail", 0, error=f"{type(exc).__name__}: {exc}"))
                    results[name] = _aggregate_samples(samples)
                    db.update_run_model(rm_id, checks=results)
            finally:
                await provider.aclose()

            metrics = _metrics(results, target.model)
            if warmup_ms is not None:
                metrics["warmup_ms"] = warmup_ms
            db.update_run_model(
                rm_id,
                status=_summarize(results),
                checks=results,
                metrics=metrics,
                finished_at=_now(),
            )

    await asyncio.gather(*(one(target) for target in targets))