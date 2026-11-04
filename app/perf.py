"""Performance runners: concurrency (load) probe and context-scaling test."""
from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Optional

from . import db
from .providers import build_provider
from .targets import Target

OK_PROMPT = "Reply with exactly: OK"
CHARS_PER_TOKEN = 4

# Deterministic pseudo-random word stream: varied text avoids tokenizer
# compression artifacts that would make actual token counts far below target.
_FILLER_WORDS = (
    "alpha bravo cloud delta echo fjord gamma helix ivory juniper kestrel lumen meadow "
    "nimbus orbit prism quartz ridge sierra tundra ultra vector willow xylem yield zenith "
    "amber boulder cobalt drift ember flint granite harbor inlet jasper kelp lagoon marble "
    "north oasis peak quill river stone timber urban valley water xylo yonder zone blade "
    "canyon dune edge forest glade hill island jetty knoll leaf mist notch plain quake reef "
).split()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _percentile(sorted_values: list[int], p: float) -> Optional[int]:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * (p / 100.0)
    lower = int(k)
    upper = min(lower + 1, len(sorted_values) - 1)
    return int(round(sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (k - lower)))


def _filler(target_tokens: int) -> str:
    words: list[str] = []
    chars = 0
    seed = 123456789
    needed = target_tokens * CHARS_PER_TOKEN
    while chars < needed:
        seed = (seed * 1103515245 + 12345) % (2 ** 31)
        word = _FILLER_WORDS[seed % len(_FILLER_WORDS)]
        words.append(word)
        chars += len(word) + 1
    return " ".join(words)


async def _raw_request(provider, model: str, timeout_s: float) -> dict:
    """A single request with NO retry: rate limits (429) must be observed."""
    result = await provider.complete(
        model, [{"role": "user", "content": OK_PROMPT}],
        max_tokens=16, temperature=0, timeout_s=timeout_s,
    )
    return {
        "status": result.status_code or 0,
        "latency_ms": result.latency_ms or 0,
        "error": result.error,
    }


def _finalize_load(levels: list[dict]) -> dict:
    total_requests = sum(int(level["requests"]) for level in levels)
    total_429 = sum(int(level["http_429"]) for level in levels)
    throughputs = [level["throughput_rps"] for level in levels if level.get("throughput_rps") is not None]
    return {
        "levels": levels,
        "total_requests": total_requests,
        "total_429": total_429,
        "max_throughput_rps": max(throughputs) if throughputs else None,
    }


async def run_load(run_id: int, opts: dict, abort: asyncio.Event, targets: list[Target]) -> None:
    """Targets run sequentially so throughput measurements don't interfere."""
    levels: list[int] = opts["levels"]
    requests_per_level: int = opts["requests_per_level"]
    timeout_s: float = opts["timeout_s"]

    for target in targets:
        rm_id = target.run_model_id
        if rm_id is None:
            continue
        if abort.is_set():
            db.update_run_model(rm_id, status="skipped", outcome="aborted", finished_at=_now())
            continue
        db.update_run_model(rm_id, status="running", started_at=_now())
        model = target.model_name
        level_results: list[dict] = []
        provider = build_provider(target.provider)
        try:
            for level in levels:
                if abort.is_set():
                    break
                per_worker = [requests_per_level // level] * level
                for index in range(requests_per_level % level):
                    per_worker[index] += 1
                wall_start = time.perf_counter()

                async def worker(count: int) -> list[dict]:
                    return [await _raw_request(provider, model, timeout_s) for _ in range(count)]

                batches = await asyncio.gather(*(worker(per_worker[index]) for index in range(level)))
                wall_ms = int((time.perf_counter() - wall_start) * 1000)
                samples = [sample for batch in batches for sample in batch]
                success = sum(1 for sample in samples if sample["status"] == 200)
                http_429 = sum(1 for sample in samples if sample["status"] == 429)
                errors = len(samples) - success - http_429
                latencies = sorted(sample["latency_ms"] for sample in samples if sample["status"] == 200)
                level_results.append({
                    "concurrency": level,
                    "requests": len(samples),
                    "success": success,
                    "http_429": http_429,
                    "errors": errors,
                    "p50_ms": _percentile(latencies, 50),
                    "p95_ms": _percentile(latencies, 95),
                    "throughput_rps": round(success / (wall_ms / 1000.0), 2) if wall_ms else None,
                    "wall_ms": wall_ms,
                })
                db.update_run_model(rm_id, metrics={"levels": level_results})
        finally:
            await provider.aclose()
        db.update_run_model(rm_id, status="done", metrics=_finalize_load(level_results),
                            finished_at=_now())


def _finalize_context(levels: list[dict]) -> dict:
    ttfts = [level["ttft_ms"] for level in levels if level.get("ttft_ms") is not None]
    prefill = [level["prefill_tok_per_s"] for level in levels if level.get("prefill_tok_per_s") is not None]
    return {
        "levels": levels,
        "max_ttft_ms": max(ttfts) if ttfts else None,
        "max_prefill_tok_per_s": round(max(prefill)) if prefill else None,
    }


async def run_context(run_id: int, opts: dict, abort: asyncio.Event, targets: list[Target]) -> None:
    sizes: list[int] = opts["sizes"]
    timeout_s: float = opts["timeout_s"]

    for target in targets:
        rm_id = target.run_model_id
        if rm_id is None:
            continue
        if abort.is_set():
            db.update_run_model(rm_id, status="skipped", outcome="aborted", finished_at=_now())
            continue
        db.update_run_model(rm_id, status="running", started_at=_now())
        model = target.model_name
        level_results: list[dict] = []
        provider = build_provider(target.provider)
        try:
            for size in sizes:
                if abort.is_set():
                    break
                prompt = _filler(size) + "\n\nReply with exactly: OK"
                result = await provider.complete(
                    model, [{"role": "user", "content": prompt}],
                    max_tokens=16, temperature=0, stream=True, timeout_s=timeout_s,
                )
                latency = result.latency_ms or 0
                tokens_in = int((result.usage or {}).get("prompt_tokens") or 0)
                ttft = result.ttft_ms
                text = (result.text or "").lower()
                if result.status_code == 200:
                    status, error = "pass", None
                elif result.status_code == 400 and "context" in text:
                    status, error = "context_exceeded", result.text
                else:
                    status, error = "error", result.text or result.error
                prefill = (tokens_in / (ttft / 1000.0)) if ttft and tokens_in else None
                level_results.append({
                    "target_tokens": size,
                    "tokens_in": tokens_in or None,
                    "ttft_ms": ttft,
                    "latency_ms": latency,
                    "prefill_tok_per_s": round(prefill, 1) if prefill is not None else None,
                    "status": status,
                    "error": error,
                })
                db.update_run_model(rm_id, metrics={"levels": level_results})
        finally:
            await provider.aclose()
        db.update_run_model(rm_id, status="done", metrics=_finalize_context(level_results),
                            finished_at=_now())