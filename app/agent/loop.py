"""Minimal multi-turn coding-agent loop with per-tool-call error accounting.

Deliberately not built on an agent SDK: the goal is to observe raw tool-call
behavior (invalid JSON args, unknown tools, failed edits, recovery) exactly as
the gateway forwards it, with no SDK retries or argument healing in between.

Turns are streamed by default so per-turn TTFT is measurable, with automatic
fallback to non-streaming when a provider rejects streaming tool calls.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Optional

import httpx

from .. import db, gateway
from . import errors as hit_metrics
from . import grading
from .tools import TOOL_FNS, TOOL_SPECS, ToolResult

MAX_RESULT_CHARS_FOR_MODEL = 12000
MAX_RESULT_CHARS_STORED = 4000
MAX_TURNS_DEFAULT = 30
TURN_MAX_TOKENS = 4096
HTTP_RETRIES = 3
RETRYABLE_STATUS = (429, 500, 502, 503, 504)


def _elapsed(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def build_system_prompt(workspace: Path) -> str:
    return (
        "You are an autonomous coding agent working in an isolated workspace.\n"
        f"Workspace root: {workspace}\n"
        "Available tools: read_file, write_file, edit_file, list_files, grep.\n"
        "There is NO shell/command execution in this environment - do not attempt to run commands.\n"
        "Guidelines:\n"
        "- File paths are always relative to the workspace root.\n"
        "- Read a file before editing it; copy old_string exactly from the file contents.\n"
        "- Keep working until the requested task is fully done.\n"
        "- When finished, reply with a short summary and stop calling tools."
    )


def _available_specs(task: dict) -> list[dict]:
    names = task.get("tools")
    if not names:
        return list(TOOL_SPECS)
    return [spec for spec in TOOL_SPECS if spec["function"]["name"] in names]


def _costs(meta: dict) -> dict:
    return {"input": meta.get("input_cost_per_token"), "output": meta.get("output_cost_per_token")}


async def execute_tool_call(name: str, raw_args: Any, workspace: Path, available: set[str]) -> ToolResult:
    if name not in available:
        return ToolResult(
            False,
            f"Error: unknown tool '{name}'. Available tools: {', '.join(sorted(available))}.",
            "unknown_tool",
        )
    if isinstance(raw_args, dict):
        args = raw_args
    else:
        raw = raw_args if isinstance(raw_args, str) else str(raw_args)
        raw = raw.strip() or "{}"
        try:
            args = json.loads(raw)
        except json.JSONDecodeError as exc:
            return ToolResult(
                False,
                f"Error: your tool arguments are not valid JSON ({exc.msg}). "
                'Send a JSON object like {"path": "index.html"}.',
                "invalid_json_args",
            )
    if not isinstance(args, dict):
        return ToolResult(False, "Error: tool arguments must be a JSON object.", "schema_violation")
    fn = TOOL_FNS.get(name)
    if fn is None:
        return ToolResult(False, f"Error: tool '{name}' is not implemented", "unknown_tool")
    try:
        return await asyncio.to_thread(fn, workspace, args)
    except Exception as exc:  # noqa: BLE001 - a tool must never crash the loop
        return ToolResult(False, f"Error: tool '{name}' failed unexpectedly: {exc}", "tool_execution_error")


async def _post_turn(client: httpx.AsyncClient, payload: dict, timeout_s: float) -> dict:
    """Non-streaming turn with retries for transient gateway errors."""
    started = time.perf_counter()
    last_error: Optional[str] = None
    for attempt in range(HTTP_RETRIES):
        try:
            response = await client.post(
                "/v1/chat/completions",
                json=payload,
                headers=gateway.test_headers(),
                timeout=gateway.timeout(timeout_s),
            )
            if response.status_code in RETRYABLE_STATUS and attempt < HTTP_RETRIES - 1:
                await asyncio.sleep(2 * (attempt + 1))
                continue
            if response.status_code == 200:
                return {"status_code": 200, "body": response.json(), "latency_ms": _elapsed(started), "error": None}
            return {"status_code": response.status_code, "text": gateway.error_text(response),
                    "latency_ms": _elapsed(started), "error": None}
        except httpx.HTTPError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < HTTP_RETRIES - 1:
                await asyncio.sleep(2 * (attempt + 1))
    return {"status_code": None, "text": None, "latency_ms": _elapsed(started), "error": last_error}


async def _stream_once(client: httpx.AsyncClient, payload: dict, timeout_s: float) -> dict:
    """One streaming request; accumulates content and tool-call deltas."""
    started = time.perf_counter()
    out: dict[str, Any] = {
        "status_code": None, "text": None, "content": "", "tool_calls": [],
        "usage": {}, "ttft_ms": None, "error": None,
    }
    try:
        async with client.stream("POST", "/v1/chat/completions", json=payload,
                                 headers=gateway.test_headers(),
                                 timeout=gateway.timeout(timeout_s)) as response:
            out["status_code"] = response.status_code
            if response.status_code != 200:
                raw = await response.aread()
                out["text"] = raw.decode("utf-8", "replace")[:500]
                return out
            acc: dict[int, dict] = {}
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    out["usage"] = obj["usage"]
                try:
                    delta = obj["choices"][0].get("delta") or {}
                except (KeyError, IndexError, TypeError):
                    continue
                if delta.get("content"):
                    out["content"] += delta["content"]
                if (delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls")) and out["ttft_ms"] is None:
                    out["ttft_ms"] = _elapsed(started)
                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    slot = acc.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] += fn["name"]
                    if fn.get("arguments"):
                        slot["arguments"] += fn["arguments"]
            for idx in sorted(acc):
                slot = acc[idx]
                out["tool_calls"].append({
                    "id": slot["id"] or f"call_{idx}",
                    "type": "function",
                    "function": {"name": slot["name"], "arguments": slot["arguments"]},
                })
    except httpx.HTTPError as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


async def _stream_turn(client: httpx.AsyncClient, payload: dict, timeout_s: float) -> dict:
    """Streaming turn with retries; adds latency_ms."""
    started = time.perf_counter()
    out: dict[str, Any] = {}
    for attempt in range(HTTP_RETRIES):
        out = await _stream_once(client, payload, timeout_s)
        retryable = out["error"] is None and out["status_code"] in RETRYABLE_STATUS
        if retryable and attempt < HTTP_RETRIES - 1:
            await asyncio.sleep(2 * (attempt + 1))
            continue
        break
    out["latency_ms"] = _elapsed(started)
    return out


async def run_model_task(
    *,
    run_id: int,
    rm_id: int,
    model: str,
    task: dict,
    workspace: Path,
    opts: dict,
    abort: asyncio.Event,
    meta: dict,
) -> tuple[str, dict]:
    timeout_s = float(opts.get("timeout_s") or 60)
    max_turns = int(opts.get("max_turns") or MAX_TURNS_DEFAULT)
    token_cap = int(opts.get("token_cap") or 0)
    stream_turns = bool(opts.get("stream_turns", True))

    workspace.mkdir(parents=True, exist_ok=True)
    messages = [
        {"role": "system", "content": build_system_prompt(workspace)},
        {"role": "user", "content": task["prompt"]},
    ]
    tools = _available_specs(task)
    available_names = {spec["function"]["name"] for spec in tools}

    tokens_in = 0
    tokens_out = 0
    turns = 0
    outcome: Optional[str] = None
    signatures: list[tuple[tuple, bool]] = []
    stream_fallback = 0

    async with httpx.AsyncClient(base_url=gateway.base_url()) as client:
        while turns < max_turns:
            if abort.is_set():
                outcome = "aborted"
                break
            turns += 1

            if stream_turns:
                payload = gateway.chat_payload(model=model, messages=messages, tools=tools,
                                               max_tokens=TURN_MAX_TOKENS, stream=True,
                                               stream_options={"include_usage": True})
                turn = await _stream_turn(client, payload, timeout_s)
                if turn.get("status_code") is not None and turn["status_code"] != 200 \
                        and "stream" in (turn.get("text") or "").lower():
                    stream_fallback += 1
                    nonstream = gateway.chat_payload(model=model, messages=messages, tools=tools,
                                                     max_tokens=TURN_MAX_TOKENS)
                    turn = await _post_turn(client, nonstream, timeout_s)
            else:
                payload = gateway.chat_payload(model=model, messages=messages, tools=tools,
                                               max_tokens=TURN_MAX_TOKENS)
                turn = await _post_turn(client, payload, timeout_s)

            latency_ms = turn.get("latency_ms") or 0

            if turn.get("error"):
                db.add_step(run_id, rm_id, turn=turns, type_="error", ok=0, category="gateway_error",
                            result=turn["error"], latency_ms=latency_ms)
                outcome = "request_error"
                break
            if turn.get("status_code") is None or turn["status_code"] != 200:
                text = turn.get("text") or "request failed"
                db.add_step(run_id, rm_id, turn=turns, type_="error", ok=0, category="http_error",
                            result=f"HTTP {turn.get('status_code')}: {text}", latency_ms=latency_ms)
                outcome = "request_error"
                break

            if turn.get("body") is not None:
                try:
                    message = turn["body"]["choices"][0]["message"] or {}
                    usage = turn["body"].get("usage") or {}
                except (KeyError, IndexError, TypeError):
                    db.add_step(run_id, rm_id, turn=turns, type_="error", ok=0, category="http_error",
                                result="respons tidak valid")
                    outcome = "request_error"
                    break
                content = message.get("content") or ""
                if not isinstance(content, str):
                    content = str(content)
                tool_calls = message.get("tool_calls") or []
                ttft_ms = None
            else:
                content = turn.get("content") or ""
                tool_calls = turn.get("tool_calls") or []
                usage = turn.get("usage") or {}
                ttft_ms = turn.get("ttft_ms")

            tin = int(usage.get("prompt_tokens") or 0)
            tout = int(usage.get("completion_tokens") or 0)
            tokens_in += tin
            tokens_out += tout

            db.add_step(run_id, rm_id, turn=turns, type_="assistant",
                        result=content[:MAX_RESULT_CHARS_STORED], ok=1, latency_ms=latency_ms,
                        ttft_ms=ttft_ms, tokens_in=tin, tokens_out=tout)

            if token_cap and (tokens_in + tokens_out) >= token_cap:
                outcome = "token_cap"
                break
            if not tool_calls:
                outcome = None
                break

            assistant_calls = []
            for call in tool_calls:
                fn = call.get("function") or {}
                raw = fn.get("arguments")
                if not isinstance(raw, str):
                    raw = json.dumps(raw or {})
                assistant_calls.append({
                    "id": call.get("id") or f"call_{turns}_{fn.get('name', 'unknown')}",
                    "type": "function",
                    "function": {"name": fn.get("name") or "", "arguments": raw},
                })
            messages.append({"role": "assistant", "content": content, "tool_calls": assistant_calls})

            for call in assistant_calls:
                name = call["function"]["name"]
                raw_args = call["function"]["arguments"]
                db.add_step(run_id, rm_id, turn=turns, type_="tool_call", tool=name,
                            args=raw_args[:MAX_RESULT_CHARS_STORED], ok=1)
                tool_started = time.perf_counter()
                result = await execute_tool_call(name, raw_args, workspace, available_names)
                tool_latency = int((time.perf_counter() - tool_started) * 1000)

                flags = ""
                signatures.append(((name, raw_args), result.ok))
                if not result.ok and len(signatures) >= 3:
                    last3 = signatures[-3:]
                    if all(not ok and sig == (name, raw_args) for sig, ok in last3):
                        flags = "repeated_loop"

                output_for_model = result.output
                if flags:
                    output_for_model += (
                        "\n[system] You have repeated the same failing tool call multiple times. "
                        "Change your approach instead of retrying it unchanged."
                    )

                db.add_step(run_id, rm_id, turn=turns, type_="tool_result", tool=name,
                            result=result.output[:MAX_RESULT_CHARS_STORED], ok=1 if result.ok else 0,
                            category=result.category, flags=flags or None, latency_ms=tool_latency)
                messages.append({"role": "tool", "tool_call_id": call["id"],
                                 "content": output_for_model[:MAX_RESULT_CHARS_FOR_MODEL]})
        else:
            outcome = "max_turns"

    steps = db.get_steps(rm_id)
    metrics = hit_metrics.summarize_steps(steps, _costs(meta))
    metrics["turns"] = turns
    metrics["artifacts"] = grading.check_artifacts(workspace, task.get("artifacts") or [])
    if outcome is None:
        artifacts = metrics["artifacts"]
        outcome = "completed" if all(a["ok"] for a in artifacts) else "completed_incomplete"
    metrics["outcome"] = outcome
    metrics["stream_fallback"] = stream_fallback

    # Per-turn performance aggregates.
    assistant_steps = [s for s in steps if s.get("type") == "assistant"]
    tool_steps = [s for s in steps if s.get("type") == "tool_result"]
    ttfts = [s["ttft_ms"] for s in assistant_steps if s.get("ttft_ms") is not None]
    turn_lats = [s["latency_ms"] for s in assistant_steps if s.get("latency_ms") is not None]
    model_time_ms = sum(turn_lats)
    tool_time_ms = sum(s.get("latency_ms") or 0 for s in tool_steps)
    gen_ms = sum((s.get("latency_ms") or 0) - (s.get("ttft_ms") or 0)
                 for s in assistant_steps if s.get("ttft_ms") is not None)
    gen_tok_per_s = (tokens_out / (gen_ms / 1000.0)) if gen_ms and tokens_out else None
    metrics["avg_turn_ttft_ms"] = int(sum(ttfts) / len(ttfts)) if ttfts else None
    metrics["avg_turn_latency_ms"] = int(sum(turn_lats) / len(turn_lats)) if turn_lats else None
    metrics["max_turn_latency_ms"] = max(turn_lats) if turn_lats else None
    metrics["gen_tok_per_s"] = round(gen_tok_per_s, 1) if gen_tok_per_s is not None else None
    metrics["model_time_ms"] = model_time_ms
    metrics["tool_time_ms"] = tool_time_ms
    metrics["model_tool_ratio"] = round(model_time_ms / max(1, tool_time_ms), 2) if tool_time_ms else None

    return outcome, metrics
