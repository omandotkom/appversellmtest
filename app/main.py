"""FastAPI app: dashboard UI + REST API + run orchestration."""
from __future__ import annotations

import asyncio
import re
import statistics
from collections import Counter
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from . import catalog, comparison, config, db, perf, runner, targets
from .agent import loop as agent_loop

# The metric used for per-target regression/trend detection, per run kind.
KIND_METRIC = {
    "functional": "avg_latency_ms",
    "bench": "error_rate",
    "load": "max_throughput_rps",
    "context": "max_prefill_tok_per_s",
}
# Kinds where a HIGHER metric is worse (latency, error rate).
KIND_WORSE_WHEN_UP = {"functional", "bench"}

GLOBAL_SETTING_KEYS = (
    "HOST", "PORT", "FUNCTIONAL_CONCURRENCY", "BENCH_CONCURRENCY",
    "DEFAULT_TIMEOUT_S", "BENCH_TOKEN_CAP", "LOAD_LEVELS",
    "LOAD_REQUESTS_PER_LEVEL", "CONTEXT_SIZES", "REGRESSION_THRESHOLD_PCT",
    "COMPARISON_WINDOW", "COMPARISON_REPEAT", "SCORE_W_TOKPS", "SCORE_W_TTFT",
    "SCORE_W_PASS", "SCORE_W_ITL", "SCORE_W_LATENCY", "SCORE_W_COST",
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
TASKS_DIR = Path(__file__).resolve().parent / "tasks"


@asynccontextmanager
async def lifespan(_: FastAPI):
    config.ensure_dirs()
    db.init()
    db.mark_interrupted()
    yield


app = FastAPI(title="Gateway Model Lab", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

ACTIVE_RUNS: dict[int, dict] = {}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")[:80] or "model"


# --- UI --------------------------------------------------------------------

@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico")
def favicon() -> Response:
    return Response(status_code=204)


# --- status / settings -----------------------------------------------------

@app.get("/api/status")
def api_status() -> dict:
    providers = db.list_providers()
    enabled = [provider for provider in providers if provider["enabled"]]
    return {
        "providers": len(providers),
        "providers_enabled": len(enabled),
        "models": db.count_enabled_models(),
        "configured": bool(enabled),
        "comparison_window": config.get_int("COMPARISON_WINDOW", 5),
        "comparison_repeat": config.get_int("COMPARISON_REPEAT", 3),
    }


@app.get("/api/settings")
def get_settings() -> dict:
    out = {key: config.get(key) for key in GLOBAL_SETTING_KEYS}
    out["score_weights"] = config.get_weights()
    return out


@app.post("/api/settings")
async def save_settings(body: dict) -> dict:
    updates: dict[str, str] = {}
    for key in GLOBAL_SETTING_KEYS:
        if body.get(key) not in (None, ""):
            updates[key] = str(body[key]).strip()
    config.set_values(updates)
    return {"ok": True}


# --- providers -------------------------------------------------------------

def _provider_or_404(provider_id: int) -> dict:
    row = db.get_provider(provider_id)
    if not row:
        raise HTTPException(404, "Provider tidak ditemukan")
    return row


@app.get("/api/providers")
def list_providers() -> dict:
    return {"providers": catalog.list_providers()}


@app.post("/api/providers")
def create_provider(body: dict) -> dict:
    try:
        provider_id = catalog.create_provider(body)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"provider": catalog.get_provider(provider_id)}


@app.get("/api/providers/{provider_id}")
def get_provider(provider_id: int) -> dict:
    _provider_or_404(provider_id)
    return {"provider": catalog.get_provider(provider_id)}


@app.patch("/api/providers/{provider_id}")
def update_provider(provider_id: int, body: dict) -> dict:
    _provider_or_404(provider_id)
    try:
        catalog.update_provider(provider_id, body)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"provider": catalog.get_provider(provider_id)}


@app.delete("/api/providers/{provider_id}")
def delete_provider(provider_id: int) -> dict:
    _provider_or_404(provider_id)
    db.set_provider_enabled(provider_id, False)
    return {"ok": True, "provider": catalog.get_provider(provider_id)}


@app.put("/api/providers/{provider_id}/secrets")
def update_provider_secrets(provider_id: int, body: dict) -> dict:
    _provider_or_404(provider_id)
    try:
        catalog.set_secrets(provider_id, body)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    row = db.get_provider(provider_id)
    return {
        "ok": True,
        "has_api_key": bool(row.get("api_key")),
        "has_admin_key": bool(row.get("admin_key")),
    }


@app.post("/api/providers/{provider_id}/test")
async def test_provider(provider_id: int) -> dict:
    _provider_or_404(provider_id)
    return await catalog.test_provider(provider_id)


@app.post("/api/providers/{provider_id}/refresh-models")
async def refresh_provider_models(provider_id: int) -> dict:
    _provider_or_404(provider_id)
    try:
        return await catalog.refresh_models(provider_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:  # noqa: BLE001 - network/provider failure
        raise HTTPException(502, f"Gagal mengambil model: {type(exc).__name__}: {exc}")


@app.get("/api/providers/{provider_id}/models")
def provider_models(provider_id: int) -> dict:
    _provider_or_404(provider_id)
    return {"models": db.get_provider_models(provider_id)}


@app.patch("/api/provider-models/{model_id}")
def update_provider_model(model_id: int, body: dict) -> dict:
    row = db.get_provider_model(model_id)
    if not row:
        raise HTTPException(404, "Model tidak ditemukan")
    allowed = {"display_name", "enabled", "mode", "blocked", "supports_tools",
               "supports_json", "context_window", "input_cost_per_token",
               "output_cost_per_token"}
    fields = {key: body[key] for key in body if key in allowed}
    db.update_provider_model(model_id, **fields)
    return {"model": db.get_provider_model(model_id)}


@app.get("/api/catalog")
def api_catalog() -> dict:
    return catalog.catalog()


# --- targets ---------------------------------------------------------------

def _prepare_targets(body: dict, *, include_blocked: bool = False) -> list[targets.Target]:
    try:
        resolved = targets.resolve_targets(
            body.get("targets"),
            body.get("models"),
            include_blocked=include_blocked,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    if not resolved:
        raise HTTPException(400, "Tidak ada model yang dipilih")
    return resolved


def _attach_run_models(run_id: int, resolved: list[targets.Target], *, bench: bool = False) -> list[dict]:
    snapshots = []
    for target in resolved:
        workspace = None
        if bench:
            workspace = f"data/bench/{run_id}/p{target.provider_id}/{_safe_name(target.model_name)}"
        target.run_model_id = db.create_run_model(
            run_id,
            target.provider_id,
            target.provider_name,
            target.model_name,
            workspace=workspace,
        )
        snapshots.append(target.as_dict())
    return snapshots


def _run_options(resolved: list[targets.Target], **extra: Any) -> dict:
    options = {key: value for key, value in extra.items() if value is not None}
    options["targets"] = [target.as_dict() for target in resolved]
    return options


# --- tasks -----------------------------------------------------------------

def _load_task(task_id: str) -> dict:
    path = TASKS_DIR / f"{task_id}.yaml"
    if not path.exists():
        raise HTTPException(404, f"Task '{task_id}' tidak ditemukan")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise HTTPException(400, f"Task '{task_id}' tidak valid: {exc}")
    if not isinstance(data, dict) or not data.get("prompt"):
        raise HTTPException(400, f"Task '{task_id}' tidak valid (prompt wajib ada)")
    data.setdefault("id", task_id)
    return data


@app.get("/api/tasks")
def api_tasks() -> dict:
    tasks = []
    for path in sorted(TASKS_DIR.glob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        tasks.append({
            "id": data.get("id") or path.stem,
            "name": data.get("name") or path.stem,
            "max_turns": data.get("max_turns"),
            "tools": data.get("tools"),
        })
    return {"tasks": tasks}


# --- runs ------------------------------------------------------------------

@app.post("/api/runs/functional")
async def start_functional(body: dict) -> dict:
    checks = [c for c in (body.get("checks") or list(runner.CHECK_NAMES))
              if c in runner.CHECK_NAMES]
    if not checks:
        raise HTTPException(400, "Pilih minimal satu jenis tes")
    include_blocked = bool(body.get("include_blocked"))
    resolved = _prepare_targets(body, include_blocked=include_blocked)
    opts = _run_options(
        resolved,
        checks=checks,
        include_blocked=include_blocked,
        concurrency=config.get_int("FUNCTIONAL_CONCURRENCY", 4),
        timeout_s=config.get_int("DEFAULT_TIMEOUT_S", 60),
        repeat=max(1, int(body.get("repeat") or 1)),
        warmup=bool(body.get("warmup", True)),
    )
    run_id = db.create_run("functional", opts)
    snapshots = _attach_run_models(run_id, resolved)
    abort = asyncio.Event()
    task = asyncio.create_task(_run_functional(run_id, checks, opts, abort, resolved))
    ACTIVE_RUNS[run_id] = {"abort": abort, "task": task, "kind": "functional"}
    return {"run_id": run_id, "models": [s["model"] for s in snapshots], "targets": snapshots}


@app.post("/api/runs/comparison")
async def start_comparison(body: dict) -> dict:
    """Full functional run (4 checks, repeat, warmup) tailored for comparison."""
    resolved = _prepare_targets(body, include_blocked=bool(body.get("include_blocked")))
    repeat = max(1, int(body.get("repeat") or config.get_int("COMPARISON_REPEAT", 3)))
    opts = _run_options(
        resolved,
        checks=list(runner.CHECK_NAMES),
        include_blocked=bool(body.get("include_blocked")),
        concurrency=config.get_int("FUNCTIONAL_CONCURRENCY", 4),
        timeout_s=config.get_int("DEFAULT_TIMEOUT_S", 60),
        repeat=repeat,
        warmup=True,
        comparison=True,
    )
    run_id = db.create_run("functional", opts)
    snapshots = _attach_run_models(run_id, resolved)
    abort = asyncio.Event()
    task = asyncio.create_task(_run_functional(run_id, list(runner.CHECK_NAMES), opts, abort, resolved))
    ACTIVE_RUNS[run_id] = {"abort": abort, "task": task, "kind": "functional"}
    return {"run_id": run_id, "models": [s["model"] for s in snapshots], "targets": snapshots}


@app.post("/api/runs/bench")
async def start_bench(body: dict) -> dict:
    task_id = body.get("task_id")
    if not task_id:
        raise HTTPException(400, "task_id wajib diisi")
    task = _load_task(task_id)
    resolved = _prepare_targets(body, include_blocked=False)
    opts = _run_options(
        resolved,
        task_id=task_id,
        max_turns=int(body.get("max_turns") or task.get("max_turns") or 30),
        token_cap=int(body.get("token_cap") or config.get_int("BENCH_TOKEN_CAP", 150000)),
        timeout_s=config.get_int("DEFAULT_TIMEOUT_S", 60),
        concurrency=config.get_int("BENCH_CONCURRENCY", 2),
        stream_turns=bool(body.get("stream_turns", True)),
    )
    run_id = db.create_run("bench", opts)
    snapshots = _attach_run_models(run_id, resolved, bench=True)
    abort = asyncio.Event()
    task_obj = asyncio.create_task(_run_bench(run_id, task, opts, abort, resolved))
    ACTIVE_RUNS[run_id] = {"abort": abort, "task": task_obj, "kind": "bench"}
    return {"run_id": run_id, "models": [s["model"] for s in snapshots], "targets": snapshots}


@app.post("/api/runs/load")
async def start_load(body: dict) -> dict:
    resolved = _prepare_targets(body, include_blocked=False)
    levels = [int(x) for x in (body.get("levels") or []) if str(x).isdigit()] \
        or config.get_list("LOAD_LEVELS", [1, 2, 4, 8])
    requests_per_level = int(body.get("requests_per_level")
                             or config.get_int("LOAD_REQUESTS_PER_LEVEL", 8))
    opts = _run_options(
        resolved,
        levels=levels,
        requests_per_level=requests_per_level,
        timeout_s=config.get_int("DEFAULT_TIMEOUT_S", 60),
    )
    run_id = db.create_run("load", opts)
    snapshots = _attach_run_models(run_id, resolved)
    abort = asyncio.Event()
    task = asyncio.create_task(_run_load(run_id, opts, abort, resolved))
    ACTIVE_RUNS[run_id] = {"abort": abort, "task": task, "kind": "load"}
    return {"run_id": run_id, "models": [s["model"] for s in snapshots], "targets": snapshots}


@app.post("/api/runs/context")
async def start_context(body: dict) -> dict:
    resolved = _prepare_targets(body, include_blocked=False)
    sizes = [int(x) for x in (body.get("sizes") or []) if str(x).isdigit()] \
        or config.get_list("CONTEXT_SIZES", [1000, 8000, 32000])
    opts = _run_options(
        resolved,
        sizes=sizes,
        timeout_s=config.get_int("DEFAULT_TIMEOUT_S", 60),
    )
    run_id = db.create_run("context", opts)
    snapshots = _attach_run_models(run_id, resolved)
    abort = asyncio.Event()
    task = asyncio.create_task(_run_context(run_id, opts, abort, resolved))
    ACTIVE_RUNS[run_id] = {"abort": abort, "task": task, "kind": "context"}
    return {"run_id": run_id, "models": [s["model"] for s in snapshots], "targets": snapshots}


async def _run_load(run_id: int, opts: dict, abort: asyncio.Event,
                    resolved: list[targets.Target]) -> None:
    try:
        await perf.run_load(run_id, opts, abort, resolved)
        _finalize_run(run_id, "load", abort)
    except Exception as exc:  # noqa: BLE001
        db.finish_run(run_id, "error", {"error": f"{type(exc).__name__}: {exc}"})
    finally:
        ACTIVE_RUNS.pop(run_id, None)


async def _run_context(run_id: int, opts: dict, abort: asyncio.Event,
                       resolved: list[targets.Target]) -> None:
    try:
        await perf.run_context(run_id, opts, abort, resolved)
        _finalize_run(run_id, "context", abort)
    except Exception as exc:  # noqa: BLE001
        db.finish_run(run_id, "error", {"error": f"{type(exc).__name__}: {exc}"})
    finally:
        ACTIVE_RUNS.pop(run_id, None)


def _finalize_run(run_id: int, kind: str, abort: asyncio.Event) -> None:
    regressions = _detect_regressions(run_id, kind)
    summary = _summarize_counts(run_id, kind)
    summary["regressions"] = regressions
    db.finish_run(run_id, "aborted" if abort.is_set() else "done", summary)


def _summarize_counts(run_id: int, kind: str) -> dict:
    rows = db.get_run_models(run_id)
    counts: Counter = Counter(row["status"] for row in rows)
    return {"counts": dict(counts), "models": len(rows), "cost_usd": None}


async def _run_functional(run_id: int, checks: list[str], opts: dict,
                          abort: asyncio.Event, resolved: list[targets.Target]) -> None:
    try:
        await runner.run_functional(run_id, checks, opts, abort, resolved)
        summary = _summarize_functional(run_id)
        summary["regressions"] = _detect_regressions(run_id, "functional")
        db.finish_run(run_id, "aborted" if abort.is_set() else "done", summary)
    except Exception as exc:  # noqa: BLE001
        db.finish_run(run_id, "error", {"error": f"{type(exc).__name__}: {exc}"})
    finally:
        ACTIVE_RUNS.pop(run_id, None)


async def _run_bench(run_id: int, task: dict, opts: dict, abort: asyncio.Event,
                     resolved: list[targets.Target]) -> None:
    try:
        concurrency = max(1, int(opts["concurrency"]))
        global_semaphore = asyncio.Semaphore(concurrency)
        provider_sems: dict[int, asyncio.Semaphore] = {}
        for target in resolved:
            limit = int(target.provider.get("max_concurrency") or concurrency)
            provider_sems[target.provider_id] = asyncio.Semaphore(max(1, limit))

        async def one(target: targets.Target) -> None:
            rm_id = target.run_model_id
            if rm_id is None:
                return
            async with provider_sems[target.provider_id]:
                async with global_semaphore:
                    if abort.is_set():
                        db.update_run_model(rm_id, status="skipped", outcome="aborted",
                                            finished_at=_now())
                        return
                    db.update_run_model(rm_id, status="running", started_at=_now())
                    row = db.get_run_model(rm_id) or {}
                    workspace = (config.BASE_DIR / row["workspace"]).resolve() \
                        if row.get("workspace") else (config.BASE_DIR / "data" / "bench" / str(run_id)).resolve()
                    workspace.mkdir(parents=True, exist_ok=True)
                    try:
                        outcome, metrics = await agent_loop.run_model_task(
                            run_id=run_id, target=target, task=task,
                            workspace=workspace, opts=opts, abort=abort,
                        )
                        db.update_run_model(rm_id, status="done", outcome=outcome,
                                            metrics=metrics, finished_at=_now())
                    except Exception as exc:  # noqa: BLE001
                        db.update_run_model(rm_id, status="error",
                                            error=f"{type(exc).__name__}: {exc}",
                                            finished_at=_now())

        await asyncio.gather(*(one(target) for target in resolved))
        summary = _summarize_bench(run_id)
        summary["regressions"] = _detect_regressions(run_id, "bench")
        db.finish_run(run_id, "aborted" if abort.is_set() else "done", summary)
    except Exception as exc:  # noqa: BLE001
        db.finish_run(run_id, "error", {"error": f"{type(exc).__name__}: {exc}"})
    finally:
        ACTIVE_RUNS.pop(run_id, None)


def _summarize_functional(run_id: int) -> dict:
    rows = db.get_run_models(run_id)
    counts: Counter = Counter(row["status"] for row in rows)
    cost = sum((row.get("metrics") or {}).get("cost_usd") or 0 for row in rows)
    return {"counts": dict(counts), "models": len(rows), "cost_usd": cost or None}


def _summarize_bench(run_id: int) -> dict:
    rows = db.get_run_models(run_id)
    outcomes: Counter = Counter(row.get("outcome") or row["status"] for row in rows)
    rates = [row["metrics"]["error_rate"] for row in rows
             if row.get("metrics") and row.get("metrics", {}).get("error_rate") is not None
             and row["status"] == "done"]
    cost = sum((row.get("metrics") or {}).get("cost_usd") or 0 for row in rows)
    return {
        "outcomes": dict(outcomes),
        "models": len(rows),
        "avg_error_rate": (sum(rates) / len(rates)) if rates else None,
        "cost_usd": cost or None,
    }


def _median(values: list[float]) -> Optional[float]:
    return statistics.median(values) if values else None


def _detect_regressions(run_id: int, kind: str) -> int:
    """Flag targets that regressed vs the median of their prior 5 runs."""
    metric = KIND_METRIC.get(kind)
    if not metric:
        return 0
    threshold = config.get_int("REGRESSION_THRESHOLD_PCT", 25) / 100.0
    worse_when_up = kind in KIND_WORSE_WHEN_UP
    count = 0
    for row in db.get_run_models(run_id):
        metrics = row.get("metrics") or {}
        current = metrics.get(metric)
        if current is None:
            continue
        history = db.model_history(
            row.get("provider_id"), row["model"], kind, run_id, 5,
        )
        prevs = [h["metrics"].get(metric) for h in history
                 if h["metrics"].get(metric) is not None]
        baseline = _median(prevs)
        if not baseline:
            continue
        delta = (current - baseline) / abs(baseline)
        worse = (delta > threshold) if worse_when_up else (delta < -threshold)
        if worse:
            metrics["regression"] = {
                "metric": metric,
                "baseline": round(baseline, 3),
                "current": round(current, 3),
                "delta_pct": round(delta * 100, 1),
            }
            db.update_run_model(row["id"], metrics=metrics)
            count += 1
    return count


# --- comparison ------------------------------------------------------------

def _parse_provider_ids(raw: Optional[str]) -> Optional[list[int]]:
    if not raw:
        return None
    ids = [int(part) for part in raw.split(",") if part.strip().isdigit()]
    return ids or None


def _comparison_payload(
    *, run_id: Optional[int], provider_ids: Optional[str],
    window: int, mode: str, top_n: int,
) -> dict:
    ids = _parse_provider_ids(provider_ids)
    if run_id is not None:
        run = db.get_run(run_id)
        if not run:
            raise HTTPException(404, "Run tidak ditemukan")
        rows = db.get_run_models(run_id)
        if ids:
            rows = [row for row in rows if row.get("provider_id") in ids]
        return comparison.build_comparison(
            rows, run_id=run_id, scope="run", top_n=top_n,
        )
    rows = db.all_run_models_with_runs(["functional"], provider_ids=ids)
    return comparison.build_from_history(
        rows, mode=mode, window=window, top_n=top_n,
    )


@app.get("/api/comparison")
def api_comparison(
    run_id: Optional[int] = None,
    provider_ids: Optional[str] = None,
    window: int = 5,
    mode: str = "latest",
    top_n: int = 20,
) -> dict:
    return _comparison_payload(
        run_id=run_id, provider_ids=provider_ids,
        window=max(1, window), mode=mode, top_n=max(1, top_n),
    )


@app.get("/api/comparison/providers")
def api_comparison_providers(
    run_id: Optional[int] = None,
    provider_ids: Optional[str] = None,
    window: int = 5,
    mode: str = "latest",
) -> dict:
    payload = _comparison_payload(
        run_id=run_id, provider_ids=provider_ids,
        window=max(1, window), mode=mode, top_n=1000,
    )
    return {"providers": payload["providers"], "weights": payload["weights"]}


# --- history / runs --------------------------------------------------------

@app.get("/api/models/history")
def model_history_api(
    model: str,
    kind: str,
    provider_id: Optional[int] = None,
    limit: int = 20,
) -> dict:
    metric = KIND_METRIC.get(kind)
    rows = db.model_history(provider_id, model, kind, None, limit)
    points = [{"run_id": r["run_id"], "started_at": r["started_at"],
               "value": (r["metrics"] or {}).get(metric)} for r in rows]
    return {"provider_id": provider_id, "model": model, "kind": kind,
            "metric": metric, "points": points}


@app.post("/api/runs/{run_id}/abort")
def abort_run(run_id: int) -> dict:
    active = ACTIVE_RUNS.get(run_id)
    if not active:
        raise HTTPException(404, "Run tidak aktif")
    active["abort"].set()
    return {"ok": True}


@app.get("/api/runs")
def list_runs(kind: Optional[str] = None, limit: int = 100) -> dict:
    return {"runs": db.list_runs(kind, limit)}


@app.get("/api/runs/{run_id}")
def run_detail(run_id: int) -> dict:
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(404, "Run tidak ditemukan")
    return {"run": run, "models": db.get_run_models(run_id)}


@app.get("/api/runs/{run_id}/models/{rm_id}")
def model_detail(run_id: int, rm_id: int) -> dict:
    row = db.get_run_model(rm_id)
    if not row or row["run_id"] != run_id:
        raise HTTPException(404, "Hasil model tidak ditemukan")
    return {"model": row, "steps": db.get_steps(rm_id)}


# --- benchmark artifacts ---------------------------------------------------

def _workspace_of(row: dict) -> Path:
    return (config.BASE_DIR / row["workspace"]).resolve() if row.get("workspace") \
        else (config.BASE_DIR / "data" / "bench").resolve()


@app.get("/api/bench/{rm_id}/files")
def bench_files(rm_id: int) -> dict:
    row = db.get_run_model(rm_id)
    if not row:
        raise HTTPException(404, "Hasil model tidak ditemukan")
    workspace = _workspace_of(row)
    files = []
    if workspace.exists():
        for path in sorted(workspace.rglob("*")):
            if path.is_file():
                files.append({"path": path.relative_to(workspace).as_posix(),
                              "size": path.stat().st_size})
            if len(files) >= 500:
                break
    return {"files": files, "workspace": str(workspace)}


@app.get("/api/bench/{rm_id}/file")
def bench_file(rm_id: int, path: str) -> FileResponse:
    row = db.get_run_model(rm_id)
    if not row:
        raise HTTPException(404, "Hasil model tidak ditemukan")
    workspace = _workspace_of(row)
    target = (workspace / path).resolve()
    try:
        target.relative_to(workspace)
    except ValueError:
        raise HTTPException(400, "path di luar workspace")
    if not target.exists() or not target.is_file():
        raise HTTPException(404, "File tidak ditemukan")
    media = "text/html" if target.suffix.lower() in (".html", ".htm") else "text/plain"
    return FileResponse(
        target,
        media_type=f"{media}; charset=utf-8",
        headers={
            "Content-Security-Policy": "sandbox; default-src 'none'; style-src 'unsafe-inline'; img-src data:; font-src data:",
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.get("/api/health")
@app.get("/health")
def health() -> dict:
    return {"ok": True}