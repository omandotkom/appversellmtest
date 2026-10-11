"""FastAPI app: dashboard UI + REST API + run orchestration."""
from __future__ import annotations

import asyncio
import re
import statistics
import time
from collections import Counter
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import httpx
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from . import config, db, gateway, perf, runner
from .agent import loop as agent_loop

# The metric used for per-model regression/trend detection, per run kind.
KIND_METRIC = {
    "functional": "avg_latency_ms",
    "bench": "error_rate",
    "load": "max_throughput_rps",
    "context": "max_prefill_tok_per_s",
}
# Kinds where a HIGHER metric is worse (latency, error rate).
KIND_WORSE_WHEN_UP = {"functional", "bench"}

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
MODELS_CACHE: dict[str, Any] = {"ts": 0.0, "models": [], "error": None}


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
    return {
        "gateway_url": gateway.configured_url(),
        "admin_key_set": bool(config.get("GATEWAY_ADMIN_KEY")),
        "test_key_set": bool(config.get("GATEWAY_TEST_KEY")),
        "models_cached": len(MODELS_CACHE["models"]),
    }


@app.get("/api/settings")
def get_settings() -> dict:
    return {
        "GATEWAY_URL": config.get("GATEWAY_URL"),
        "HOST": config.get("HOST"),
        "PORT": config.get("PORT"),
        "FUNCTIONAL_CONCURRENCY": config.get("FUNCTIONAL_CONCURRENCY"),
        "BENCH_CONCURRENCY": config.get("BENCH_CONCURRENCY"),
        "DEFAULT_TIMEOUT_S": config.get("DEFAULT_TIMEOUT_S"),
        "BENCH_TOKEN_CAP": config.get("BENCH_TOKEN_CAP"),
        "admin_key_set": bool(config.get("GATEWAY_ADMIN_KEY")),
        "test_key_set": bool(config.get("GATEWAY_TEST_KEY")),
    }


@app.post("/api/settings")
async def save_settings(body: dict) -> dict:
    updates: dict[str, str] = {}
    for key in ("GATEWAY_URL", "HOST", "PORT", "FUNCTIONAL_CONCURRENCY",
                "BENCH_CONCURRENCY", "DEFAULT_TIMEOUT_S", "BENCH_TOKEN_CAP"):
        if body.get(key) not in (None, ""):
            updates[key] = str(body[key]).strip()
    if body.get("GATEWAY_ADMIN_KEY"):
        updates["GATEWAY_ADMIN_KEY"] = str(body["GATEWAY_ADMIN_KEY"]).strip()
    if body.get("GATEWAY_TEST_KEY"):
        updates["GATEWAY_TEST_KEY"] = str(body["GATEWAY_TEST_KEY"]).strip()
    config.set_values(updates)
    MODELS_CACHE["models"] = []
    MODELS_CACHE["ts"] = 0.0
    return {"ok": True}


# --- models ----------------------------------------------------------------

async def _refresh_models() -> list[dict]:
    try:
        models = await gateway.list_models()
        MODELS_CACHE.update(ts=time.time(), models=models, error=None)
        return models
    except gateway.GatewayNotConfigured as exc:
        MODELS_CACHE["error"] = str(exc)
        raise HTTPException(400, str(exc))
    except httpx.HTTPError as exc:
        MODELS_CACHE["error"] = str(exc)
        raise HTTPException(502, f"Gagal mengambil daftar model: {exc}")


@app.get("/api/models")
async def api_models(refresh: int = 0) -> dict:
    if refresh or not MODELS_CACHE["models"] or time.time() - MODELS_CACHE["ts"] > 300:
        await _refresh_models()
    return {"models": MODELS_CACHE["models"], "error": MODELS_CACHE["error"], "fetched_at": MODELS_CACHE["ts"]}


async def _resolve_models(names: Optional[list], include_blocked: bool) -> list[str]:
    try:
        models = await _refresh_models()
    except HTTPException:
        models = MODELS_CACHE["models"]
        if not models:
            raise
    if names:
        known = {m["name"] for m in models}
        return [n for n in names if n in known]
    return [m["name"] for m in models
            if gateway.is_chat_candidate(m) and (include_blocked or not m.get("blocked"))]


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
    checks = [c for c in (body.get("checks") or ["basic", "streaming", "tools", "json"]) if c in runner.CHECK_NAMES]
    if not checks:
        raise HTTPException(400, "Pilih minimal satu jenis tes")
    include_blocked = bool(body.get("include_blocked"))
    models = await _resolve_models(body.get("models"), include_blocked)
    if not models:
        raise HTTPException(400, "Tidak ada model yang dipilih")
    opts = {
        "checks": checks,
        "include_blocked": include_blocked,
        "concurrency": config.get_int("FUNCTIONAL_CONCURRENCY", 4),
        "timeout_s": config.get_int("DEFAULT_TIMEOUT_S", 60),
        "repeat": max(1, int(body.get("repeat") or 1)),
        "warmup": bool(body.get("warmup", True)),
    }
    run_id = db.create_run("functional", opts)
    for model in models:
        db.create_run_model(run_id, model)
    metas = {m["name"]: m for m in MODELS_CACHE["models"]}
    abort = asyncio.Event()
    task = asyncio.create_task(_run_functional(run_id, checks, opts, abort, metas))
    ACTIVE_RUNS[run_id] = {"abort": abort, "task": task, "kind": "functional"}
    return {"run_id": run_id, "models": models}


@app.post("/api/runs/bench")
async def start_bench(body: dict) -> dict:
    task_id = body.get("task_id")
    if not task_id:
        raise HTTPException(400, "task_id wajib diisi")
    task = _load_task(task_id)
    models = await _resolve_models(body.get("models"), include_blocked=False)
    if not models:
        raise HTTPException(400, "Tidak ada model yang dipilih")
    opts = {
        "task_id": task_id,
        "max_turns": int(body.get("max_turns") or task.get("max_turns") or 30),
        "token_cap": int(body.get("token_cap") or config.get_int("BENCH_TOKEN_CAP", 150000)),
        "timeout_s": config.get_int("DEFAULT_TIMEOUT_S", 60),
        "concurrency": config.get_int("BENCH_CONCURRENCY", 2),
        "stream_turns": bool(body.get("stream_turns", True)),
    }
    run_id = db.create_run("bench", opts)
    for model in models:
        db.create_run_model(run_id, model, workspace=f"data/bench/{run_id}/{_safe_name(model)}")
    metas = {m["name"]: m for m in MODELS_CACHE["models"]}
    abort = asyncio.Event()
    task_obj = asyncio.create_task(_run_bench(run_id, task, opts, abort, metas))
    ACTIVE_RUNS[run_id] = {"abort": abort, "task": task_obj, "kind": "bench"}
    return {"run_id": run_id, "models": models}


@app.post("/api/runs/load")
async def start_load(body: dict) -> dict:
    models = await _resolve_models(body.get("models"), include_blocked=False)
    if not models:
        raise HTTPException(400, "Tidak ada model yang dipilih")
    levels = [int(x) for x in (body.get("levels") or []) if str(x).isdigit()] or config.get_list("LOAD_LEVELS", [1, 2, 4, 8])
    requests_per_level = int(body.get("requests_per_level") or config.get_int("LOAD_REQUESTS_PER_LEVEL", 8))
    opts = {
        "levels": levels,
        "requests_per_level": requests_per_level,
        "timeout_s": config.get_int("DEFAULT_TIMEOUT_S", 60),
    }
    run_id = db.create_run("load", opts)
    for model in models:
        db.create_run_model(run_id, model)
    abort = asyncio.Event()
    task = asyncio.create_task(_run_load(run_id, opts, abort))
    ACTIVE_RUNS[run_id] = {"abort": abort, "task": task, "kind": "load"}
    return {"run_id": run_id, "models": models}


@app.post("/api/runs/context")
async def start_context(body: dict) -> dict:
    models = await _resolve_models(body.get("models"), include_blocked=False)
    if not models:
        raise HTTPException(400, "Tidak ada model yang dipilih")
    sizes = [int(x) for x in (body.get("sizes") or []) if str(x).isdigit()] or config.get_list("CONTEXT_SIZES", [1000, 8000, 32000])
    opts = {
        "sizes": sizes,
        "timeout_s": config.get_int("DEFAULT_TIMEOUT_S", 60),
    }
    run_id = db.create_run("context", opts)
    for model in models:
        db.create_run_model(run_id, model)
    abort = asyncio.Event()
    task = asyncio.create_task(_run_context(run_id, opts, abort))
    ACTIVE_RUNS[run_id] = {"abort": abort, "task": task, "kind": "context"}
    return {"run_id": run_id, "models": models}


async def _run_load(run_id: int, opts: dict, abort: asyncio.Event) -> None:
    try:
        await perf.run_load(run_id, opts, abort, {})
        _finalize_run(run_id, "load", abort)
    except Exception as exc:  # noqa: BLE001
        db.finish_run(run_id, "error", {"error": f"{type(exc).__name__}: {exc}"})
    finally:
        ACTIVE_RUNS.pop(run_id, None)


async def _run_context(run_id: int, opts: dict, abort: asyncio.Event) -> None:
    try:
        await perf.run_context(run_id, opts, abort, {})
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


async def _run_functional(run_id: int, checks: list[str], opts: dict, abort: asyncio.Event, metas: dict) -> None:
    try:
        await runner.run_functional(run_id, checks, opts, abort, metas)
        summary = _summarize_functional(run_id)
        summary["regressions"] = _detect_regressions(run_id, "functional")
        db.finish_run(run_id, "aborted" if abort.is_set() else "done", summary)
    except Exception as exc:  # noqa: BLE001
        db.finish_run(run_id, "error", {"error": f"{type(exc).__name__}: {exc}"})
    finally:
        ACTIVE_RUNS.pop(run_id, None)


async def _run_bench(run_id: int, task: dict, opts: dict, abort: asyncio.Event, metas: dict) -> None:
    try:
        semaphore = asyncio.Semaphore(max(1, int(opts["concurrency"])))

        async def one(row: dict) -> None:
            async with semaphore:
                if abort.is_set():
                    db.update_run_model(row["id"], status="skipped", outcome="aborted", finished_at=_now())
                    return
                db.update_run_model(row["id"], status="running", started_at=_now())
                workspace = (config.BASE_DIR / row["workspace"]).resolve()
                workspace.mkdir(parents=True, exist_ok=True)
                try:
                    outcome, metrics = await agent_loop.run_model_task(
                        run_id=run_id, rm_id=row["id"], model=row["model"], task=task,
                        workspace=workspace, opts=opts, abort=abort, meta=metas.get(row["model"], {}),
                    )
                    db.update_run_model(row["id"], status="done", outcome=outcome,
                                        metrics=metrics, finished_at=_now())
                except Exception as exc:  # noqa: BLE001
                    db.update_run_model(row["id"], status="error",
                                        error=f"{type(exc).__name__}: {exc}", finished_at=_now())

        rows = db.get_run_models(run_id)
        await asyncio.gather(*(one(row) for row in rows))
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
    """Flag models that regressed vs the median of their prior 5 runs."""
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
        history = db.model_history(row["model"], kind, run_id, 5)
        prevs = [h["metrics"].get(metric) for h in history if h["metrics"].get(metric) is not None]
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


@app.get("/api/models/history")
def model_history_api(model: str, kind: str, limit: int = 20) -> dict:
    metric = KIND_METRIC.get(kind)
    rows = db.model_history(model, kind, None, limit)
    points = [{"run_id": r["run_id"], "started_at": r["started_at"],
               "value": (r["metrics"] or {}).get(metric)} for r in rows]
    return {"model": model, "kind": kind, "metric": metric, "points": points}


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
    return (config.BASE_DIR / row["workspace"]).resolve() if row.get("workspace") else (config.BASE_DIR / "data" / "bench").resolve()


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
                files.append({"path": path.relative_to(workspace).as_posix(), "size": path.stat().st_size})
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
