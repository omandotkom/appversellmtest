"""SQLite persistence for runs, per-model results and agent trace steps."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from typing import Any, Optional

from . import config

_write_lock = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    options TEXT,
    summary TEXT,
    started_at TEXT,
    finished_at TEXT
);
CREATE TABLE IF NOT EXISTS run_models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    model TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    outcome TEXT,
    checks TEXT,
    metrics TEXT,
    workspace TEXT,
    error TEXT,
    started_at TEXT,
    finished_at TEXT
);
CREATE TABLE IF NOT EXISTS steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    run_model_id INTEGER NOT NULL,
    turn INTEGER,
    type TEXT,
    tool TEXT,
    args TEXT,
    result TEXT,
    ok INTEGER,
    category TEXT,
    flags TEXT,
    latency_ms INTEGER,
    tokens_in INTEGER DEFAULT 0,
    tokens_out INTEGER DEFAULT 0,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_steps_run_model ON steps(run_model_id);
CREATE INDEX IF NOT EXISTS idx_run_models_run ON run_models(run_id);
"""

_JSON_FIELDS_RUN = ("options", "summary")
_JSON_FIELDS_RM = ("checks", "metrics")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    path = config.db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_columns(conn: sqlite3.Connection) -> None:
    """Idempotent additive migrations for older databases."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(steps)")}
    if "ttft_ms" not in cols:
        conn.execute("ALTER TABLE steps ADD COLUMN ttft_ms INTEGER")


def init() -> None:
    with _write_lock:
        conn = _connect()
        try:
            conn.executescript("PRAGMA journal_mode=WAL;" + _SCHEMA)
            _ensure_columns(conn)
            conn.commit()
        finally:
            conn.close()


def _query(sql: str, params: tuple = ()) -> list[dict]:
    conn = _connect()
    try:
        return [dict(row) for row in conn.execute(sql, params)]
    finally:
        conn.close()


def _write(sql: str, params: tuple = ()) -> int:
    with _write_lock:
        conn = _connect()
        try:
            cur = conn.execute(sql, params)
            conn.commit()
            return int(cur.lastrowid or 0)
        finally:
            conn.close()


def _load_json(row: dict, fields: tuple) -> dict:
    for field in fields:
        if row.get(field):
            try:
                row[field] = json.loads(row[field])
            except (json.JSONDecodeError, TypeError):
                pass
    return row


# --- runs -----------------------------------------------------------------

def create_run(kind: str, options: dict) -> int:
    return _write(
        "INSERT INTO runs (kind, status, options, started_at) VALUES (?,?,?,?)",
        (kind, "running", json.dumps(options), _now()),
    )


def finish_run(run_id: int, status: str, summary: dict) -> None:
    _write(
        "UPDATE runs SET status=?, summary=?, finished_at=? WHERE id=?",
        (status, json.dumps(summary), _now(), run_id),
    )


def mark_interrupted() -> None:
    _write("UPDATE runs SET status='interrupted', finished_at=? WHERE status='running'", (_now(),))


def get_run(run_id: int) -> Optional[dict]:
    rows = _query("SELECT * FROM runs WHERE id=?", (run_id,))
    return _load_json(rows[0], _JSON_FIELDS_RUN) if rows else None


def list_runs(kind: Optional[str] = None, limit: int = 100) -> list[dict]:
    if kind:
        rows = _query("SELECT * FROM runs WHERE kind=? ORDER BY id DESC LIMIT ?", (kind, limit))
    else:
        rows = _query("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,))
    return [_load_json(row, _JSON_FIELDS_RUN) for row in rows]


# --- run models -----------------------------------------------------------

_RM_FIELDS = {"status", "outcome", "checks", "metrics", "workspace", "error", "started_at", "finished_at"}


def create_run_model(run_id: int, model: str, workspace: Optional[str] = None) -> int:
    return _write(
        "INSERT INTO run_models (run_id, model, status, workspace) VALUES (?,?,?,?)",
        (run_id, model, "pending", workspace),
    )


def update_run_model(rm_id: int, **fields: Any) -> None:
    sets, params = [], []
    for key, value in fields.items():
        if key not in _RM_FIELDS:
            continue
        if key in _JSON_FIELDS_RM and not isinstance(value, str):
            value = json.dumps(value)
        sets.append(f"{key}=?")
        params.append(value)
    if not sets:
        return
    params.append(rm_id)
    _write(f"UPDATE run_models SET {', '.join(sets)} WHERE id=?", tuple(params))


def get_run_models(run_id: int) -> list[dict]:
    rows = _query("SELECT * FROM run_models WHERE run_id=? ORDER BY id", (run_id,))
    return [_load_json(row, _JSON_FIELDS_RM) for row in rows]


def get_run_model(rm_id: int) -> Optional[dict]:
    rows = _query("SELECT * FROM run_models WHERE id=?", (rm_id,))
    return _load_json(rows[0], _JSON_FIELDS_RM) if rows else None


# --- steps ----------------------------------------------------------------

def add_step(
    run_id: int,
    run_model_id: int,
    *,
    turn: Optional[int] = None,
    type_: str = "note",
    tool: Optional[str] = None,
    args: Optional[str] = None,
    result: Optional[str] = None,
    ok: Optional[bool] = None,
    category: Optional[str] = None,
    flags: Optional[str] = None,
    latency_ms: Optional[int] = None,
    ttft_ms: Optional[int] = None,
    tokens_in: int = 0,
    tokens_out: int = 0,
) -> int:
    return _write(
        """INSERT INTO steps
           (run_id, run_model_id, turn, type, tool, args, result, ok, category, flags,
            latency_ms, ttft_ms, tokens_in, tokens_out, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            run_id,
            run_model_id,
            turn,
            type_,
            tool,
            args,
            result,
            None if ok is None else (1 if ok else 0),
            category,
            flags,
            latency_ms,
            ttft_ms,
            tokens_in,
            tokens_out,
            _now(),
        ),
    )


def get_steps(run_model_id: int) -> list[dict]:
    return _query("SELECT * FROM steps WHERE run_model_id=? ORDER BY id", (run_model_id,))


def model_history(model: str, kind: str, exclude_run_id: Optional[int], limit: int) -> list[dict]:
    """Metrics for a model across prior finished runs of a given kind."""
    if exclude_run_id is None:
        sql = (
            "SELECT rm.run_id, rm.metrics, r.started_at AS started_at FROM run_models rm "
            "JOIN runs r ON rm.run_id = r.id "
            "WHERE rm.model = ? AND r.kind = ? AND r.status = 'done' "
            "ORDER BY rm.run_id DESC LIMIT ?"
        )
        params: tuple = (model, kind, limit)
    else:
        sql = (
            "SELECT rm.run_id, rm.metrics, r.started_at AS started_at FROM run_models rm "
            "JOIN runs r ON rm.run_id = r.id "
            "WHERE rm.model = ? AND r.kind = ? AND rm.run_id != ? AND r.status = 'done' "
            "ORDER BY rm.run_id DESC LIMIT ?"
        )
        params = (model, kind, exclude_run_id, limit)
    rows = _query(sql, params)
    out = []
    for row in rows:
        metrics = row["metrics"]
        if isinstance(metrics, str):
            try:
                metrics = json.loads(metrics)
            except (json.JSONDecodeError, TypeError):
                metrics = {}
        out.append({"run_id": row["run_id"], "started_at": row["started_at"], "metrics": metrics or {}})
    return out
