"""SQLite persistence for providers, models, runs, results and agent steps."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from . import config

SCHEMA_VERSION = 2

_write_lock = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS providers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL DEFAULT 'openai',
    base_url TEXT NOT NULL,
    chat_path TEXT NOT NULL DEFAULT '/v1/chat/completions',
    models_path TEXT NOT NULL DEFAULT '/v1/models',
    admin_models_paths TEXT,
    auth_scheme TEXT NOT NULL DEFAULT 'bearer',
    extra_headers TEXT,
    api_key TEXT,
    admin_key TEXT,
    timeout_s INTEGER,
    max_concurrency INTEGER,
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT,
    updated_at TEXT
);
CREATE TABLE IF NOT EXISTS provider_models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider_id INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    display_name TEXT,
    mode TEXT,
    blocked INTEGER NOT NULL DEFAULT 0,
    supports_tools INTEGER,
    supports_json INTEGER,
    context_window INTEGER,
    input_cost_per_token REAL,
    output_cost_per_token REAL,
    meta TEXT,
    enabled INTEGER NOT NULL DEFAULT 1,
    last_seen_at TEXT,
    created_at TEXT,
    updated_at TEXT,
    UNIQUE(provider_id, model_name)
);
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
    provider_id INTEGER,
    provider_name TEXT,
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
    ttft_ms INTEGER,
    tokens_in INTEGER DEFAULT 0,
    tokens_out INTEGER DEFAULT 0,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_steps_run_model ON steps(run_model_id);
CREATE INDEX IF NOT EXISTS idx_run_models_run ON run_models(run_id);
CREATE INDEX IF NOT EXISTS idx_provider_models_provider ON provider_models(provider_id);
"""

_JSON_FIELDS_RUN = ("options", "summary")
_JSON_FIELDS_RM = ("checks", "metrics")
_JSON_PROVIDER_FIELDS = ("admin_models_paths", "extra_headers")
_JSON_MODEL_FIELDS = ("meta",)

UNSET = object()
_UNSET = UNSET


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    path = config.db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def _table_names(conn: sqlite3.Connection) -> set[str]:
    return {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )}


def _backup_db(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    backup = path.with_name(path.name + f".bak.{stamp}")
    path.replace(backup)
    for suffix in ("-wal", "-shm"):
        side = Path(str(path) + suffix)
        if side.exists():
            side.replace(Path(str(backup) + suffix))
    return backup


def init() -> None:
    """Create the schema, resetting an older database first.

    Resetting is intentional in the multi-provider overhaul: the old
    single-gateway database has a different shape. The previous file is kept as
    a timestamped ``.bak`` next to the new one.
    """
    path = config.db_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    needs_reset = False
    if path.exists() and path.stat().st_size > 0:
        conn = sqlite3.connect(str(path), timeout=5)
        try:
            version = int(conn.execute("PRAGMA user_version").fetchone()[0] or 0)
            needs_reset = version < SCHEMA_VERSION and bool(_table_names(conn))
        finally:
            conn.close()

    if needs_reset:
        _backup_db(path)

    with _write_lock:
        conn = _connect()
        try:
            conn.executescript("PRAGMA journal_mode=WAL;" + _SCHEMA)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
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


def _jsonify(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return value


def _as_int(value: Any) -> Any:
    return int(bool(value)) if isinstance(value, bool) else value


# --- providers -------------------------------------------------------------

_PROVIDER_FIELDS = {
    "name", "kind", "base_url", "chat_path", "models_path", "admin_models_paths",
    "auth_scheme", "extra_headers", "api_key", "admin_key", "timeout_s",
    "max_concurrency", "enabled",
}


def create_provider(data: dict) -> int:
    values = dict(data)
    values.setdefault("kind", "openai")
    values.setdefault("chat_path", "/v1/chat/completions")
    values.setdefault("models_path", "/v1/models")
    values.setdefault("auth_scheme", "bearer")
    values.setdefault("enabled", 1)
    values["created_at"] = _now()
    values["updated_at"] = values["created_at"]

    fields = [field for field in _PROVIDER_FIELDS if field in values]
    placeholders = ",".join("?" for _ in fields)
    params = []
    for field in fields:
        value = values[field]
        if field in _JSON_PROVIDER_FIELDS:
            value = _jsonify(value)
        params.append(_as_int(value))
    return _write(
        f"INSERT INTO providers ({', '.join(fields)}) VALUES ({placeholders})",
        tuple(params),
    )


def update_provider(provider_id: int, **fields: Any) -> None:
    sets, params = [], []
    for key, value in fields.items():
        if key not in _PROVIDER_FIELDS:
            continue
        if key in _JSON_PROVIDER_FIELDS:
            value = _jsonify(value)
        sets.append(f"{key}=?")
        params.append(_as_int(value))
    if not sets:
        return
    sets.append("updated_at=?")
    params.extend([_now(), provider_id])
    _write(f"UPDATE providers SET {', '.join(sets)} WHERE id=?", tuple(params))


def get_provider(provider_id: int) -> Optional[dict]:
    rows = _query("SELECT * FROM providers WHERE id=?", (provider_id,))
    return _load_json(rows[0], _JSON_PROVIDER_FIELDS) if rows else None


def list_providers(enabled_only: bool = False) -> list[dict]:
    sql = "SELECT * FROM providers"
    if enabled_only:
        sql += " WHERE enabled=1"
    sql += " ORDER BY name"
    return [_load_json(row, _JSON_PROVIDER_FIELDS) for row in _query(sql)]


def set_provider_secrets(
    provider_id: int,
    api_key: Any = _UNSET,
    admin_key: Any = _UNSET,
) -> None:
    fields: dict[str, Any] = {}
    if api_key is not _UNSET:
        fields["api_key"] = api_key
    if admin_key is not _UNSET:
        fields["admin_key"] = admin_key
    update_provider(provider_id, **fields)


def set_provider_enabled(provider_id: int, enabled: bool) -> None:
    update_provider(provider_id, enabled=enabled)


# --- provider models -------------------------------------------------------

_MODEL_FIELDS = {
    "model_name", "display_name", "mode", "blocked", "supports_tools",
    "supports_json", "context_window", "input_cost_per_token",
    "output_cost_per_token", "meta", "enabled", "last_seen_at",
}


def _model_name(model: dict) -> str:
    return str(model.get("model_name") or model.get("name") or "").strip()


def upsert_provider_model(provider_id: int, model: dict) -> int:
    """Insert or refresh a model from provider discovery.

    Manual overrides (display name, enabled, prices) survive a refresh: the
    discovery payload only fills values that are still empty.
    """
    name = _model_name(model)
    if not name:
        raise ValueError("model_name is required")
    existing_rows = _query(
        "SELECT * FROM provider_models WHERE provider_id=? AND model_name=?",
        (provider_id, name),
    )
    now = _now()
    meta = _jsonify(model.get("meta"))

    if existing_rows:
        existing = _load_json(existing_rows[0], _JSON_MODEL_FIELDS)
        fields: dict[str, Any] = {
            "mode": model.get("mode", existing.get("mode")),
            "blocked": _as_int(model.get("blocked", existing.get("blocked"))),
            "supports_tools": _as_int(model.get("supports_tools", existing.get("supports_tools"))),
            "supports_json": _as_int(model.get("supports_json", existing.get("supports_json"))),
            "context_window": model.get("context_window", existing.get("context_window")),
            "meta": meta if meta is not None else existing.get("meta"),
            "last_seen_at": now,
            "updated_at": now,
        }
        for cost_field in ("input_cost_per_token", "output_cost_per_token"):
            if existing.get(cost_field) is None and model.get(cost_field) is not None:
                fields[cost_field] = model[cost_field]
        if existing.get("display_name") is None and model.get("display_name") is not None:
            fields["display_name"] = model["display_name"]
        sets = ", ".join(f"{key}=?" for key in fields)
        _write(
            f"UPDATE provider_models SET {sets} WHERE id=?",
            tuple(fields.values()) + (existing["id"],),
        )
        return int(existing["id"])

    values = {
        "provider_id": provider_id,
        "model_name": name,
        "display_name": model.get("display_name"),
        "mode": model.get("mode"),
        "blocked": _as_int(model.get("blocked", False)),
        "supports_tools": _as_int(model.get("supports_tools")),
        "supports_json": _as_int(model.get("supports_json")),
        "context_window": model.get("context_window"),
        "input_cost_per_token": model.get("input_cost_per_token"),
        "output_cost_per_token": model.get("output_cost_per_token"),
        "meta": meta,
        "enabled": _as_int(model.get("enabled", True)),
        "last_seen_at": now,
        "created_at": now,
        "updated_at": now,
    }
    fields = list(values)
    return _write(
        f"INSERT INTO provider_models ({', '.join(fields)}) VALUES ({','.join('?' for _ in fields)})",
        tuple(values[field] for field in fields),
    )


def update_provider_model(model_id: int, **fields: Any) -> None:
    sets, params = [], []
    for key, value in fields.items():
        if key not in _MODEL_FIELDS:
            continue
        if key in _JSON_MODEL_FIELDS:
            value = _jsonify(value)
        sets.append(f"{key}=?")
        params.append(_as_int(value))
    if not sets:
        return
    sets.append("updated_at=?")
    params.extend([_now(), model_id])
    _write(f"UPDATE provider_models SET {', '.join(sets)} WHERE id=?", tuple(params))


def get_provider_model(model_id: int) -> Optional[dict]:
    rows = _query("SELECT * FROM provider_models WHERE id=?", (model_id,))
    return _load_json(rows[0], _JSON_MODEL_FIELDS) if rows else None


def get_provider_models(provider_id: int, enabled_only: bool = False) -> list[dict]:
    sql = "SELECT * FROM provider_models WHERE provider_id=?"
    if enabled_only:
        sql += " AND enabled=1"
    sql += " ORDER BY model_name"
    return [_load_json(row, _JSON_MODEL_FIELDS) for row in _query(sql, (provider_id,))]


def list_models_by_name(model_name: str) -> list[dict]:
    rows = _query(
        "SELECT pm.*, p.name AS provider_name, p.enabled AS provider_enabled "
        "FROM provider_models pm JOIN providers p ON p.id = pm.provider_id "
        "WHERE pm.model_name=? ORDER BY p.name",
        (model_name,),
    )
    return [_load_json(row, _JSON_MODEL_FIELDS) for row in rows]


def disable_missing_provider_models(provider_id: int, seen_names: list[str]) -> None:
    now = _now()
    if seen_names:
        placeholders = ",".join("?" for _ in seen_names)
        _write(
            "UPDATE provider_models SET enabled=0, updated_at=? "
            f"WHERE provider_id=? AND model_name NOT IN ({placeholders})",
            tuple([now, provider_id] + seen_names),
        )
    else:
        _write(
            "UPDATE provider_models SET enabled=0, updated_at=? WHERE provider_id=?",
            (now, provider_id),
        )


def count_enabled_models() -> int:
    rows = _query(
        "SELECT COUNT(*) AS n FROM provider_models pm "
        "JOIN providers p ON p.id = pm.provider_id "
        "WHERE pm.enabled=1 AND p.enabled=1"
    )
    return int(rows[0]["n"]) if rows else 0


# --- runs ------------------------------------------------------------------

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


# --- run models ------------------------------------------------------------

_RM_FIELDS = {
    "provider_id", "provider_name", "model", "status", "outcome", "checks",
    "metrics", "workspace", "error", "started_at", "finished_at",
}


def create_run_model(
    run_id: int,
    provider_id: Optional[int],
    provider_name: Optional[str],
    model: str,
    workspace: Optional[str] = None,
) -> int:
    return _write(
        "INSERT INTO run_models (run_id, provider_id, provider_name, model, status, workspace) "
        "VALUES (?,?,?,?,?,?)",
        (run_id, provider_id, provider_name, model, "pending", workspace),
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


# --- steps -----------------------------------------------------------------

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


def all_run_models_with_runs(
    kinds: Optional[list[str]] = None,
    provider_ids: Optional[list[int]] = None,
) -> list[dict]:
    clauses, params = [], []
    if kinds:
        clauses.append("r.kind IN (%s)" % ",".join("?" for _ in kinds))
        params.extend(kinds)
    if provider_ids:
        clauses.append("rm.provider_id IN (%s)" % ",".join("?" for _ in provider_ids))
        params.extend(provider_ids)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    rows = _query(
        "SELECT rm.*, r.kind, r.started_at AS run_started_at, r.status AS run_status "
        f"FROM run_models rm JOIN runs r ON rm.run_id = r.id{where} "
        "ORDER BY rm.run_id DESC, rm.id",
        tuple(params),
    )
    return [_load_json(row, _JSON_FIELDS_RM) for row in rows]


def model_history(
    provider_id: Optional[int],
    model: str,
    kind: str,
    exclude_run_id: Optional[int],
    limit: int,
) -> list[dict]:
    """Metrics for a target across prior finished runs of a given kind."""
    clauses = [
        "rm.model = ?",
        "r.kind = ?",
        "r.status = 'done'",
        "rm.status NOT IN ('pending', 'running', 'skipped', 'error')",
    ]
    params: list[Any] = [model, kind]
    if provider_id is None:
        clauses.append("rm.provider_id IS NULL")
    else:
        clauses.append("rm.provider_id = ?")
        params.append(provider_id)
    if exclude_run_id is not None:
        clauses.append("rm.run_id != ?")
        params.append(exclude_run_id)
    params.append(limit)
    rows = _query(
        "SELECT rm.run_id, rm.metrics, r.started_at AS started_at FROM run_models rm "
        "JOIN runs r ON rm.run_id = r.id "
        f"WHERE {' AND '.join(clauses)} ORDER BY rm.run_id DESC LIMIT ?",
        tuple(params),
    )
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