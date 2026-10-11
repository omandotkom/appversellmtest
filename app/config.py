"""Runtime configuration sourced from .env at the project root."""
from __future__ import annotations

import threading
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

DEFAULTS: dict[str, str] = {
    "GATEWAY_URL": "http://localhost:4000",
    "GATEWAY_ADMIN_KEY": "",
    "GATEWAY_TEST_KEY": "",
    "HOST": "127.0.0.1",
    "PORT": "8000",
    "DB_PATH": "data/health.db",
    "FUNCTIONAL_CONCURRENCY": "4",
    "BENCH_CONCURRENCY": "2",
    "DEFAULT_TIMEOUT_S": "60",
    "BENCH_TOKEN_CAP": "150000",
    "LOAD_LEVELS": "1,2,4,8",
    "LOAD_REQUESTS_PER_LEVEL": "8",
    "CONTEXT_SIZES": "1000,8000,32000",
    "REGRESSION_THRESHOLD_PCT": "25",
}

SECRET_KEYS = ("GATEWAY_ADMIN_KEY", "GATEWAY_TEST_KEY")

_lock = threading.Lock()
_env: dict[str, str] = dict(DEFAULTS)


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def reload() -> None:
    parsed = _parse_env_file(ENV_PATH)
    with _lock:
        _env.clear()
        _env.update(DEFAULTS)
        _env.update(parsed)


def get(key: str, default: str = "") -> str:
    with _lock:
        return _env.get(key, default)


def get_int(key: str, default: int) -> int:
    try:
        return int(get(key, str(default)))
    except ValueError:
        return default


def get_list(key: str, default: list[int]) -> list[int]:
    """Parse a comma-separated list of integers (e.g. '1,2,4,8')."""
    raw = get(key, "")
    if not raw:
        return list(default)
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            values.append(int(part))
    return values or list(default)


def set_values(updates: dict[str, str]) -> None:
    """Persist values to .env and reload the in-memory config."""
    current = _parse_env_file(ENV_PATH)
    for key, value in updates.items():
        if value is None:
            continue
        current[key] = str(value)
    ordered = [k for k in DEFAULTS if k in current]
    ordered += [k for k in current if k not in DEFAULTS]
    lines = ["# Gateway Model Lab configuration", ""]
    lines += [f"{key}={current[key]}" for key in ordered]
    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    reload()


def db_path() -> Path:
    path = Path(get("DB_PATH", "data/health.db"))
    return path if path.is_absolute() else BASE_DIR / path


def ensure_dirs() -> None:
    (BASE_DIR / "data").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "data" / "bench").mkdir(parents=True, exist_ok=True)


reload()
