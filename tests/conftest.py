from __future__ import annotations

import pytest

from app import config, db


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Isolate every test in its own SQLite database."""
    path = tmp_path / "health.db"
    monkeypatch.setattr(config, "db_path", lambda: path)
    db.init()
    return path