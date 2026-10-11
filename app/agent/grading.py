"""Deterministic task grading: check that required artifacts exist."""
from __future__ import annotations

from pathlib import Path


def check_artifacts(workspace: Path, artifacts: list[dict]) -> list[dict]:
    results = []
    for artifact in artifacts or []:
        rel = artifact.get("path")
        if not rel:
            continue
        min_bytes = int(artifact.get("min_bytes") or 0)
        target = (workspace / rel)
        exists = target.exists() and target.is_file()
        size = target.stat().st_size if exists else 0
        results.append({
            "path": rel,
            "exists": exists,
            "size": size,
            "min_bytes": min_bytes,
            "ok": exists and size >= min_bytes,
        })
    return results
