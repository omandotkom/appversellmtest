"""Coding-agent file tools with workspace sandboxing.

Each tool returns a :class:`ToolResult` carrying an `ok` flag and, on failure, an
error `category` used to build the per-tool-call error taxonomy.
"""
from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

MAX_FILE_CHARS = 12000
MAX_LIST_ENTRIES = 300
MAX_GREP_MATCHES = 50
MAX_GREP_FILE_BYTES = 1_000_000
READ_DEFAULT_LINES = 400
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".next", ".cache"}


@dataclass
class ToolResult:
    ok: bool
    output: str
    category: Optional[str] = None


class _PathEscape(Exception):
    pass


def _resolve(root: Path, rel: Any) -> Path:
    if not isinstance(rel, str) or not rel.strip():
        raise ValueError("path is required")
    cleaned = rel.strip().replace("\\", "/")
    if cleaned.startswith("/") or re.match(r"^[A-Za-z]:", cleaned) or cleaned.startswith("../") or cleaned == "..":
        raise _PathEscape(f"path '{rel}' is outside the workspace")
    target = (root / cleaned).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError:
        raise _PathEscape(f"path '{rel}' is outside the workspace")
    return target


def _bad_arg(exc: Exception) -> ToolResult:
    if isinstance(exc, _PathEscape):
        return ToolResult(False, f"Error: {exc}", "path_escape")
    return ToolResult(False, f"Error: {exc}", "schema_violation")


def read_file(root: Path, args: dict) -> ToolResult:
    try:
        path = _resolve(root, args.get("path"))
    except (ValueError, _PathEscape) as exc:
        return _bad_arg(exc)
    if not path.exists():
        return ToolResult(False, f"Error: file not found: {args.get('path')}", "file_not_found")
    if path.is_dir():
        return ToolResult(False, f"Error: {args.get('path')} is a directory. Use list_files.", "tool_execution_error")
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    offset = args.get("offset") or 1
    limit = args.get("limit") or READ_DEFAULT_LINES
    if not isinstance(offset, int) or offset < 1:
        offset = 1
    if not isinstance(limit, int) or limit < 1:
        limit = READ_DEFAULT_LINES
    limit = min(limit, 2000)
    window = lines[offset - 1: offset - 1 + limit]
    if not window:
        return ToolResult(True, f"(no lines at offset {offset}; file has {len(lines)} lines)")
    numbered = "\n".join(f"{n:>5}| {line}" for n, line in enumerate(window, start=offset))
    truncated = len(numbered) > MAX_FILE_CHARS
    output = numbered[:MAX_FILE_CHARS]
    if truncated:
        output += "\n... [truncated]"
    return ToolResult(True, output)


def write_file(root: Path, args: dict) -> ToolResult:
    content = args.get("content")
    if not isinstance(content, str):
        return ToolResult(False, "Error: 'content' must be a string", "schema_violation")
    try:
        path = _resolve(root, args.get("path"))
    except (ValueError, _PathEscape) as exc:
        return _bad_arg(exc)
    if path.exists() and path.is_dir():
        return ToolResult(False, f"Error: {args.get('path')} is a directory", "tool_execution_error")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return ToolResult(True, f"Wrote {len(content)} chars to {args.get('path')}")


def edit_file(root: Path, args: dict) -> ToolResult:
    old = args.get("old_string")
    new = args.get("new_string")
    if not isinstance(old, str) or old == "":
        return ToolResult(False, "Error: 'old_string' must be a non-empty string", "schema_violation")
    if not isinstance(new, str):
        return ToolResult(False, "Error: 'new_string' must be a string", "schema_violation")
    if old == new:
        return ToolResult(False, "Error: old_string and new_string are identical", "schema_violation")
    try:
        path = _resolve(root, args.get("path"))
    except (ValueError, _PathEscape) as exc:
        return _bad_arg(exc)
    if not path.exists() or not path.is_file():
        return ToolResult(False, f"Error: file not found: {args.get('path')}", "file_not_found")
    content = path.read_text(encoding="utf-8", errors="replace")
    count = content.count(old)
    if count == 0:
        return ToolResult(
            False,
            f"Error: old_string was not found in {args.get('path')}. "
            "Read the file and copy the exact text, including whitespace.",
            "edit_not_found",
        )
    replace_all = bool(args.get("replace_all"))
    if count > 1 and not replace_all:
        return ToolResult(
            False,
            f"Error: old_string appears {count} times in {args.get('path')}. "
            "Add more surrounding context or set replace_all=true.",
            "edit_ambiguous",
        )
    updated = content.replace(old, new) if replace_all else content.replace(old, new, 1)
    path.write_text(updated, encoding="utf-8")
    return ToolResult(True, f"Edited {args.get('path')}: replaced {count if replace_all else 1} occurrence(s). "
                            f"File is now {len(updated)} chars.")


def list_files(root: Path, args: dict) -> ToolResult:
    try:
        base = _resolve(root, args.get("path") or ".")
    except (ValueError, _PathEscape) as exc:
        return _bad_arg(exc)
    if not base.exists():
        return ToolResult(False, f"Error: path not found: {args.get('path') or '.'}", "file_not_found")
    entries: list[str] = []
    if base.is_file():
        entries.append(base.relative_to(root.resolve()).as_posix())
    else:
        for path in sorted(base.rglob("*")):
            rel_parts = path.relative_to(base).parts
            if any(part in SKIP_DIRS for part in rel_parts):
                continue
            if len(entries) >= MAX_LIST_ENTRIES:
                entries.append("... [truncated]")
                break
            rel = path.relative_to(root.resolve()).as_posix()
            entries.append(rel + ("/" if path.is_dir() else ""))
    return ToolResult(True, "\n".join(entries) or "(empty directory)")


def grep(root: Path, args: dict) -> ToolResult:
    pattern = args.get("pattern")
    if not isinstance(pattern, str) or not pattern:
        return ToolResult(False, "Error: 'pattern' is required", "schema_violation")
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        return ToolResult(False, f"Error: invalid regex pattern: {exc}", "schema_violation")
    try:
        base = _resolve(root, args.get("path") or ".")
    except (ValueError, _PathEscape) as exc:
        return _bad_arg(exc)
    if not base.exists():
        return ToolResult(False, f"Error: path not found: {args.get('path') or '.'}", "file_not_found")
    include = args.get("include")
    if include is not None and not isinstance(include, str):
        return ToolResult(False, "Error: 'include' must be a glob string like '*.html'", "schema_violation")
    matches: list[str] = []
    root_res = root.resolve()
    candidates = [base] if base.is_file() else sorted(base.rglob("*"))
    for path in candidates:
        if len(matches) >= MAX_GREP_MATCHES:
            break
        if not path.is_file():
            continue
        try:
            if any(part in SKIP_DIRS for part in path.relative_to(root_res).parts):
                continue
        except ValueError:
            continue
        if include and not fnmatch.fnmatch(path.name, include):
            continue
        try:
            if path.stat().st_size > MAX_GREP_FILE_BYTES:
                continue
            raw = path.read_bytes()
        except OSError:
            continue
        if b"\x00" in raw[:4096]:
            continue
        text = raw.decode("utf-8", "replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                rel = path.relative_to(root_res).as_posix()
                matches.append(f"{rel}:{lineno}: {line.strip()[:200]}")
                if len(matches) >= MAX_GREP_MATCHES:
                    break
    if not matches:
        return ToolResult(True, f"No matches found for {pattern!r}")
    output = "\n".join(matches)
    if len(matches) >= MAX_GREP_MATCHES:
        output += "\n... [truncated]"
    return ToolResult(True, output)


TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file from the workspace. Returns numbered lines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to the workspace root."},
                    "offset": {"type": "integer", "description": "1-based line number to start reading from."},
                    "limit": {"type": "integer", "description": "Maximum number of lines to read (default 400)."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file with the given full content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to the workspace root."},
                    "content": {"type": "string", "description": "The complete file content."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace an exact old_string with new_string in a file (string replacement edit).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to the workspace root."},
                    "old_string": {"type": "string", "description": "Exact text to replace (must match exactly once)."},
                    "new_string": {"type": "string", "description": "Replacement text."},
                    "replace_all": {"type": "boolean", "description": "Replace every occurrence instead of just the first."},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and directories in the workspace (recursive, capped).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path relative to the workspace root (default: root)."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search file contents with a regular expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regular expression to search for."},
                    "path": {"type": "string", "description": "Directory or file path relative to the workspace root."},
                    "include": {"type": "string", "description": "Optional filename glob filter, e.g. '*.html'."},
                },
                "required": ["pattern"],
            },
        },
    },
]

TOOL_FNS: dict[str, Callable[[Path, dict], ToolResult]] = {
    "read_file": read_file,
    "write_file": write_file,
    "edit_file": edit_file,
    "list_files": list_files,
    "grep": grep,
}
