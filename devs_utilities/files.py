"""File helpers shared across repository scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def resolve_path(path: Path | str, base_dir: Path) -> Path:
    """Resolve a path relative to the provided base directory."""

    normalized = path if isinstance(path, Path) else Path(path)
    if normalized.is_absolute():
        return normalized

    cwd_candidate = Path.cwd() / normalized
    if cwd_candidate.exists():
        return cwd_candidate

    return base_dir / normalized


def read_text_with_fallback(
    path: Path,
    *,
    primary_encoding: str = "utf-8",
    fallback_encoding: str = "utf-8-sig",
) -> str:
    """Read text, retrying with a fallback encoding when needed."""

    try:
        return path.read_text(encoding=primary_encoding)
    except UnicodeDecodeError:
        return path.read_text(encoding=fallback_encoding)


def write_json(
    path: Path,
    payload: Any,
    *,
    ensure_ascii: bool = False,
    indent: int = 2,
    trailing_newline: bool = True,
) -> None:
    """Write JSON to disk with consistent formatting."""

    serialized = json.dumps(payload, ensure_ascii=ensure_ascii, indent=indent)
    if trailing_newline:
        serialized += "\n"
    path.write_text(serialized, encoding="utf-8")
