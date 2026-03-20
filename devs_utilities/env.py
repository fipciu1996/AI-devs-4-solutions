"""Environment helpers shared across the repository."""

from __future__ import annotations

import os
from pathlib import Path


def parse_env_file(path: Path) -> dict[str, str]:
    """Parse a simple dotenv-style file into a mapping."""

    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def find_repo_root(start: Path) -> Path:
    """Locate the repository root by walking up until `.git` is found."""

    current = start.resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return current


def load_repo_env(start: str | Path, *, override: bool = False) -> dict[str, str]:
    """Load the repository `.env` file into the current process environment."""

    repo_root = find_repo_root(Path(start))
    env_values = parse_env_file(repo_root / ".env")
    for key, value in env_values.items():
        if override or key not in os.environ:
            os.environ[key] = value
    return env_values


def get_env(name: str, default: str = "") -> str:
    """Read a stripped environment variable with a default."""

    return (os.getenv(name) or default).strip()


def get_optional_env(name: str) -> str | None:
    """Read an optional stripped environment variable."""

    value = get_env(name)
    return value or None
