"""Helpers for bootstrapping repository-local scripts."""

from __future__ import annotations

import sys
from pathlib import Path

from .env import find_repo_root, load_repo_env


def bootstrap_repo(
    start: str | Path,
    *,
    add_to_syspath: bool = True,
    load_env: bool = True,
    override_env: bool = False,
) -> Path:
    """Ensure the repo root is importable and optionally load the repo env."""

    repo_root = find_repo_root(Path(start))
    if add_to_syspath and str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if load_env:
        load_repo_env(repo_root, override=override_env)
    return repo_root
