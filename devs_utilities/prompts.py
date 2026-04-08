"""Helpers for loading task-specific prompt text files."""

from __future__ import annotations

from pathlib import Path


def resolve_prompt_path(source: str | Path, filename: str | Path) -> Path:
    """Resolve a prompt path relative to the calling module or directory."""

    source_path = Path(source).resolve()
    base_dir = source_path if source_path.is_dir() else source_path.parent
    prompt_path = Path(filename)
    if prompt_path.is_absolute():
        return prompt_path
    return (base_dir / prompt_path).resolve()


def load_prompt_text(source: str | Path, filename: str | Path) -> str:
    """Read a non-empty UTF-8 prompt file relative to a task module."""

    prompt_path = resolve_prompt_path(source, filename)
    try:
        prompt = prompt_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(f"Failed to read prompt file: {prompt_path}") from exc
    if not prompt:
        raise RuntimeError(f"Prompt file is empty: {prompt_path}")
    return prompt


__all__ = ["load_prompt_text", "resolve_prompt_path"]
