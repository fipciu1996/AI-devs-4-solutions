"""Helpers for extracting task flags from nested responses."""

from __future__ import annotations

from typing import Any


def extract_flag(payload: Any) -> str | None:
    """Recursively search a nested payload for a `{FLG:...}` token."""

    if isinstance(payload, str):
        return payload if payload.startswith("{FLG:") else None

    if isinstance(payload, dict):
        for value in payload.values():
            flag = extract_flag(value)
            if flag:
                return flag
        return None

    if isinstance(payload, list):
        for item in payload:
            flag = extract_flag(item)
            if flag:
                return flag

    return None
