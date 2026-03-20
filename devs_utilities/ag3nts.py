"""Helpers for building and submitting AG3NTS task answers."""

from __future__ import annotations

from typing import Any

from .http import RAW_TEXT, post_json


def build_task_answer_payload(
    api_key: str,
    task: str,
    answer: dict[str, Any],
) -> dict[str, Any]:
    """Build the standard AG3NTS verify payload."""

    return {
        "apikey": api_key,
        "task": task,
        "answer": answer,
    }


def submit_task_answer(
    url: str,
    *,
    api_key: str,
    task: str,
    answer: dict[str, Any],
    timeout_seconds: float = 30.0,
) -> Any:
    """Submit a task answer to an AG3NTS verify endpoint."""

    payload = build_task_answer_payload(api_key, task, answer)
    return post_json(
        url,
        payload,
        timeout_seconds=timeout_seconds,
        on_decode_error=RAW_TEXT,
    )
