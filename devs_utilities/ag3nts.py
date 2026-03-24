"""Helpers for building and submitting AG3NTS task answers."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

from .http import RAW_TEXT, post_json


AG3NTS_BASE_URL = "https://example.invalid"
AG3NTS_VERIFY_URL = f"{AG3NTS_BASE_URL}/verify"
AG3NTS_LOCATION_URL = f"{AG3NTS_BASE_URL}/location"
AG3NTS_ACCESS_LEVEL_URL = f"{AG3NTS_BASE_URL}/accesslevel"
AG3NTS_TASK_DATA_BASE_URL = f"{AG3NTS_BASE_URL}/data"
AG3NTS_PUBLIC_DATA_BASE_URL = f"{AG3NTS_BASE_URL}/dane"
AG3NTS_SHELL_URL = f"{AG3NTS_BASE_URL}/api/shell"
AG3NTS_ZMAIL_URL = f"{AG3NTS_BASE_URL}/api/zmail"
AG3NTS_RAILWAY_URL = f"{AG3NTS_BASE_URL}/"
AG3NTS_PACKAGES_URL = f"{AG3NTS_BASE_URL}/packages"


def build_ag3nts_task_data_url(api_key: str, resource_name: str) -> str:
    """Build a task data URL under `/data/{api_key}/...`."""

    normalized_resource = resource_name.strip().lstrip("/")
    return (
        f"{AG3NTS_TASK_DATA_BASE_URL}/"
        f"{quote(api_key.strip(), safe='')}/"
        f"{normalized_resource}"
    )


def build_ag3nts_public_data_url(resource_path: str) -> str:
    """Build a public asset URL under `/dane/...`."""

    normalized_resource = resource_path.strip().lstrip("/")
    return f"{AG3NTS_PUBLIC_DATA_BASE_URL}/{normalized_resource}"


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
