"""Helpers for building and submitting AG3NTS task answers."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import quote

from .env import get_env, load_repo_env
from .http import RAW_TEXT, post_json


load_repo_env(Path(__file__))
AG3NTS_BASE_URL = get_env("AG3NTS_BASE_URL").rstrip("/")
if not AG3NTS_BASE_URL:
    raise RuntimeError("Missing AG3NTS_BASE_URL in the repository .env file.")


def build_ag3nts_url(resource_path: str) -> str:
    """Build an absolute AG3NTS URL under the configured base host."""

    normalized_resource = resource_path.strip().lstrip("/")
    return f"{AG3NTS_BASE_URL}/{normalized_resource}"


def build_ag3nts_api_url(resource_path: str) -> str:
    """Build an AG3NTS API URL under `/api/...`."""

    normalized_resource = resource_path.strip().lstrip("/")
    return f"{AG3NTS_API_BASE_URL}/{normalized_resource}"


AG3NTS_API_BASE_URL = build_ag3nts_url("api")
AG3NTS_VERIFY_URL = build_ag3nts_url("verify")
AG3NTS_LOCATION_URL = build_ag3nts_url("location")
AG3NTS_ACCESS_LEVEL_URL = build_ag3nts_url("accesslevel")
AG3NTS_TASK_DATA_BASE_URL = build_ag3nts_url("data")
AG3NTS_PUBLIC_DATA_BASE_URL = build_ag3nts_url("dane")
AG3NTS_TIMETRAVEL_PREVIEW_URL = build_ag3nts_url("timetravel_preview")
AG3NTS_SHELL_URL = build_ag3nts_api_url("shell")
AG3NTS_ZMAIL_URL = build_ag3nts_api_url("zmail")
AG3NTS_RAILWAY_URL = AG3NTS_VERIFY_URL
AG3NTS_PACKAGES_URL = build_ag3nts_url("packages")


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
