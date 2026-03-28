"""Runtime settings loader for the proxy API server."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

REPO_ROOT_HINT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT_HINT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_HINT))

from devs_utilities.bootstrap import bootstrap_repo
from repo_env import (
    get_env,
    get_llm_api_key,
    get_llm_base_url,
    get_package_service_key,
    get_package_service_url,
    load_repo_env,
)


DEFAULT_SYSTEM_PROMPT_RESOURCE = "prompts/system_prompt.txt"
REPO_ROOT = bootstrap_repo(__file__)


def _normalize_api_path(raw_path: str | None) -> str:
    path = (raw_path or "/mcp").strip()
    if not path:
        return "/mcp"
    if not path.startswith("/"):
        path = f"/{path}"
    return path.rstrip("/") or "/"


def _parse_port(raw_port: str | None) -> int:
    if raw_port in (None, ""):
        return 18080
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise RuntimeError("API_PORT must be a valid integer.") from exc
    if not 1 <= port <= 65535:
        raise RuntimeError("API_PORT must be between 1 and 65535.")
    return port


def _parse_float(raw_value: str | None, env_name: str, default: float) -> float:
    if raw_value in (None, ""):
        return default
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{env_name} must be a valid number.") from exc
    if value <= 0:
        raise RuntimeError(f"{env_name} must be greater than zero.")
    return value


def _parse_int(
    raw_value: str | None,
    env_name: str,
    default: int,
    *,
    minimum: int = 0,
) -> int:
    if raw_value in (None, ""):
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{env_name} must be a valid integer.") from exc
    if value < minimum:
        raise RuntimeError(f"{env_name} must be at least {minimum}.")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    api_host: str
    api_port: int
    api_path: str
    conversation_log_file: Path
    openrouter_api_key: str
    openrouter_model: str
    openrouter_base_url: str
    openrouter_timeout_seconds: float
    openrouter_system_prompt: str
    openrouter_app_url: str | None
    openrouter_app_title: str | None
    max_context_messages: int
    max_tool_round_trips: int
    packages_api_key: str
    packages_api_url: str
    packages_timeout_seconds: float


def _load_system_prompt(prompt_file: str | None) -> str:
    if prompt_file:
        prompt_path = Path(prompt_file)
        try:
            content = prompt_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(
                f"Failed to read OPENROUTER_SYSTEM_PROMPT_FILE: {prompt_path}"
            ) from exc
    else:
        try:
            content = files("proxy_api_server").joinpath(
                DEFAULT_SYSTEM_PROMPT_RESOURCE
            ).read_text(encoding="utf-8")
        except OSError as exc:
            raise RuntimeError("Failed to read bundled system prompt file.") from exc

    normalized = content.strip()
    if not normalized:
        raise RuntimeError("System prompt file must not be empty.")
    return normalized


def load_settings() -> Settings:
    """Build runtime settings from environment variables."""

    load_repo_env(__file__)

    api_host = get_env("API_HOST", "0.0.0.0") or "0.0.0.0"
    api_port = _parse_port(get_env("API_PORT"))
    api_path = _normalize_api_path(get_env("API_PATH"))
    log_path = Path(
        (get_env("CONVERSATION_LOG_FILE", "conversation.log"))
        or "conversation.log"
    )

    openrouter_api_key = get_llm_api_key()
    if not openrouter_api_key:
        raise RuntimeError("Missing LLM_API_KEY.")

    openrouter_model = get_env("OPENROUTER_MODEL", "openai/gpt-4.1-mini")
    if not openrouter_model:
        raise RuntimeError("OPENROUTER_MODEL must not be empty.")

    openrouter_base_url = get_llm_base_url()
    if not openrouter_base_url:
        raise RuntimeError("LLM_BASE_URL must not be empty.")

    openrouter_timeout_seconds = _parse_float(
        get_env("OPENROUTER_TIMEOUT_SECONDS"),
        "OPENROUTER_TIMEOUT_SECONDS",
        30.0,
    )
    max_context_messages = _parse_int(
        get_env("MAX_CONTEXT_MESSAGES"),
        "MAX_CONTEXT_MESSAGES",
        20,
    )
    max_tool_round_trips = _parse_int(
        get_env("MAX_TOOL_ROUND_TRIPS"),
        "MAX_TOOL_ROUND_TRIPS",
        8,
        minimum=1,
    )

    prompt_file = get_env("OPENROUTER_SYSTEM_PROMPT_FILE") or None
    custom_system_prompt = get_env("OPENROUTER_SYSTEM_PROMPT")
    openrouter_system_prompt = _load_system_prompt(prompt_file)
    if custom_system_prompt:
        openrouter_system_prompt = (
            f"{openrouter_system_prompt}\n\nDodatkowe instrukcje:\n{custom_system_prompt}"
        )

    app_url = get_env("OPENROUTER_APP_URL") or None
    app_title = get_env("OPENROUTER_APP_TITLE") or None

    packages_api_key = get_package_service_key()
    if not packages_api_key:
        raise RuntimeError("Missing PACKAGE_SERVICE_KEY.")

    packages_api_url = get_package_service_url()
    if not packages_api_url:
        raise RuntimeError("PACKAGE_SERVICE_URL must not be empty.")

    packages_timeout_seconds = _parse_float(
        get_env("PACKAGES_TIMEOUT_SECONDS"),
        "PACKAGES_TIMEOUT_SECONDS",
        15.0,
    )

    return Settings(
        api_host=api_host,
        api_port=api_port,
        api_path=api_path,
        conversation_log_file=log_path,
        openrouter_api_key=openrouter_api_key,
        openrouter_model=openrouter_model,
        openrouter_base_url=openrouter_base_url,
        openrouter_timeout_seconds=openrouter_timeout_seconds,
        openrouter_system_prompt=openrouter_system_prompt,
        openrouter_app_url=app_url,
        openrouter_app_title=app_title,
        max_context_messages=max_context_messages,
        max_tool_round_trips=max_tool_round_trips,
        packages_api_key=packages_api_key,
        packages_api_url=packages_api_url,
        packages_timeout_seconds=packages_timeout_seconds,
    )
