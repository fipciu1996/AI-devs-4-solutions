"""Compatibility helpers for repository-wide environment access."""

from __future__ import annotations

from .env import (
    find_repo_root,
    get_env,
    get_int_env,
    get_optional_env,
    load_repo_env,
    parse_env_file,
)


def _get_first_env(*names: str) -> str:
    """Read the first non-empty environment variable from a list of aliases."""

    for name in names:
        value = get_env(name)
        if value:
            return value
    return ""


def get_course_api_key() -> str:
    """Read the generic course-service API key."""

    return _get_first_env("COURSE_API_KEY", "AG3NTS_API_KEY")


def get_llm_api_key() -> str:
    """Read the generic LLM gateway API key."""

    return _get_first_env("LLM_API_KEY", "OPENROUTER_API_KEY")


def get_llm_base_url() -> str:
    """Read the generic LLM gateway base URL."""

    return _get_first_env("LLM_BASE_URL", "OPENROUTER_BASE_URL")


def get_llm_model(*names: str, default: str = "") -> str:
    """Read the repository-wide default LLM model, then optional task-specific fallbacks."""

    return _get_first_env("OPENROUTER_MODEL", "LLM_MODEL", *names) or default


def get_ngrok_auth_token() -> str:
    """Read the repository-wide ngrok auth token, with a legacy alias fallback."""

    return _get_first_env("NGROK_AUTH_TOKEN", "NGROK_AUTHTOKEN")


def get_package_service_key() -> str:
    """Read the generic package-service API key."""

    return _get_first_env("PACKAGE_SERVICE_KEY", "PACKAGES_API_KEY")


def get_package_service_url() -> str:
    """Read the generic package-service base URL."""

    return _get_first_env("PACKAGE_SERVICE_URL", "PACKAGES_API_URL")


__all__ = [
    "find_repo_root",
    "get_env",
    "get_int_env",
    "get_optional_env",
    "get_course_api_key",
    "get_llm_api_key",
    "get_llm_base_url",
    "get_llm_model",
    "get_ngrok_auth_token",
    "get_package_service_key",
    "get_package_service_url",
    "load_repo_env",
    "parse_env_file",
]
