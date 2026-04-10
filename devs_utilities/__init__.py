"""Shared utilities for the AI devs repository.

The package exports selected helpers lazily to avoid import-time side effects.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS: dict[str, tuple[str, str]] = {
    "AG3NTS_ACCESS_LEVEL_URL": ("devs_utilities.ag3nts", "AG3NTS_ACCESS_LEVEL_URL"),
    "AG3NTS_API_BASE_URL": ("devs_utilities.ag3nts", "AG3NTS_API_BASE_URL"),
    "AG3NTS_BASE_URL": ("devs_utilities.ag3nts", "AG3NTS_BASE_URL"),
    "AG3NTS_LOCATION_URL": ("devs_utilities.ag3nts", "AG3NTS_LOCATION_URL"),
    "AG3NTS_PACKAGES_URL": ("devs_utilities.ag3nts", "AG3NTS_PACKAGES_URL"),
    "AG3NTS_PUBLIC_DATA_BASE_URL": ("devs_utilities.ag3nts", "AG3NTS_PUBLIC_DATA_BASE_URL"),
    "AG3NTS_RAILWAY_URL": ("devs_utilities.ag3nts", "AG3NTS_RAILWAY_URL"),
    "AG3NTS_SHELL_URL": ("devs_utilities.ag3nts", "AG3NTS_SHELL_URL"),
    "AG3NTS_TASK_DATA_BASE_URL": ("devs_utilities.ag3nts", "AG3NTS_TASK_DATA_BASE_URL"),
    "AG3NTS_TIMETRAVEL_PREVIEW_URL": (
        "devs_utilities.ag3nts",
        "AG3NTS_TIMETRAVEL_PREVIEW_URL",
    ),
    "AG3NTS_VERIFY_URL": ("devs_utilities.ag3nts", "AG3NTS_VERIFY_URL"),
    "AG3NTS_ZMAIL_URL": ("devs_utilities.ag3nts", "AG3NTS_ZMAIL_URL"),
    "ChatCompletionResult": ("devs_utilities.openrouter", "ChatCompletionResult"),
    "HttpRequestError": ("devs_utilities.http", "HttpRequestError"),
    "JsonResponseError": ("devs_utilities.http", "JsonResponseError"),
    "OpenRouterClient": ("devs_utilities.openrouter", "OpenRouterClient"),
    "OpenRouterConfig": ("devs_utilities.openrouter", "OpenRouterConfig"),
    "OpenRouterError": ("devs_utilities.openrouter", "OpenRouterError"),
    "ToolCall": ("devs_utilities.openrouter", "ToolCall"),
    "build_task_openrouter_client": ("devs_utilities.openrouter", "build_task_openrouter_client"),
    "build_task_site_name": ("devs_utilities.openrouter", "build_task_site_name"),
    "get_default_openrouter_site_url": ("devs_utilities.openrouter", "get_default_openrouter_site_url"),
    "build_ag3nts_api_url": ("devs_utilities.ag3nts", "build_ag3nts_api_url"),
    "build_ag3nts_public_data_url": ("devs_utilities.ag3nts", "build_ag3nts_public_data_url"),
    "build_ag3nts_task_data_url": ("devs_utilities.ag3nts", "build_ag3nts_task_data_url"),
    "build_ag3nts_url": ("devs_utilities.ag3nts", "build_ag3nts_url"),
    "build_task_answer_payload": ("devs_utilities.ag3nts", "build_task_answer_payload"),
    "configure_logging": ("devs_utilities.logging", "configure_logging"),
    "extract_completion_result": ("devs_utilities.openrouter", "extract_completion_result"),
    "extract_flag": ("devs_utilities.flags", "extract_flag"),
    "extract_text_content": ("devs_utilities.openrouter", "extract_text_content"),
    "extract_tool_calls": ("devs_utilities.openrouter", "extract_tool_calls"),
    "find_repo_root": ("devs_utilities.env", "find_repo_root"),
    "get_bytes": ("devs_utilities.http", "get_bytes"),
    "get_env": ("devs_utilities.env", "get_env"),
    "get_int_env": ("devs_utilities.env", "get_int_env"),
    "get_json": ("devs_utilities.http", "get_json"),
    "get_optional_env": ("devs_utilities.env", "get_optional_env"),
    "get_text": ("devs_utilities.http", "get_text"),
    "load_repo_env": ("devs_utilities.env", "load_repo_env"),
    "logger": ("devs_utilities.logging", "logger"),
    "parse_env_file": ("devs_utilities.env", "parse_env_file"),
    "post_json": ("devs_utilities.http", "post_json"),
    "read_text_with_fallback": ("devs_utilities.files", "read_text_with_fallback"),
    "resolve_path": ("devs_utilities.files", "resolve_path"),
    "submit_task_answer": ("devs_utilities.ag3nts", "submit_task_answer"),
    "write_json": ("devs_utilities.files", "write_json"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load exported helpers only when first accessed."""

    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose lazy exports in interactive completions."""

    return sorted({*globals(), *_EXPORTS})
