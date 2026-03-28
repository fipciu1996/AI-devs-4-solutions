"""Compatibility shim for the shared repository environment helpers."""

from devs_utilities.env import (
    find_repo_root,
    get_env,
    get_int_env,
    get_optional_env,
    load_repo_env,
    parse_env_file,
)


def get_course_api_key() -> str:
    """Read the generic course-service API key."""

    return get_env("COURSE_API_KEY")


def get_llm_api_key() -> str:
    """Read the generic LLM gateway API key."""

    return get_env("LLM_API_KEY")


def get_llm_base_url() -> str:
    """Read the generic LLM gateway base URL."""

    return get_env("LLM_BASE_URL")


def get_package_service_key() -> str:
    """Read the generic package-service API key."""

    return get_env("PACKAGE_SERVICE_KEY")


def get_package_service_url() -> str:
    """Read the generic package-service base URL."""

    return get_env("PACKAGE_SERVICE_URL")


__all__ = [
    "find_repo_root",
    "get_env",
    "get_int_env",
    "get_optional_env",
    "get_course_api_key",
    "get_llm_api_key",
    "get_llm_base_url",
    "get_package_service_key",
    "get_package_service_url",
    "load_repo_env",
    "parse_env_file",
]
