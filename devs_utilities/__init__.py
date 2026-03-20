"""Shared utilities for the AI devs repository."""

from .ag3nts import build_task_answer_payload, submit_task_answer
from .env import (
    find_repo_root,
    get_env,
    get_optional_env,
    load_repo_env,
    parse_env_file,
)
from .files import read_text_with_fallback, resolve_path, write_json
from .flags import extract_flag
from .http import HttpRequestError, JsonResponseError, get_bytes, get_json, get_text, post_json
from .logging import configure_logging, logger
from .openrouter import (
    ChatCompletionResult,
    OpenRouterClient,
    OpenRouterConfig,
    OpenRouterError,
    ToolCall,
    extract_completion_result,
    extract_text_content,
    extract_tool_calls,
)

__all__ = [
    "ChatCompletionResult",
    "HttpRequestError",
    "JsonResponseError",
    "OpenRouterClient",
    "OpenRouterConfig",
    "OpenRouterError",
    "ToolCall",
    "build_task_answer_payload",
    "configure_logging",
    "extract_completion_result",
    "extract_flag",
    "extract_text_content",
    "extract_tool_calls",
    "find_repo_root",
    "get_bytes",
    "get_env",
    "get_json",
    "get_optional_env",
    "get_text",
    "load_repo_env",
    "logger",
    "parse_env_file",
    "post_json",
    "read_text_with_fallback",
    "resolve_path",
    "submit_task_answer",
    "write_json",
]
