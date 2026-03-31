"""Shared OpenRouter client and response parsing helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .env import get_optional_env
from .http import HttpRequestError, JsonResponseError, post_json


class OpenRouterError(RuntimeError):
    """Raised when OpenRouter returns an invalid or unsuccessful response."""


@dataclass(frozen=True, slots=True)
class OpenRouterConfig:
    api_key: str
    base_url: str
    model: str
    timeout_seconds: float = 120.0
    site_url: str | None = None
    site_name: str | None = None


def build_task_site_name(source_path: str | Path) -> str:
    """Build the standard X-Title header for a task script."""

    task_dir_name = Path(source_path).resolve().parent.name
    return f"AI Devs 4 - {task_dir_name}"


def get_default_openrouter_site_url() -> str | None:
    """Read the standard optional site URL headers for OpenRouter."""

    return get_optional_env("OPENROUTER_SITE_URL") or get_optional_env("OPENROUTER_APP_URL")


def build_task_openrouter_client(
    source_path: str | Path,
    *,
    api_key: str,
    base_url: str,
    model: str,
    timeout_seconds: float = 120.0,
    site_url: str | None = None,
    site_name: str | None = None,
) -> "OpenRouterClient":
    """Create a task-scoped OpenRouter client with standard headers."""

    return OpenRouterClient(
        OpenRouterConfig(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout_seconds=timeout_seconds,
            site_url=site_url if site_url is not None else get_default_openrouter_site_url(),
            site_name=site_name if site_name is not None else build_task_site_name(source_path),
        )
    )


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    def to_message_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }


@dataclass(frozen=True, slots=True)
class ChatCompletionResult:
    content: str | None
    tool_calls: list[ToolCall]


class OpenRouterClient:
    """Thin client for OpenRouter chat completions."""

    def __init__(self, config: OpenRouterConfig) -> None:
        self._config = config

    def create_raw_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        extra_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
        if extra_payload:
            payload.update(extra_payload)

        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
        }
        if self._config.site_url:
            headers["HTTP-Referer"] = self._config.site_url
        if self._config.site_name:
            headers["X-Title"] = self._config.site_name

        try:
            parsed = post_json(
                self._config.base_url,
                payload,
                headers=headers,
                timeout_seconds=self._config.timeout_seconds,
            )
        except HttpRequestError as exc:
            raise OpenRouterError(str(exc)) from exc
        except JsonResponseError as exc:
            raise OpenRouterError(
                f"OpenRouter returned invalid JSON for {exc.url}."
            ) from exc

        if not isinstance(parsed, dict):
            raise OpenRouterError("OpenRouter returned a non-object response.")
        if parsed.get("error"):
            raise OpenRouterError(str(parsed["error"]))
        return parsed

    def create_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        extra_payload: dict[str, Any] | None = None,
    ) -> ChatCompletionResult:
        return extract_completion_result(
            self.create_raw_completion(
                messages,
                tools=tools,
                tool_choice=tool_choice,
                extra_payload=extra_payload,
            )
        )


def extract_completion_result(payload: dict[str, Any]) -> ChatCompletionResult:
    """Extract the first assistant message from a chat completion response."""

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise OpenRouterError("OpenRouter response did not contain choices.")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise OpenRouterError("OpenRouter response choice has invalid format.")

    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise OpenRouterError("OpenRouter response did not contain a message.")

    content = extract_text_content(message.get("content"))
    tool_calls = extract_tool_calls(message.get("tool_calls"))
    if not content and not tool_calls:
        raise OpenRouterError("OpenRouter returned an empty response.")

    return ChatCompletionResult(content=content, tool_calls=tool_calls)


def extract_text_content(content: Any) -> str | None:
    """Extract normalized text from an OpenRouter content payload."""

    if content is None:
        return None

    if isinstance(content, str):
        normalized = content.strip()
        return normalized or None

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                stripped = item["text"].strip()
                if stripped:
                    parts.append(stripped)
        if parts:
            return "\n".join(parts)
        return None

    raise OpenRouterError("OpenRouter response content has unsupported format.")


def extract_tool_calls(raw_tool_calls: Any) -> list[ToolCall]:
    """Parse OpenRouter tool calls into strongly typed objects."""

    if raw_tool_calls is None:
        return []
    if not isinstance(raw_tool_calls, list):
        raise OpenRouterError("OpenRouter tool_calls field has invalid format.")

    tool_calls: list[ToolCall] = []
    for raw_tool_call in raw_tool_calls:
        if not isinstance(raw_tool_call, dict):
            raise OpenRouterError("OpenRouter tool call has invalid format.")
        function = raw_tool_call.get("function")
        if not isinstance(function, dict):
            raise OpenRouterError("OpenRouter tool call is missing function data.")
        tool_id = raw_tool_call.get("id")
        name = function.get("name")
        arguments_raw = function.get("arguments")
        if not isinstance(tool_id, str) or not tool_id:
            raise OpenRouterError("OpenRouter tool call is missing an id.")
        if not isinstance(name, str) or not name:
            raise OpenRouterError("OpenRouter tool call is missing a function name.")
        if not isinstance(arguments_raw, str):
            raise OpenRouterError("OpenRouter tool call arguments must be a string.")
        try:
            arguments = json.loads(arguments_raw)
        except json.JSONDecodeError as exc:
            raise OpenRouterError(
                f"OpenRouter tool call arguments for {name} are not valid JSON."
            ) from exc
        if not isinstance(arguments, dict):
            raise OpenRouterError(
                f"OpenRouter tool call arguments for {name} must decode to an object."
            )
        tool_calls.append(ToolCall(id=tool_id, name=name, arguments=arguments))

    return tool_calls
