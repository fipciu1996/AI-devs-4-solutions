"""OpenRouter client utilities used by the proxy API server."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib import error, request

from proxy_api_server.config import Settings


class OpenRouterError(RuntimeError):
    """Raised when the OpenRouter request fails."""


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

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def create_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatCompletionResult:
        payload: dict[str, Any] = {
            "model": self._settings.openrouter_model,
            "messages": messages,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        headers = {
            "Authorization": f"Bearer {self._settings.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        if self._settings.openrouter_app_url:
            headers["HTTP-Referer"] = self._settings.openrouter_app_url
        if self._settings.openrouter_app_title:
            headers["X-Title"] = self._settings.openrouter_app_title

        req = request.Request(
            self._settings.openrouter_base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with request.urlopen(
                req,
                timeout=self._settings.openrouter_timeout_seconds,
            ) as response:
                raw_response = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            raise OpenRouterError(
                f"OpenRouter returned HTTP {exc.code}: {detail or exc.reason}"
            ) from exc
        except error.URLError as exc:
            raise OpenRouterError(f"OpenRouter request failed: {exc.reason}") from exc
        except TimeoutError as exc:
            raise OpenRouterError("OpenRouter request timed out.") from exc

        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise OpenRouterError("OpenRouter returned invalid JSON.") from exc

        return _extract_completion_result(parsed)


def _extract_completion_result(payload: dict[str, Any]) -> ChatCompletionResult:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise OpenRouterError("OpenRouter response did not contain choices.")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise OpenRouterError("OpenRouter response choice has invalid format.")

    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise OpenRouterError("OpenRouter response did not contain a message.")

    content = _extract_message_content(message.get("content"))
    tool_calls = _extract_tool_calls(message.get("tool_calls"))

    if not content and not tool_calls:
        raise OpenRouterError("OpenRouter returned an empty response.")

    return ChatCompletionResult(content=content, tool_calls=tool_calls)


def _extract_message_content(content: Any) -> str | None:
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


def _extract_tool_calls(raw_tool_calls: Any) -> list[ToolCall]:
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
