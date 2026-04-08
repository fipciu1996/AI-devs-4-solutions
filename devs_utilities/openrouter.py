"""Shared OpenRouter client and response parsing helpers."""

from __future__ import annotations

import ast
import json
import re
import time
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
    usage_output_path: Path | None = None
    usage_task_name: str | None = None


def build_task_site_name(source_path: str | Path, *, task_name: str | None = None) -> str:
    """Build the standard X-Title header for a task script."""

    normalized_task_name = (task_name or "").strip()
    if not normalized_task_name:
        normalized_task_name = Path(source_path).resolve().parent.name
    return f"AI Devs 4 - {normalized_task_name}"


def build_task_usage_output_path(source_path: str | Path, *, task_name: str | None = None) -> Path:
    """Build the standard per-task OpenRouter usage report path."""

    task_dir = Path(source_path).resolve().parent
    normalized_task_name = (task_name or "").strip()
    if not normalized_task_name or normalized_task_name == task_dir.name:
        return task_dir / "openrouter_usage.json"
    return task_dir / f"openrouter_usage_{normalized_task_name}.json"


def get_default_openrouter_site_url() -> str | None:
    """Read the standard optional site URL headers for OpenRouter."""

    return get_optional_env("OPENROUTER_SITE_URL") or get_optional_env("OPENROUTER_APP_URL")


def build_task_openrouter_client(
    source_path: str | Path,
    *,
    api_key: str,
    base_url: str,
    model: str,
    task_name: str | None = None,
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
            site_name=site_name if site_name is not None else build_task_site_name(source_path, task_name=task_name),
            usage_output_path=build_task_usage_output_path(source_path, task_name=task_name),
            usage_task_name=(task_name or "").strip() or Path(source_path).resolve().parent.name,
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


@dataclass(frozen=True, slots=True)
class OpenRouterUsageSnapshot:
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    reasoning_tokens: int
    cache_write_tokens: int
    total_tokens: int

    def as_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_tokens": self.cached_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "total_tokens": self.total_tokens,
        }


def _coerce_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    return 0


def extract_usage_snapshot(payload: dict[str, Any]) -> OpenRouterUsageSnapshot:
    """Extract token-usage details from an OpenRouter response payload."""

    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return OpenRouterUsageSnapshot(
            input_tokens=0,
            output_tokens=0,
            cached_tokens=0,
            reasoning_tokens=0,
            cache_write_tokens=0,
            total_tokens=0,
        )

    prompt_details = usage.get("prompt_tokens_details")
    if not isinstance(prompt_details, dict):
        prompt_details = {}

    completion_details = usage.get("completion_tokens_details")
    if not isinstance(completion_details, dict):
        completion_details = {}

    input_tokens = _coerce_int(usage.get("prompt_tokens"))
    output_tokens = _coerce_int(usage.get("completion_tokens"))
    cached_tokens = _coerce_int(prompt_details.get("cached_tokens"))
    reasoning_tokens = _coerce_int(completion_details.get("reasoning_tokens"))
    cache_write_tokens = _coerce_int(prompt_details.get("cache_write_tokens"))
    total_tokens = _coerce_int(usage.get("total_tokens")) or (input_tokens + output_tokens)
    return OpenRouterUsageSnapshot(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
        reasoning_tokens=reasoning_tokens,
        cache_write_tokens=cache_write_tokens,
        total_tokens=total_tokens,
    )


class _UsageTracker:
    """Aggregate OpenRouter token usage for one solver invocation."""

    def __init__(self, *, output_path: Path, task_name: str | None) -> None:
        self.output_path = output_path
        self.task_name = task_name or output_path.parent.name
        self._calls: list[dict[str, Any]] = []
        self._by_model: dict[str, OpenRouterUsageSnapshot] = {}
        self._write()

    def record(self, *, payload: dict[str, Any], configured_model: str) -> None:
        snapshot = extract_usage_snapshot(payload)
        model_name = str(payload.get("model") or configured_model or "unknown").strip() or "unknown"
        self._calls.append(
            {
                "index": len(self._calls) + 1,
                "model": model_name,
                **snapshot.as_dict(),
            }
        )
        current = self._by_model.get(model_name)
        if current is None:
            self._by_model[model_name] = snapshot
        else:
            self._by_model[model_name] = OpenRouterUsageSnapshot(
                input_tokens=current.input_tokens + snapshot.input_tokens,
                output_tokens=current.output_tokens + snapshot.output_tokens,
                cached_tokens=current.cached_tokens + snapshot.cached_tokens,
                reasoning_tokens=current.reasoning_tokens + snapshot.reasoning_tokens,
                cache_write_tokens=current.cache_write_tokens + snapshot.cache_write_tokens,
                total_tokens=current.total_tokens + snapshot.total_tokens,
            )
        self._write()

    def _totals(self) -> OpenRouterUsageSnapshot:
        totals = OpenRouterUsageSnapshot(
            input_tokens=0,
            output_tokens=0,
            cached_tokens=0,
            reasoning_tokens=0,
            cache_write_tokens=0,
            total_tokens=0,
        )
        for snapshot in self._by_model.values():
            totals = OpenRouterUsageSnapshot(
                input_tokens=totals.input_tokens + snapshot.input_tokens,
                output_tokens=totals.output_tokens + snapshot.output_tokens,
                cached_tokens=totals.cached_tokens + snapshot.cached_tokens,
                reasoning_tokens=totals.reasoning_tokens + snapshot.reasoning_tokens,
                cache_write_tokens=totals.cache_write_tokens + snapshot.cache_write_tokens,
                total_tokens=totals.total_tokens + snapshot.total_tokens,
            )
        return totals

    def _write(self) -> None:
        payload = {
            "task": self.task_name,
            "request_count": len(self._calls),
            "models": sorted(self._by_model),
            "totals": self._totals().as_dict(),
            "usage_by_model": {
                model_name: snapshot.as_dict()
                for model_name, snapshot in sorted(self._by_model.items())
            },
            "calls": list(self._calls),
        }
        self.output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


class OpenRouterClient:
    """Thin client for OpenRouter chat completions."""

    _trackers: dict[str, _UsageTracker] = {}

    def __init__(self, config: OpenRouterConfig) -> None:
        self._config = config
        self._tracker = self._build_tracker()

    def _build_tracker(self) -> _UsageTracker | None:
        if self._config.usage_output_path is None:
            return None
        key = str(self._config.usage_output_path.resolve())
        tracker = self._trackers.get(key)
        if tracker is None:
            tracker = _UsageTracker(
                output_path=self._config.usage_output_path.resolve(),
                task_name=self._config.usage_task_name,
            )
            self._trackers[key] = tracker
        return tracker

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

        for attempt in range(3):
            try:
                parsed = post_json(
                    self._config.base_url,
                    payload,
                    headers=headers,
                    timeout_seconds=self._config.timeout_seconds,
                )
                break
            except HttpRequestError as exc:
                if exc.status_code == 429 and attempt < 2:
                    time.sleep(5 * (attempt + 1))
                    continue
                raise OpenRouterError(str(exc)) from exc
            except JsonResponseError as exc:
                raise OpenRouterError(
                    f"OpenRouter returned invalid JSON for {exc.url}."
                ) from exc

        if not isinstance(parsed, dict):
            raise OpenRouterError("OpenRouter returned a non-object response.")
        if parsed.get("error"):
            raise OpenRouterError(str(parsed["error"]))
        if self._tracker is not None:
            self._tracker.record(payload=parsed, configured_model=self._config.model)
        return parsed

    def create_raw_completion_legacy(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        extra_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Backward-compatible alias for older task scripts."""

        return self.create_raw_completion(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            extra_payload=extra_payload,
        )

    def create_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        extra_payload: dict[str, Any] | None = None,
    ) -> ChatCompletionResult:
        last_error: OpenRouterError | None = None
        for attempt in range(3):
            try:
                return extract_completion_result(
                    self.create_raw_completion(
                        messages,
                        tools=tools,
                        tool_choice=tool_choice,
                        extra_payload=extra_payload,
                    )
                )
            except OpenRouterError as exc:
                if "empty response" not in str(exc).lower() or attempt == 2:
                    raise
                last_error = exc
        if last_error is not None:
            raise last_error
        raise OpenRouterError("OpenRouter completion failed unexpectedly.")


def strip_code_fences(text: str) -> str:
    """Remove a single surrounding fenced code block when present."""

    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def parse_json_object_content(content: str) -> dict[str, Any]:
    """Parse a model text response as a JSON object."""

    normalized = strip_code_fences(content)
    try:
        parsed = json.loads(normalized)
    except json.JSONDecodeError as exc:
        raise OpenRouterError("OpenRouter did not return valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise OpenRouterError("OpenRouter returned a non-object JSON payload.")
    return parsed


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


def _load_json_with_common_repairs(arguments_raw: str) -> Any:
    """Parse JSON with a few conservative repairs for common model mistakes."""

    normalized = strip_code_fences(arguments_raw).strip()
    normalized = normalized.removeprefix("TOOLCALL>").removesuffix("ALL>").strip()
    candidates: list[str] = []

    def add_candidate(candidate: str) -> None:
        stripped = candidate.strip()
        if stripped and stripped not in candidates:
            candidates.append(stripped)

    def balance_unclosed_structures(candidate: str) -> str:
        closers: list[str] = []
        in_string = False
        escape = False
        for char in candidate:
            if escape:
                escape = False
                continue
            if char == "\\" and in_string:
                escape = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                closers.append("}")
            elif char == "[":
                closers.append("]")
            elif char in "}]" and closers and char == closers[-1]:
                closers.pop()
        suffix = ""
        if in_string:
            suffix += '"'
        if closers:
            suffix += "".join(reversed(closers))
        return candidate + suffix

    def quote_bare_object_keys(candidate: str) -> str:
        return re.sub(r'([{\[,]\s*)([A-Za-z_][A-Za-z0-9_-]*)\s*:', r'\1"\2":', candidate)

    def replace_bare_key_equals(candidate: str) -> str:
        return re.sub(r'([{\[,]\s*)([A-Za-z_][A-Za-z0-9_-]*)\s*=', r'\1"\2":', candidate)

    add_candidate(normalized)

    if ":" in normalized and not normalized.startswith(("{", "[")):
        add_candidate("{" + normalized + "}")
    if "=" in normalized and not normalized.startswith(("{", "[")):
        add_candidate("{" + normalized + "}")

    without_trailing_commas = re.sub(r",\s*([}\]])", r"\1", normalized)
    if without_trailing_commas != normalized:
        add_candidate(without_trailing_commas)

    for candidate in list(candidates):
        add_candidate(quote_bare_object_keys(candidate))
        add_candidate(replace_bare_key_equals(candidate))
        add_candidate(re.sub(r",\s*([}\]])", r"\1", quote_bare_object_keys(candidate)))
        add_candidate(re.sub(r",\s*([}\]])", r"\1", replace_bare_key_equals(candidate)))
        add_candidate(balance_unclosed_structures(candidate))

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    for candidate in candidates:
        try:
            return ast.literal_eval(candidate)
        except (SyntaxError, ValueError):
            continue
    raise json.JSONDecodeError("invalid JSON", normalized, 0)


def _normalize_tool_call_arguments(name: str, arguments: Any) -> dict[str, Any]:
    """Coerce slightly-off tool-call payloads into the expected object shape."""

    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, list):
        if name == "batch_get_records":
            return {"requests": arguments}
        if name == "batch_update_records":
            return {"updates": arguments}
        if name == "get_messages":
            return {"ids": arguments}
    raise OpenRouterError(
        f"OpenRouter tool call arguments for {name} must decode to an object."
    )


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
        name = re.sub(r"<\|.*$", "", name.strip())
        name = re.sub(r"[^A-Za-z0-9_:-].*$", "", name)
        if not name:
            raise OpenRouterError("OpenRouter tool call function name is invalid.")
        if not isinstance(arguments_raw, str):
            raise OpenRouterError("OpenRouter tool call arguments must be a string.")
        try:
            arguments = _load_json_with_common_repairs(arguments_raw)
        except json.JSONDecodeError as exc:
            raise OpenRouterError(
                f"OpenRouter tool call arguments for {name} are not valid JSON."
            ) from exc
        arguments = _normalize_tool_call_arguments(name, arguments)
        tool_calls.append(ToolCall(id=tool_id, name=name, arguments=arguments))

    return tool_calls
