"""Shared OpenRouter tool-calling runtime for timetravel agents."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable

from devs_utilities.files import write_json
from devs_utilities.logging import logger as shared_logger
from devs_utilities.openrouter import OpenRouterClient, OpenRouterError, ToolCall


logger = shared_logger.bind(component="timetravel-openrouter")


ToolHandler = Callable[[dict[str, Any]], Any]
FinishPredicate = Callable[[], bool]
PlainTextHandler = Callable[[str | None], bool]


@dataclass(frozen=True, slots=True)
class ToolCallingAgentConfig:
    """Configuration for a single OpenRouter tool-calling loop."""

    name: str
    system_prompt: str
    initial_user_prompt: str
    tools: list[dict[str, Any]]
    max_steps: int
    transcript_path: Path
    show_tool_results: bool = False
    completion_retry_attempts: int = 2
    continue_prompt: str = "Continue with tools only."


def run_tool_calling_agent(
    *,
    client: OpenRouterClient,
    config: ToolCallingAgentConfig,
    tool_handlers: dict[str, ToolHandler],
    finish_predicate: FinishPredicate,
    plain_text_handler: PlainTextHandler | None = None,
) -> list[dict[str, Any]]:
    """Run a generic OpenRouter tool-calling loop and persist the transcript."""

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": config.system_prompt},
        {"role": "user", "content": config.initial_user_prompt},
    ]

    try:
        for _step in range(config.max_steps):
            completion = _create_completion_with_retry(
                client=client,
                messages=messages,
                tools=config.tools,
                retry_attempts=config.completion_retry_attempts,
            )
            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": completion.content or "",
            }
            if completion.tool_calls:
                assistant_message["tool_calls"] = [
                    tool_call.to_message_dict() for tool_call in completion.tool_calls
                ]
            messages.append(assistant_message)

            if completion.tool_calls:
                for tool_call in completion.tool_calls:
                    tool_message = execute_tool_call(
                        tool_call,
                        handlers=tool_handlers,
                        show_tool_results=config.show_tool_results,
                    )
                    messages.append(tool_message)
                    if finish_predicate():
                        return messages
                continue

            if plain_text_handler is not None and plain_text_handler(completion.content):
                return messages
            if finish_predicate():
                return messages

            messages.append({"role": "user", "content": config.continue_prompt})
    finally:
        write_json(config.transcript_path, messages)

    raise OpenRouterError(
        f"{config.name} did not finish within {config.max_steps} tool-calling rounds."
    )


def execute_tool_call(
    tool_call: ToolCall,
    *,
    handlers: dict[str, ToolHandler],
    show_tool_results: bool,
) -> dict[str, Any]:
    """Execute one model-requested tool and convert the result into a tool message."""

    handler = handlers.get(tool_call.name)
    if handler is None:
        result: dict[str, Any] = {
            "ok": False,
            "error": f"Unknown tool requested: {tool_call.name}",
        }
    else:
        try:
            raw_result = handler(tool_call.arguments)
        except Exception as exc:  # noqa: BLE001 - tool failures should be fed back to the model
            result = {
                "ok": False,
                "error": str(exc),
                "tool": tool_call.name,
            }
        else:
            result = _normalize_tool_result(raw_result)

    if show_tool_results:
        logger.info(
            "Tool {} args:\n{}\nTool result:\n{}",
            tool_call.name,
            json.dumps(tool_call.arguments, ensure_ascii=False, indent=2),
            json.dumps(result, ensure_ascii=False, indent=2),
        )

    return {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "name": tool_call.name,
        "content": json.dumps(result, ensure_ascii=False),
    }


def _create_completion_with_retry(
    *,
    client: OpenRouterClient,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    retry_attempts: int,
):
    completion = None
    for attempt in range(1, max(1, retry_attempts) + 1):
        try:
            completion = client.create_completion(messages, tools=tools)
            break
        except OpenRouterError as exc:
            if attempt >= retry_attempts:
                raise
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous step failed or returned malformed tool arguments. "
                        "Retry using only valid tool calls with strict JSON arguments."
                    ),
                }
            )
            logger.warning("Retrying OpenRouter step after transient error: {}", exc)
    if completion is None:
        raise OpenRouterError("OpenRouter did not produce a completion.")
    return completion


def _normalize_tool_result(raw_result: Any) -> dict[str, Any]:
    if isinstance(raw_result, dict):
        return raw_result
    return {"ok": True, "result": raw_result}
