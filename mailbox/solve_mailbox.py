"""Solve the AG3NTS mailbox task with a tool-calling agent."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
REPO_ROOT_HINT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_HINT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_HINT))

from devs_utilities.ag3nts import (
    AG3NTS_VERIFY_URL,
    AG3NTS_ZMAIL_URL,
    submit_task_answer,
)
from devs_utilities.bootstrap import bootstrap_repo
from devs_utilities.files import write_json
from devs_utilities.flags import extract_flag
from devs_utilities.http import HttpRequestError, JsonResponseError, post_json
from devs_utilities.logging import configure_logging, logger as shared_logger
from devs_utilities.openrouter import (
    build_task_site_name,
    build_task_openrouter_client,
    OpenRouterClient,
    OpenRouterError,
    ToolCall,
)
from devs_utilities.prompts import load_prompt_text
from devs_utilities.repo_env import (
    get_course_api_key,
    get_env,
    get_int_env,
    get_llm_api_key,
    get_llm_base_url,
    get_llm_model,
    get_optional_env,
)


REPO_ROOT = bootstrap_repo(__file__)
logger = shared_logger.bind(component="mailbox")


TASK_NAME = "mailbox"
DEFAULT_ZMAIL_URL = AG3NTS_ZMAIL_URL
DEFAULT_MODEL = get_llm_model("MAILBOX_MODEL")
DEFAULT_API_TIMEOUT_SECONDS = (
    get_int_env(
        "MAILBOX_TIMEOUT_SECONDS",
        get_int_env("AG3NTS_TIMEOUT_SECONDS", 120) or 120,
    )
    or 120
)
DEFAULT_OPENROUTER_TIMEOUT_SECONDS = get_int_env("OPENROUTER_TIMEOUT_SECONDS", 120) or 120
DEFAULT_MAX_STEPS = get_int_env("MAILBOX_MAX_STEPS", 24) or 24
DEFAULT_TOOL_PAGE_SIZE = get_int_env("MAILBOX_TOOL_PAGE_SIZE", 5) or 5
DEFAULT_WAIT_SECONDS = get_int_env("MAILBOX_WAIT_SECONDS", 5) or 5
MAX_WAIT_SECONDS = get_int_env("MAILBOX_MAX_WAIT_SECONDS", 30) or 30
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
CONFIRMATION_CODE_PATTERN = re.compile(r"^SEC-[A-Za-z0-9]{28,32}$")
OUTPUT_DIR = Path(__file__).resolve().parent
LAST_RESPONSE_PATH = OUTPUT_DIR / "last_verify_response.json"
LAST_ANSWER_PATH = OUTPUT_DIR / "last_answer.json"
LAST_TRANSCRIPT_PATH = OUTPUT_DIR / "last_transcript.json"
SYSTEM_PROMPT = load_prompt_text(__file__, "system_prompt.txt")

INITIAL_USER_PROMPT = """Solve the mailbox task. Search iteratively, inspect message bodies,
and keep going until verify returns the final flag. If the mailbox changes while you work,
refresh the inbox or repeat promising searches."""


class MailboxError(RuntimeError):
    """Raised when a remote API or the agent returns an invalid response."""


@dataclass(slots=True)
class AppConfig:
    ag3nts_api_key: str
    verify_url: str
    zmail_url: str
    openrouter_api_key: str
    openrouter_url: str
    model: str
    api_timeout_seconds: int
    openrouter_timeout_seconds: int
    site_url: str | None
    site_name: str | None
    max_steps: int
    show_tool_results: bool


@dataclass(slots=True)
class AgentState:
    final_flag: str | None = None
    final_response: Any = None
    last_answer: dict[str, str] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=None,
        help="OpenRouter model. Defaults to the model configured in the repository .env.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"Maximum tool-calling rounds. Default: {DEFAULT_MAX_STEPS}.",
    )
    parser.add_argument(
        "--show-tool-results",
        action="store_true",
        help="Print every tool result during the run.",
    )
    parser.add_argument(
        "--transcript-path",
        default=str(LAST_TRANSCRIPT_PATH),
        help="Where to save the final message transcript.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> AppConfig:
    ag3nts_api_key = get_course_api_key()
    openrouter_api_key = get_llm_api_key()
    openrouter_url = get_llm_base_url()
    model = (args.model or DEFAULT_MODEL).strip()
    zmail_url = DEFAULT_ZMAIL_URL

    timeout_raw = get_env("MAILBOX_TIMEOUT_SECONDS") or get_env(
        "AG3NTS_TIMEOUT_SECONDS",
        str(DEFAULT_API_TIMEOUT_SECONDS),
    )
    try:
        api_timeout_seconds = max(10, int(timeout_raw))
    except ValueError as exc:
        raise SystemExit(
            f"MAILBOX_TIMEOUT_SECONDS/AG3NTS_TIMEOUT_SECONDS must be an integer, got: {timeout_raw}"
        ) from exc
    try:
        openrouter_timeout_seconds = max(10, int(DEFAULT_OPENROUTER_TIMEOUT_SECONDS))
    except ValueError as exc:
        raise SystemExit(
            "OPENROUTER_TIMEOUT_SECONDS must be an integer, "
            f"got: {DEFAULT_OPENROUTER_TIMEOUT_SECONDS}"
        ) from exc

    missing: list[str] = []
    if not ag3nts_api_key:
        missing.append("COURSE_API_KEY")
    if not openrouter_api_key:
        missing.append("LLM_API_KEY")
    if not openrouter_url:
        missing.append("LLM_BASE_URL")

    if missing:
        joined = ", ".join(missing)
        raise SystemExit(f"Missing required settings: {joined}")
    if args.max_steps < 1:
        raise SystemExit("--max-steps must be a positive integer.")

    site_url = get_optional_env("OPENROUTER_SITE_URL") or get_optional_env("OPENROUTER_APP_URL")
    site_name = build_task_site_name(__file__, task_name=TASK_NAME)

    return AppConfig(
        ag3nts_api_key=ag3nts_api_key,
        verify_url=AG3NTS_VERIFY_URL,
        zmail_url=zmail_url,
        openrouter_api_key=openrouter_api_key,
        openrouter_url=openrouter_url,
        model=model,
        api_timeout_seconds=api_timeout_seconds,
        openrouter_timeout_seconds=openrouter_timeout_seconds,
        site_url=site_url,
        site_name=site_name,
        max_steps=args.max_steps,
        show_tool_results=args.show_tool_results,
    )


class ZmailClient:
    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def help(self) -> Any:
        return self._call({"action": "help", "page": 1})

    def get_inbox(self, *, page: int = 1, per_page: int = DEFAULT_TOOL_PAGE_SIZE) -> Any:
        return self._call({"action": "getInbox", "page": page, "perPage": per_page})

    def search(self, *, query: str, page: int = 1, per_page: int = DEFAULT_TOOL_PAGE_SIZE) -> Any:
        return self._call(
            {
                "action": "search",
                "query": query,
                "page": page,
                "perPage": per_page,
            }
        )

    def get_thread(self, *, thread_id: int) -> Any:
        return self._call({"action": "getThread", "threadID": thread_id})

    def get_thread_messages(self, *, thread_id: int) -> Any:
        thread_result = self.get_thread(thread_id=thread_id)
        items = thread_result.get("items") if isinstance(thread_result, dict) else None
        if not isinstance(items, list):
            return {
                "ok": False,
                "thread_id": thread_id,
                "thread": thread_result,
                "error": "Thread lookup did not return message metadata.",
            }

        message_ids = [
            item.get("messageID")
            for item in items
            if isinstance(item, dict) and isinstance(item.get("messageID"), str)
        ]
        message_ids = [message_id for message_id in message_ids if message_id]
        if not message_ids:
            return {
                "ok": False,
                "thread_id": thread_id,
                "thread": thread_result,
                "error": "Thread does not expose any current messageID values.",
            }

        messages_result = self.get_messages(ids=message_ids)
        return {
            "ok": bool(messages_result.get("ok", True)),
            "thread_id": thread_id,
            "message_ids": message_ids,
            "thread": thread_result,
            "messages": messages_result,
            "agent_note": (
                "Use this tool when a thread matters. It refreshes the live thread and "
                "immediately fetches the current messages to avoid stale messageID values."
            ),
        }

    def get_messages(self, *, ids: int | str | list[int | str]) -> Any:
        return self._call({"action": "getMessages", "ids": normalize_message_ids(ids)})

    def reset(self) -> Any:
        return self._call({"action": "reset"})

    def _call(self, payload: dict[str, Any]) -> Any:
        full_payload = {"apikey": self._config.ag3nts_api_key, **payload}
        try:
            result = post_json(
                self._config.zmail_url,
                full_payload,
                timeout_seconds=self._config.api_timeout_seconds,
            )
        except HttpRequestError as exc:
            result = exc.to_response_dict()
            if isinstance(result, dict):
                result.setdefault("ok", False)
                result.setdefault(
                    "agent_note",
                    "Some mailbox ids may be stale. Refresh the thread or inbox and retry with current messageID values.",
                )
                return result
            raise MailboxError(str(exc)) from exc
        except JsonResponseError as exc:
            raise MailboxError(str(exc)) from exc
        if isinstance(result, dict):
            result.setdefault(
                "agent_note",
                "Prefer messageID as the stable fetch identifier. The mailbox is live.",
            )
        return result


def normalize_message_ids(ids: int | str | list[int | str]) -> int | str | list[int | str]:
    """Coerce model-produced message identifiers into valid mailbox API payloads."""

    if isinstance(ids, list):
        return [normalize_message_ids(item) for item in ids]
    if isinstance(ids, int):
        return ids
    raw_value = str(ids).strip()
    if not raw_value:
        return raw_value
    if raw_value.startswith("[") and raw_value.endswith("]"):
        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError:
            return raw_value
        if isinstance(parsed, list):
            return [normalize_message_ids(item) for item in parsed]
    if raw_value.isdigit():
        return int(raw_value)
    return raw_value


class VerifyClient:
    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def submit_answer(self, answer: dict[str, str]) -> Any:
        try:
            return submit_task_answer(
                self._config.verify_url,
                api_key=self._config.ag3nts_api_key,
                task=TASK_NAME,
                answer=answer,
                timeout_seconds=self._config.api_timeout_seconds,
            )
        except HttpRequestError as exc:
            raise MailboxError(str(exc)) from exc


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "zmail_help",
            "description": "Show the currently available zmail actions and parameters.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_inbox",
            "description": "Read the latest inbox page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "page": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 1,
                    },
                    "per_page": {
                        "type": "integer",
                        "minimum": 5,
                        "maximum": 20,
                        "default": DEFAULT_TOOL_PAGE_SIZE,
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_messages",
            "description": "Search mailbox messages with Gmail-like query operators.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Full-text or operator-based search query.",
                    },
                    "page": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 1,
                    },
                    "per_page": {
                        "type": "integer",
                        "minimum": 5,
                        "maximum": 20,
                        "default": DEFAULT_TOOL_PAGE_SIZE,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_thread",
            "description": "List the current message IDs in a thread.",
            "parameters": {
                "type": "object",
                "properties": {
                    "thread_id": {
                        "type": "integer",
                        "description": "Numeric thread identifier.",
                    }
                },
                "required": ["thread_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_messages",
            "description": (
                "Fetch one or more full messages by messageID or rowID. "
                "Prefer messageID because the mailbox is live."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ids": {
                        "oneOf": [
                            {"type": "integer"},
                            {"type": "string"},
                            {
                                "type": "array",
                                "items": {
                                    "oneOf": [
                                        {"type": "integer"},
                                        {"type": "string"},
                                    ]
                                },
                                "minItems": 1,
                            },
                        ]
                    }
                },
                "required": ["ids"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_thread_messages",
            "description": (
                "Refresh a live thread and immediately fetch its current full messages. "
                "Prefer this over separate get_thread and get_messages when a thread is important."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "thread_id": {
                        "type": "integer",
                        "description": "Numeric thread identifier.",
                    }
                },
                "required": ["thread_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": (
                "Submit the current candidate answer to the AG3NTS verify endpoint. "
                "Use empty strings only when a field is still unknown and you need hub feedback."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "password": {
                        "type": "string",
                        "description": "Candidate employee-system password.",
                    },
                    "date": {
                        "type": "string",
                        "description": "Candidate attack date in YYYY-MM-DD.",
                    },
                    "confirmation_code": {
                        "type": "string",
                        "description": "Candidate SEC confirmation code.",
                    },
                },
                "required": ["password", "date", "confirmation_code"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wait_for_new_messages",
            "description": "Wait briefly before refreshing the live mailbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "seconds": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_WAIT_SECONDS,
                        "default": DEFAULT_WAIT_SECONDS,
                    },
                    "reason": {
                        "type": "string",
                        "description": "Short reason for waiting.",
                    },
                },
                "required": ["seconds", "reason"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reset_zmail_counter",
            "description": "Reset the zmail request counter if the mailbox API starts refusing requests.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
]


def normalize_answer(arguments: dict[str, Any]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key in ("password", "date", "confirmation_code"):
        value = arguments.get(key, "")
        if value is None:
            normalized[key] = ""
        else:
            normalized[key] = str(value).strip()
    return normalized


def build_tool_handlers(
    *,
    zmail_client: ZmailClient,
    verify_client: VerifyClient,
    state: AgentState,
) -> dict[str, Any]:
    def require_argument(arguments: dict[str, Any], key: str) -> Any:
        value = arguments.get(key)
        if value is None or (isinstance(value, str) and not value.strip()):
            raise MailboxError(f"{key} is required for this tool call.")
        return value

    def submit_answer(arguments: dict[str, Any]) -> Any:
        answer = normalize_answer(arguments)
        if answer["date"] and not DATE_PATTERN.fullmatch(answer["date"]):
            raise MailboxError("date must use the YYYY-MM-DD format.")
        if (
            answer["confirmation_code"]
            and not CONFIRMATION_CODE_PATTERN.fullmatch(answer["confirmation_code"])
        ):
            raise MailboxError(
                "confirmation_code must match SEC- followed by 28 letters or digits."
            )
        state.last_answer = answer
        write_json(LAST_ANSWER_PATH, answer)
        response = verify_client.submit_answer(answer)
        state.final_response = response
        write_json(LAST_RESPONSE_PATH, response)
        flag = extract_flag(response)
        if flag:
            state.final_flag = flag
        return response

    def wait_for_new_messages(arguments: dict[str, Any]) -> Any:
        seconds = int(arguments.get("seconds", DEFAULT_WAIT_SECONDS))
        seconds = max(1, min(MAX_WAIT_SECONDS, seconds))
        reason = str(arguments.get("reason", "")).strip() or "refresh mailbox"
        started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        time.sleep(seconds)
        return {
            "ok": True,
            "waited_seconds": seconds,
            "reason": reason,
            "started_at_local": started_at,
            "finished_at_local": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    return {
        "zmail_help": lambda _args: zmail_client.help(),
        "get_inbox": lambda args: zmail_client.get_inbox(
            page=int(args.get("page", 1)),
            per_page=int(args.get("per_page", DEFAULT_TOOL_PAGE_SIZE)),
        ),
        "search_messages": lambda args: zmail_client.search(
            query=str(require_argument(args, "query")),
            page=int(args.get("page", 1)),
            per_page=int(args.get("per_page", DEFAULT_TOOL_PAGE_SIZE)),
        ),
        "search": lambda args: zmail_client.search(
            query=str(require_argument(args, "query")),
            page=int(args.get("page", 1)),
            per_page=int(args.get("per_page", DEFAULT_TOOL_PAGE_SIZE)),
        ),
        "get_thread": lambda args: zmail_client.get_thread(
            thread_id=int(require_argument(args, "thread_id"))
        ),
        "get_thread_messages": lambda args: zmail_client.get_thread_messages(
            thread_id=int(require_argument(args, "thread_id"))
        ),
        "get_messages": lambda args: zmail_client.get_messages(ids=require_argument(args, "ids")),
        "submit_answer": submit_answer,
        "wait_for_new_messages": wait_for_new_messages,
        "reset_zmail_counter": lambda _args: zmail_client.reset(),
    }


def execute_tool_call(
    tool_call: ToolCall,
    handlers: dict[str, Any],
    *,
    show_tool_results: bool,
) -> dict[str, Any]:
    if tool_call.name not in handlers:
        raise MailboxError(f"Model called an unknown tool: {tool_call.name}")

    try:
        result = handlers[tool_call.name](tool_call.arguments)
    except Exception as exc:  # noqa: BLE001 - agent safety net
        result = {
            "ok": False,
            "error": str(exc),
            "tool_name": tool_call.name,
            "arguments": tool_call.arguments,
        }
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


def run_agent(
    *,
    config: AppConfig,
    transcript_path: Path,
) -> tuple[str, AgentState, list[dict[str, Any]]]:
    state = AgentState()
    openrouter_client = build_task_openrouter_client(
        __file__,
        api_key=config.openrouter_api_key,
        base_url=config.openrouter_url,
        model=config.model,
        task_name=TASK_NAME,
        timeout_seconds=config.openrouter_timeout_seconds,
        site_url=config.site_url,
        site_name=config.site_name,
    )
    zmail_client = ZmailClient(config)
    verify_client = VerifyClient(config)
    handlers = build_tool_handlers(
        zmail_client=zmail_client,
        verify_client=verify_client,
        state=state,
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": INITIAL_USER_PROMPT},
    ]

    for _step in range(config.max_steps):
        completion = openrouter_client.create_completion(messages, tools=TOOLS)
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
                    handlers,
                    show_tool_results=config.show_tool_results,
                )
                messages.append(tool_message)
                if state.final_flag:
                    write_json(transcript_path, messages)
                    return state.final_flag, state, messages
            continue

        if completion.content:
            if state.final_flag:
                write_json(transcript_path, messages)
                return state.final_flag, state, messages
            if "FLG:" in completion.content:
                write_json(transcript_path, messages)
                return completion.content.strip(), state, messages

        if state.final_flag:
            write_json(transcript_path, messages)
            return state.final_flag, state, messages

    write_json(transcript_path, messages)
    raise MailboxError(
        f"Agent did not finish within {config.max_steps} tool-calling rounds."
    )


def main() -> int:
    args = parse_args()
    configure_logging(name="mailbox", verbose=args.show_tool_results)
    config = build_config(args)
    transcript_path = Path(args.transcript_path).resolve()

    try:
        flag, state, _messages = run_agent(config=config, transcript_path=transcript_path)
    except (MailboxError, OpenRouterError) as exc:
        logger.error("Error: {}", exc)
        return 1

    logger.info("Model: {}", config.model)
    if state.last_answer:
        logger.info(
            "Answer:\n{}",
            json.dumps(state.last_answer, ensure_ascii=False, indent=2),
        )
    if state.final_response is not None:
        logger.info(
            "Verify response:\n{}",
            json.dumps(state.final_response, ensure_ascii=False, indent=2),
        )
    logger.success("Flag: {}", flag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
