"""Solve the AG3NTS mailbox task with a tool-calling agent."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from repo_env import get_env, get_optional_env, load_repo_env


load_repo_env(__file__)


TASK_NAME = "mailbox"
DEFAULT_ZMAIL_URL = "https://example.invalid/api/zmail"
DEFAULT_MODEL = "openai/gpt-4.1-mini"
DEFAULT_MAX_STEPS = 24
DEFAULT_TOOL_PAGE_SIZE = 5
DEFAULT_WAIT_SECONDS = 5
MAX_WAIT_SECONDS = 30
OUTPUT_DIR = Path(__file__).resolve().parent
LAST_RESPONSE_PATH = OUTPUT_DIR / "last_verify_response.json"
LAST_ANSWER_PATH = OUTPUT_DIR / "last_answer.json"
LAST_TRANSCRIPT_PATH = OUTPUT_DIR / "last_transcript.json"

SYSTEM_PROMPT = """You are a mailbox investigation agent working through tools only.

Goal:
- find the date when security plans to attack the power plant
- find the current employee-system password from the mailbox
- find the confirmation code from the security ticket
- keep searching and submitting answers until the hub returns a flag

Hard rules:
- Start by calling zmail_help unless its result is already in the conversation.
- Always read full message bodies with get_messages before trusting a clue.
- Prefer messageID over rowID when fetching messages. The mailbox is live and rowIDs may shift.
- When a search hit belongs to an interesting thread, inspect the whole thread with get_thread and then read the newest relevant messages.
- Newer corrective emails override older contradictory emails.
- The mailbox is active. If you cannot find something, refresh page 1 of inbox or repeat key searches after a short wait.
- Use submit_answer whenever you have a serious candidate set. Hub feedback tells you what is still wrong or missing.
- Stop only after submit_answer returns a flag.

Known facts:
- Wiktor sent a denunciation email from the proton.me domain.
- Search supports Gmail-like operators: from:, to:, subject:, OR, AND.
- The final answer must contain:
  - date in YYYY-MM-DD
  - password as found in the mailbox
  - confirmation_code in format SEC- followed by 28 characters

Good search angles:
- from:proton.me
- PWR6132PL
- SEC-
- security
- pracowniczy
- haslo / password

Be concise in assistant text. Think with the tools.
"""

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
    timeout_seconds: int
    site_url: str | None
    site_name: str | None
    max_steps: int
    show_tool_results: bool


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
        help=(
            "OpenRouter model. Defaults to MAILBOX_OPENROUTER_MODEL or "
            f"{DEFAULT_MODEL}."
        ),
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
    ag3nts_api_key = get_env("AG3NTS_API_KEY")
    verify_url = get_env("AG3NTS_VERIFY_URL")
    openrouter_api_key = get_env("OPENROUTER_API_KEY")
    openrouter_url = get_env("OPENROUTER_BASE_URL")
    model = (args.model or get_env("MAILBOX_OPENROUTER_MODEL") or DEFAULT_MODEL).strip()
    zmail_url = get_env("AG3NTS_ZMAIL_URL") or DEFAULT_ZMAIL_URL

    timeout_raw = get_env("OPENROUTER_TIMEOUT_SECONDS", "120")
    try:
        timeout_seconds = max(10, int(timeout_raw))
    except ValueError as exc:
        raise SystemExit(f"OPENROUTER_TIMEOUT_SECONDS must be an integer, got: {timeout_raw}") from exc

    missing: list[str] = []
    if not ag3nts_api_key:
        missing.append("AG3NTS_API_KEY")
    if not verify_url:
        missing.append("AG3NTS_VERIFY_URL")
    if not openrouter_api_key:
        missing.append("OPENROUTER_API_KEY")
    if not openrouter_url:
        missing.append("OPENROUTER_BASE_URL")
    if not model:
        missing.append("MAILBOX_OPENROUTER_MODEL")

    if missing:
        joined = ", ".join(missing)
        raise SystemExit(f"Missing required settings: {joined}")
    if args.max_steps < 1:
        raise SystemExit("--max-steps must be a positive integer.")

    site_url = get_optional_env("OPENROUTER_SITE_URL") or get_optional_env("OPENROUTER_APP_URL")
    site_name = get_optional_env("OPENROUTER_SITE_NAME") or get_optional_env("OPENROUTER_APP_TITLE")

    return AppConfig(
        ag3nts_api_key=ag3nts_api_key,
        verify_url=verify_url,
        zmail_url=zmail_url,
        openrouter_api_key=openrouter_api_key,
        openrouter_url=openrouter_url,
        model=model,
        timeout_seconds=timeout_seconds,
        site_url=site_url,
        site_name=site_name,
        max_steps=args.max_steps,
        show_tool_results=args.show_tool_results,
    )


def post_json(
    url: str,
    payload: dict[str, Any],
    *,
    timeout_seconds: int,
    headers: dict[str, str] | None = None,
) -> Any:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request_headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if headers:
        request_headers.update(headers)

    http_request = request.Request(
        url=url,
        data=body,
        headers=request_headers,
        method="POST",
    )
    try:
        with request.urlopen(http_request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        raise MailboxError(f"HTTP {exc.code} for {url}: {detail or exc.reason}") from exc
    except error.URLError as exc:
        raise MailboxError(f"Network error for {url}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise MailboxError(f"Request to {url} timed out.") from exc

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def maybe_extract_flag(payload: Any) -> str | None:
    if isinstance(payload, str):
        return payload if payload.startswith("{FLG:") else None

    if isinstance(payload, dict):
        for value in payload.values():
            flag = maybe_extract_flag(value)
            if flag:
                return flag
        return None

    if isinstance(payload, list):
        for item in payload:
            flag = maybe_extract_flag(item)
            if flag:
                return flag

    return None


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

    def get_messages(self, *, ids: int | str | list[int | str]) -> Any:
        return self._call({"action": "getMessages", "ids": ids})

    def reset(self) -> Any:
        return self._call({"action": "reset"})

    def _call(self, payload: dict[str, Any]) -> Any:
        full_payload = {"apikey": self._config.ag3nts_api_key, **payload}
        result = post_json(
            self._config.zmail_url,
            full_payload,
            timeout_seconds=self._config.timeout_seconds,
        )
        if isinstance(result, dict):
            result.setdefault(
                "agent_note",
                "Prefer messageID as the stable fetch identifier. The mailbox is live.",
            )
        return result


class VerifyClient:
    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def submit_answer(self, answer: dict[str, str]) -> Any:
        payload = {
            "apikey": self._config.ag3nts_api_key,
            "task": TASK_NAME,
            "answer": answer,
        }
        return post_json(
            self._config.verify_url,
            payload,
            timeout_seconds=self._config.timeout_seconds,
        )


class OpenRouterClient:
    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def create_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
    ) -> ChatCompletionResult:
        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
        }
        headers = {
            "Authorization": f"Bearer {self._config.openrouter_api_key}",
        }
        if self._config.site_url:
            headers["HTTP-Referer"] = self._config.site_url
        if self._config.site_name:
            headers["X-Title"] = self._config.site_name

        parsed = post_json(
            self._config.openrouter_url,
            payload,
            timeout_seconds=self._config.timeout_seconds,
            headers=headers,
        )
        if not isinstance(parsed, dict):
            raise MailboxError("OpenRouter returned a non-object response.")
        return extract_completion_result(parsed)


def extract_completion_result(payload: dict[str, Any]) -> ChatCompletionResult:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise MailboxError("OpenRouter response is missing choices.")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise MailboxError("OpenRouter choice has an invalid format.")

    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise MailboxError("OpenRouter response is missing a message.")

    return ChatCompletionResult(
        content=extract_text_content(message.get("content")),
        tool_calls=extract_tool_calls(message.get("tool_calls")),
    )


def extract_text_content(content: Any) -> str | None:
    if content is None:
        return None
    if isinstance(content, str):
        stripped = content.strip()
        return stripped or None
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
    raise MailboxError("OpenRouter returned unsupported message content.")


def extract_tool_calls(raw_tool_calls: Any) -> list[ToolCall]:
    if raw_tool_calls is None:
        return []
    if not isinstance(raw_tool_calls, list):
        raise MailboxError("OpenRouter tool_calls field has an invalid format.")

    tool_calls: list[ToolCall] = []
    for raw_tool_call in raw_tool_calls:
        if not isinstance(raw_tool_call, dict):
            raise MailboxError("OpenRouter tool call item has an invalid format.")
        function_block = raw_tool_call.get("function")
        if not isinstance(function_block, dict):
            raise MailboxError("OpenRouter tool call is missing a function block.")
        tool_id = raw_tool_call.get("id")
        name = function_block.get("name")
        arguments_raw = function_block.get("arguments")
        if not isinstance(tool_id, str) or not tool_id:
            raise MailboxError("OpenRouter tool call is missing an id.")
        if not isinstance(name, str) or not name:
            raise MailboxError("OpenRouter tool call is missing a function name.")
        if not isinstance(arguments_raw, str):
            raise MailboxError(f"OpenRouter arguments for {name} must be a JSON string.")
        try:
            arguments = json.loads(arguments_raw)
        except json.JSONDecodeError as exc:
            raise MailboxError(f"OpenRouter arguments for {name} are not valid JSON.") from exc
        if not isinstance(arguments, dict):
            raise MailboxError(f"OpenRouter arguments for {name} must decode to an object.")
        tool_calls.append(ToolCall(id=tool_id, name=name, arguments=arguments))
    return tool_calls


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


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_tool_handlers(
    *,
    zmail_client: ZmailClient,
    verify_client: VerifyClient,
    state: AgentState,
) -> dict[str, Any]:
    def submit_answer(arguments: dict[str, Any]) -> Any:
        answer = normalize_answer(arguments)
        state.last_answer = answer
        save_json(LAST_ANSWER_PATH, answer)
        response = verify_client.submit_answer(answer)
        state.final_response = response
        save_json(LAST_RESPONSE_PATH, response)
        flag = maybe_extract_flag(response)
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
            query=str(args["query"]),
            page=int(args.get("page", 1)),
            per_page=int(args.get("per_page", DEFAULT_TOOL_PAGE_SIZE)),
        ),
        "get_thread": lambda args: zmail_client.get_thread(thread_id=int(args["thread_id"])),
        "get_messages": lambda args: zmail_client.get_messages(ids=args["ids"]),
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

    result = handlers[tool_call.name](tool_call.arguments)
    if show_tool_results:
        print(f"\n--- tool {tool_call.name} ---")
        print(json.dumps(tool_call.arguments, ensure_ascii=False, indent=2))
        print(json.dumps(result, ensure_ascii=False, indent=2))

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
    openrouter_client = OpenRouterClient(config)
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
                    save_json(transcript_path, messages)
                    return state.final_flag, state, messages
            continue

        if completion.content:
            if state.final_flag:
                save_json(transcript_path, messages)
                return state.final_flag, state, messages
            if "FLG:" in completion.content:
                save_json(transcript_path, messages)
                return completion.content.strip(), state, messages

        if state.final_flag:
            save_json(transcript_path, messages)
            return state.final_flag, state, messages

    save_json(transcript_path, messages)
    raise MailboxError(
        f"Agent did not finish within {config.max_steps} tool-calling rounds."
    )


def main() -> int:
    args = parse_args()
    config = build_config(args)
    transcript_path = Path(args.transcript_path).resolve()

    try:
        flag, state, _messages = run_agent(config=config, transcript_path=transcript_path)
    except MailboxError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Model: {config.model}")
    if state.last_answer:
        print("Answer:")
        print(json.dumps(state.last_answer, ensure_ascii=False, indent=2))
    if state.final_response is not None:
        print("Verify response:")
        print(json.dumps(state.final_response, ensure_ascii=False, indent=2))
    print(f"Flag: {flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
