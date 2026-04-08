"""Solve the AG3NTS shellaccess task with an OpenRouter tool-calling agent."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

REPO_ROOT_HINT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_HINT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_HINT))

from devs_utilities.ag3nts import AG3NTS_VERIFY_URL, submit_task_answer
from devs_utilities.bootstrap import bootstrap_repo
from devs_utilities.files import write_json
from devs_utilities.flags import extract_flag
from devs_utilities.http import HttpRequestError, JsonResponseError
from devs_utilities.logging import configure_logging, logger as shared_logger
from devs_utilities.openrouter import (
    OpenRouterError,
    ToolCall,
    build_task_openrouter_client,
    build_task_site_name,
    parse_json_object_content,
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
logger = shared_logger.bind(component="shellaccess")

TASK_NAME = "shellaccess"
DEFAULT_MODEL = get_llm_model("SHELLACCESS_MODEL")
DEFAULT_API_TIMEOUT_SECONDS = (
    get_int_env(
        "SHELLACCESS_TIMEOUT_SECONDS",
        get_int_env("AG3NTS_TIMEOUT_SECONDS", 30) or 30,
    )
    or 30
)
DEFAULT_OPENROUTER_TIMEOUT_SECONDS = get_int_env("OPENROUTER_TIMEOUT_SECONDS", 120) or 120
DEFAULT_MAX_STEPS = get_int_env("SHELLACCESS_MAX_STEPS", 18) or 18

OUTPUT_DIR = Path(__file__).resolve().parent
LAST_ANSWER_PATH = OUTPUT_DIR / "last_answer.json"
LAST_COMMAND_HISTORY_PATH = OUTPUT_DIR / "command_history.json"
LAST_TRANSCRIPT_PATH = OUTPUT_DIR / "last_transcript.json"
LAST_VERIFY_RESPONSE_PATH = OUTPUT_DIR / "last_verify_response.json"
LAST_SESSION_PATH = OUTPUT_DIR / "last_session.json"

ALLOWED_REMOTE_COMMANDS = {
    "awk",
    "cat",
    "cut",
    "date",
    "echo",
    "egrep",
    "fgrep",
    "file",
    "find",
    "grep",
    "head",
    "jq",
    "ls",
    "paste",
    "printf",
    "pwd",
    "sed",
    "sort",
    "tail",
    "tr",
    "true",
    "uniq",
    "wc",
    "xargs",
}
FORBIDDEN_REDIRECTION_RE = re.compile(r"[<>]")
SYSTEM_PROMPT = load_prompt_text(__file__, "system_prompt.txt")

INITIAL_USER_PROMPT = """Solve the shellaccess task. Explore the remote files,
correlate the date, city, and coordinates, then submit the final answer."""


class ShellAccessError(RuntimeError):
    """Raised when the shellaccess agent cannot complete its workflow."""


@dataclass(frozen=True, slots=True)
class CommandRecord:
    """A single remote command together with its parsed response."""

    command: str
    response: Any


@dataclass(slots=True)
class AgentState:
    """Mutable solver state shared across tool calls."""

    final_flag: str | None = None
    final_response: Any = None
    last_answer: dict[str, Any] | None = None
    command_history: list[CommandRecord] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Runtime configuration for the shellaccess agent."""

    api_key: str
    verify_url: str
    openrouter_api_key: str
    openrouter_url: str
    model: str
    api_timeout_seconds: int
    openrouter_timeout_seconds: int
    site_url: str | None
    site_name: str | None
    max_steps: int
    show_tool_results: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "OpenRouter model override. Defaults to the model configured in the "
            "repository .env."
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"Maximum OpenRouter tool-calling rounds. Default: {DEFAULT_MAX_STEPS}.",
    )
    parser.add_argument(
        "--show-tool-results",
        action="store_true",
        help="Print every OpenRouter tool result during the run.",
    )
    parser.add_argument(
        "--transcript-path",
        default=str(LAST_TRANSCRIPT_PATH),
        help="Where to save the final OpenRouter transcript.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> AppConfig:
    api_key = (get_env("AG3NTS_API_KEY") or get_course_api_key()).strip()
    openrouter_api_key = (get_llm_api_key() or get_env("OPENROUTER_API_KEY")).strip()
    openrouter_url = (get_llm_base_url() or get_env("OPENROUTER_BASE_URL")).strip()
    model = (args.model or DEFAULT_MODEL).strip()

    api_timeout_raw = (
        get_optional_env("SHELLACCESS_TIMEOUT_SECONDS")
        or get_optional_env("AG3NTS_TIMEOUT_SECONDS")
        or str(DEFAULT_API_TIMEOUT_SECONDS)
    )
    try:
        api_timeout_seconds = max(10, int(api_timeout_raw))
    except ValueError as exc:
        raise SystemExit(
            "SHELLACCESS_TIMEOUT_SECONDS/AG3NTS_TIMEOUT_SECONDS must be an integer, "
            f"got: {api_timeout_raw}"
        ) from exc
    try:
        openrouter_timeout_seconds = max(10, int(DEFAULT_OPENROUTER_TIMEOUT_SECONDS))
    except ValueError as exc:
        raise SystemExit(
            "OPENROUTER_TIMEOUT_SECONDS must be an integer, "
            f"got: {DEFAULT_OPENROUTER_TIMEOUT_SECONDS}"
        ) from exc

    missing: list[str] = []
    if not api_key:
        missing.append("AG3NTS_API_KEY/COURSE_API_KEY")
    if not openrouter_api_key:
        missing.append("LLM_API_KEY")
    if not openrouter_url:
        missing.append("LLM_BASE_URL")
    if missing:
        raise SystemExit(f"Missing required settings: {', '.join(missing)}")
    if args.max_steps < 1:
        raise SystemExit("--max-steps must be a positive integer.")

    site_url = get_optional_env("OPENROUTER_SITE_URL") or get_optional_env("OPENROUTER_APP_URL")
    site_name = build_task_site_name(__file__, task_name=TASK_NAME)

    return AppConfig(
        api_key=api_key,
        verify_url=AG3NTS_VERIFY_URL,
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


def validate_remote_command(command: str) -> str:
    """Allow only read-only shell commands needed for the task."""

    normalized = command.strip()
    if not normalized:
        raise ShellAccessError("Remote command cannot be empty.")
    if "\n" in normalized or "\r" in normalized:
        raise ShellAccessError("Remote command must be a single line.")
    if FORBIDDEN_REDIRECTION_RE.search(normalized):
        raise ShellAccessError("Redirections are not allowed in remote commands.")

    for segment in split_shell_segments(normalized):
        if not segment:
            continue
        try:
            tokens = shlex.split(segment, posix=True)
        except ValueError as exc:
            raise ShellAccessError(f"Invalid shell syntax: {exc}") from exc
        if not tokens:
            continue
        if tokens[0] not in ALLOWED_REMOTE_COMMANDS:
            raise ShellAccessError(
                f"Remote command `{tokens[0]}` is not allowed by the local safety policy."
            )
    return normalized


def split_shell_segments(command: str) -> list[str]:
    """Split a shell command on unquoted pipeline and chaining operators."""

    segments: list[str] = []
    current: list[str] = []
    quote: str | None = None
    index = 0

    while index < len(command):
        char = command[index]
        next_two = command[index : index + 2]

        if quote is not None:
            current.append(char)
            if char == quote:
                quote = None
            elif char == "\\" and quote == '"' and index + 1 < len(command):
                index += 1
                current.append(command[index])
            index += 1
            continue

        if char in {"'", '"'}:
            quote = char
            current.append(char)
            index += 1
            continue

        if next_two in {"&&", "||"}:
            segment = "".join(current).strip()
            if segment:
                segments.append(segment)
            current = []
            index += 2
            continue

        if char in {"|", ";"}:
            segment = "".join(current).strip()
            if segment:
                segments.append(segment)
            current = []
            index += 1
            continue

        current.append(char)
        index += 1

    if quote is not None:
        raise ShellAccessError("Invalid shell syntax: No closing quotation")

    segment = "".join(current).strip()
    if segment:
        segments.append(segment)
    return segments


def build_answer_payload(
    *,
    date: str,
    city: str,
    longitude: float,
    latitude: float,
) -> dict[str, Any]:
    """Normalize the final JSON payload expected by the task."""

    return {
        "date": str(date).strip(),
        "city": str(city).strip(),
        "longitude": float(longitude),
        "latitude": float(latitude),
    }


def build_answer_command(answer: dict[str, Any]) -> str:
    """Build a safe POSIX echo command for the final answer JSON."""

    json_payload = json.dumps(answer, ensure_ascii=False, separators=(",", ":"))
    return f"echo {shlex.quote(json_payload)}"


class VerifyClient:
    """Thin wrapper around the task verify endpoint."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def execute_command(self, command: str) -> dict[str, Any]:
        normalized = validate_remote_command(command)
        return self._submit({"cmd": normalized}, command=normalized)

    def submit_answer(
        self,
        *,
        date: str,
        city: str,
        longitude: float,
        latitude: float,
    ) -> dict[str, Any]:
        answer = build_answer_payload(
            date=date,
            city=city,
            longitude=longitude,
            latitude=latitude,
        )
        return self._submit(
            {"cmd": build_answer_command(answer)},
            submitted_answer=answer,
        )

    def _submit(
        self,
        task_answer: dict[str, Any],
        *,
        command: str | None = None,
        submitted_answer: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            response = submit_task_answer(
                self._config.verify_url,
                api_key=self._config.api_key,
                task=TASK_NAME,
                answer=task_answer,
                timeout_seconds=self._config.api_timeout_seconds,
            )
        except HttpRequestError as exc:
            parsed = exc.body_as_json()
            if isinstance(parsed, dict):
                result = dict(parsed)
            else:
                result = exc.to_response_dict()
            result.setdefault("http_status", exc.status_code)
            if command is not None:
                result.setdefault("command", command)
            if submitted_answer is not None:
                result.setdefault("submitted_answer", submitted_answer)
            return result
        except JsonResponseError as exc:
            result = {
                "error": f"Verify endpoint returned invalid JSON for {exc.url}.",
            }
            if command is not None:
                result["command"] = command
            if submitted_answer is not None:
                result["submitted_answer"] = submitted_answer
            return result

        if isinstance(response, dict):
            result = dict(response)
        else:
            result = {"raw": response}
        if command is not None:
            result.setdefault("command", command)
        if submitted_answer is not None:
            result.setdefault("submitted_answer", submitted_answer)
        return result


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_time_logs",
            "description": (
                "Search time_logs.csv with grep and return only a small tail of matches. "
                "Use extended_regex=true for alternations like foo|bar."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search phrase or regex pattern.",
                    },
                    "extended_regex": {
                        "type": "boolean",
                        "description": "Use grep -E when true.",
                        "default": False,
                    },
                    "tail_lines": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 40,
                        "default": 20,
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
            "name": "read_log_window",
            "description": "Read a specific line window from time_logs.csv with sed -n.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_line": {
                        "type": "integer",
                        "minimum": 1,
                    },
                    "end_line": {
                        "type": "integer",
                        "minimum": 1,
                    },
                },
                "required": ["start_line", "end_line"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_location_name",
            "description": "Resolve one location_id from locations.json into a city name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location_id": {
                        "type": "integer",
                        "minimum": 0,
                    }
                },
                "required": ["location_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_gps_entry",
            "description": (
                "Resolve one entry_id from gps.json into latitude, longitude, type, "
                "and location_id."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entry_id": {
                        "type": "integer",
                        "minimum": 1,
                    }
                },
                "required": ["entry_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_remote_command",
            "description": (
                "Execute one read-only shell command on the remote server through the "
                "shellaccess task interface. Keep commands narrow."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "A single read-only shell command.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Short reason for running the command.",
                    },
                },
                "required": ["command", "reason"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": (
                "Submit the structured final answer. The tool formats the exact echo "
                "command required by the task and sends it to verify."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "The required day in YYYY-MM-DD format.",
                    },
                    "city": {
                        "type": "string",
                        "description": "The city where Rafal's body was found.",
                    },
                    "longitude": {
                        "type": "number",
                        "description": "Longitude of the target point.",
                    },
                    "latitude": {
                        "type": "number",
                        "description": "Latitude of the target point.",
                    },
                },
                "required": ["date", "city", "longitude", "latitude"],
                "additionalProperties": False,
            },
        },
    },
]


def build_tool_handlers(
    *,
    verify_client: VerifyClient,
    state: AgentState,
) -> dict[str, Callable[[dict[str, Any]], dict[str, Any]]]:
    def serialize_history() -> list[dict[str, Any]]:
        return [asdict(record) for record in state.command_history]

    def run_command(command: str) -> dict[str, Any]:
        result = verify_client.execute_command(command)
        state.command_history.append(CommandRecord(command=command, response=result))
        write_json(LAST_COMMAND_HISTORY_PATH, serialize_history())
        maybe_capture_final_flag(state, result)
        return result

    def execute_remote_command(arguments: dict[str, Any]) -> dict[str, Any]:
        command = str(arguments["command"])
        return run_command(command)

    def search_time_logs(arguments: dict[str, Any]) -> dict[str, Any]:
        query = str(arguments["query"]).strip()
        if not query:
            raise ShellAccessError("search_time_logs requires a non-empty query.")
        tail_lines = int(arguments.get("tail_lines", 20))
        tail_lines = max(1, min(40, tail_lines))
        grep_options = "-n -E" if bool(arguments.get("extended_regex")) else "-n"
        command = (
            f"grep {grep_options} {shlex.quote(query)} /data/time_logs.csv | "
            f"tail -n {tail_lines}"
        )
        result = run_command(command)
        return {
            "query": query,
            "extended_regex": bool(arguments.get("extended_regex")),
            "tail_lines": tail_lines,
            "result": result,
        }

    def read_log_window(arguments: dict[str, Any]) -> dict[str, Any]:
        start_line = int(arguments["start_line"])
        end_line = int(arguments["end_line"])
        if start_line < 1 or end_line < start_line:
            raise ShellAccessError("read_log_window expects 1 <= start_line <= end_line.")
        if end_line - start_line > 80:
            raise ShellAccessError("read_log_window range is too wide; keep it under 80 lines.")
        command = f"sed -n '{start_line},{end_line}p' /data/time_logs.csv"
        result = run_command(command)
        return {
            "start_line": start_line,
            "end_line": end_line,
            "result": result,
        }

    def get_location_name(arguments: dict[str, Any]) -> dict[str, Any]:
        location_id = int(arguments["location_id"])
        command = f"jq -r '.[] | select(.location_id=={location_id}) | .name' /data/locations.json"
        result = run_command(command)
        output = str(result.get("output", "")).strip()
        name = output.splitlines()[0].strip() if output else ""
        if name == "null":
            name = ""
        return {
            "location_id": location_id,
            "name": name or None,
            "result": result,
        }

    def get_gps_entry(arguments: dict[str, Any]) -> dict[str, Any]:
        entry_id = int(arguments["entry_id"])
        command = (
            f"jq -r '.[] | select(.entry_id=={entry_id}) | "
            " [.latitude,.longitude,.type,.location_id,.entry_id] | @tsv' /data/gps.json"
        )
        result = run_command(command)
        output = str(result.get("output", "")).strip()
        latitude: float | None = None
        longitude: float | None = None
        entry_type: str | None = None
        location_id: int | None = None
        resolved_entry_id: int | None = None
        if output:
            parts = output.splitlines()[0].split("\t")
            if len(parts) >= 5:
                try:
                    latitude = float(parts[0])
                    longitude = float(parts[1])
                    entry_type = parts[2]
                    location_id = int(parts[3])
                    resolved_entry_id = int(parts[4])
                except ValueError:
                    latitude = None
                    longitude = None
                    entry_type = None
                    location_id = None
                    resolved_entry_id = None
        return {
            "entry_id": entry_id,
            "latitude": latitude,
            "longitude": longitude,
            "type": entry_type,
            "location_id": location_id,
            "resolved_entry_id": resolved_entry_id,
            "result": result,
        }

    def submit_answer(arguments: dict[str, Any]) -> dict[str, Any]:
        answer = build_answer_payload(
            date=str(arguments["date"]),
            city=str(arguments["city"]),
            longitude=float(arguments["longitude"]),
            latitude=float(arguments["latitude"]),
        )
        state.last_answer = answer
        write_json(LAST_ANSWER_PATH, answer)
        result = verify_client.submit_answer(**answer)
        state.final_response = result
        write_json(LAST_VERIFY_RESPONSE_PATH, result)
        maybe_capture_final_flag(state, result)
        return result

    return {
        "search_time_logs": search_time_logs,
        "read_log_window": read_log_window,
        "get_location_name": get_location_name,
        "get_gps_entry": get_gps_entry,
        "execute_remote_command": execute_remote_command,
        "submit_answer": submit_answer,
    }


def maybe_capture_final_flag(state: AgentState, result: Any) -> None:
    """Extract and store a flag when the hub returns one."""

    flag = extract_flag(result)
    if flag:
        state.final_flag = flag
        state.final_response = result


def execute_tool_call(
    tool_call: ToolCall,
    handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]],
    *,
    show_tool_results: bool,
) -> dict[str, Any]:
    if tool_call.name not in handlers:
        raise ShellAccessError(f"Model called an unknown tool: {tool_call.name}")

    try:
        result = handlers[tool_call.name](tool_call.arguments)
    except (ShellAccessError, ValueError, TypeError) as exc:
        result = {
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


def maybe_finish_from_plain_text(
    content: str | None,
    *,
    handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]],
    state: AgentState,
) -> bool:
    """Handle the occasional plain-text answer instead of a tool call."""

    if not content:
        return False

    direct_flag = extract_flag(content)
    if direct_flag:
        state.final_flag = direct_flag
        state.final_response = {"message": content}
        return True

    try:
        payload = parse_json_object_content(content)
    except OpenRouterError:
        return False

    required_keys = {"date", "city", "longitude", "latitude"}
    if not required_keys.issubset(payload):
        return False

    result = handlers["submit_answer"](
        {
            "date": payload["date"],
            "city": payload["city"],
            "longitude": payload["longitude"],
            "latitude": payload["latitude"],
        }
    )
    maybe_capture_final_flag(state, result)
    return True


def build_session_payload(
    *,
    config: AppConfig,
    state: AgentState,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a compact debug snapshot for the last solver run."""

    return {
        "task": TASK_NAME,
        "model": config.model,
        "max_steps": config.max_steps,
        "show_tool_results": config.show_tool_results,
        "command_count": len(state.command_history),
        "commands": [asdict(record) for record in state.command_history],
        "last_answer": state.last_answer,
        "final_response": state.final_response,
        "final_flag": state.final_flag,
        "transcript_length": len(messages),
    }


def run_agent(
    *,
    config: AppConfig,
    transcript_path: Path,
) -> tuple[str | None, AgentState, list[dict[str, Any]]]:
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
    verify_client = VerifyClient(config)
    handlers = build_tool_handlers(verify_client=verify_client, state=state)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": INITIAL_USER_PROMPT},
    ]

    try:
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
                        return state.final_flag, state, messages
                continue

            if maybe_finish_from_plain_text(
                completion.content,
                handlers=handlers,
                state=state,
            ):
                return state.final_flag, state, messages

            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Continue with tools only. Explore /data with narrow commands or "
                        "submit a concrete final answer."
                    ),
                }
            )
    finally:
        write_json(transcript_path, messages)
        write_json(
            LAST_SESSION_PATH,
            build_session_payload(config=config, state=state, messages=messages),
        )

    raise ShellAccessError(
        f"Agent did not finish within {config.max_steps} tool-calling rounds."
    )


def main() -> int:
    args = parse_args()
    config = build_config(args)
    transcript_path = Path(args.transcript_path).resolve()
    configure_logging(name=TASK_NAME, verbose=config.show_tool_results)

    try:
        final_flag, state, _messages = run_agent(
            config=config,
            transcript_path=transcript_path,
        )
    except (OpenRouterError, ShellAccessError) as exc:
        logger.error("Solver failed: {}", exc)
        return 1

    logger.info("Model: {}", config.model)
    if state.last_answer is not None:
        logger.info(
            "Last answer:\n{}",
            json.dumps(state.last_answer, ensure_ascii=False, indent=2),
        )
    if state.final_response is not None:
        logger.info(
            "Verify response:\n{}",
            json.dumps(state.final_response, ensure_ascii=False, indent=2),
        )
    if final_flag:
        logger.success("Flag: {}", final_flag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
