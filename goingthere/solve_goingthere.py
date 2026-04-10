"""Solve the AG3NTS `goingthere` task with resilient radio and radar handling."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar
from urllib.parse import urlsplit

REPO_ROOT_HINT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_HINT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_HINT))

from devs_utilities.bootstrap import bootstrap_repo
from devs_utilities.files import write_json
from devs_utilities.http import HttpRequestError, RAW_TEXT, get_text, post_json
from devs_utilities.logging import configure_logging, logger as shared_logger
from devs_utilities.openrouter import (
    build_task_openrouter_client,
    OpenRouterClient,
    OpenRouterError,
    parse_json_content,
    ToolCall,
)
from devs_utilities.repo_env import get_env, get_int_env, get_llm_model, get_optional_env


REPO_ROOT = bootstrap_repo(__file__)
logger = shared_logger.bind(component="goingthere")

TASK_NAME = "goingthere"
BASE_URL = get_env("AG3NTS_BASE_URL").rstrip("/")
VERIFY_URL = f"{BASE_URL}/verify"
MESSAGE_URL = f"{BASE_URL}/api/getmessage"
SCANNER_POST_URL = f"{BASE_URL}/api/frequencyScanner"
HUB_HOST = urlsplit(BASE_URL).netloc.lower()

VERIFY_TIMEOUT_SECONDS = get_int_env("AG3NTS_TIMEOUT_SECONDS", 30) or 30
API_RETRY_ATTEMPTS = get_int_env("GOINGTHERE_API_RETRY_ATTEMPTS", 8) or 8
API_RETRY_BASE_DELAY_SECONDS = float(
    get_int_env("GOINGTHERE_API_RETRY_BASE_DELAY_SECONDS", 1) or 1
)
API_RETRY_MAX_DELAY_SECONDS = float(
    get_int_env("GOINGTHERE_API_RETRY_MAX_DELAY_SECONDS", 8) or 8
)
MAX_GAMES = get_int_env("GOINGTHERE_MAX_GAMES", 40) or 40
MAX_DISARM_CYCLES = get_int_env("GOINGTHERE_MAX_DISARM_CYCLES", 10) or 10
MAX_AGENT_STEPS = get_int_env("GOINGTHERE_TOOL_CALL_MAX_STEPS", 80) or 80
HUB_REQUEST_SPACING_SECONDS = float(
    get_optional_env("GOINGTHERE_REQUEST_SPACING_SECONDS") or "2.0"
)
OPENROUTER_TIMEOUT_SECONDS = min(
    get_int_env("OPENROUTER_TIMEOUT_SECONDS", 60) or 60,
    30,
)

OUTPUT_DIR = Path(__file__).resolve().parent
TRACE_PATH = OUTPUT_DIR / "last_game_trace.json"
VERIFY_RESPONSE_PATH = OUTPUT_DIR / "last_verify_response.json"
GAME_AGENT_SYSTEM_PROMPT_PATH = OUTPUT_DIR / "game_agent_system_prompt.txt"

MOVE_DELTAS = {
    "left": -1,
    "go": 0,
    "right": 1,
}
ALL_MOVES = frozenset(MOVE_DELTAS)
GAME_AGENT_TOOL_NAMES = frozenset(
    {"start_game", "scan_frequency", "disarm_trap", "get_radio_hint", "move_rocket"}
)
FLAG_PATTERN = re.compile(r"\{FLG:[^}]+\}")
FREQUENCY_PATTERN = re.compile(r"frequency[^0-9]{0,20}(?P<value>\d{1,10})", re.IGNORECASE)
DETECTION_CODE_PATTERN = re.compile(
    r"detection\W*code\W*[:=]\W*[\"']?(?P<value>[^\s,\"'}\]]+)",
    re.IGNORECASE,
)
FALLBACK_NUMBER_VALUE_PATTERN = re.compile(r"[:=]\s*(?P<value>\d{1,10})")
FALLBACK_STRING_VALUE_PATTERN = re.compile(r"[:=]\s*[\"'](?P<value>[A-Za-z0-9_]{4,})[\"']")
FALLBACK_BARE_TOKEN_VALUE_PATTERN = re.compile(r"[:=]\s*[`\"']*(?P<value>[A-Za-z0-9_]{5,})")

T = TypeVar("T")
_GAME_AGENT_CLIENT: OpenRouterClient | None | bool = False
_LAST_HUB_REQUEST_AT = 0.0


@dataclass(frozen=True, slots=True)
class Position:
    row: int
    col: int


@dataclass(frozen=True, slots=True)
class GameState:
    player: Position
    base: Position
    current_stone_row: int | None = None


@dataclass(frozen=True, slots=True)
class RadarTrap:
    frequency: int
    detection_code: str


class RetryablePayloadError(RuntimeError):
    """Raised when a response shape is invalid but worth retrying."""


class GameLostError(RuntimeError):
    """Raised when the rocket crashes or a run otherwise ends unsuccessfully."""

    def __init__(self, message: str, *, trace: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.trace = list(trace or [])


class GameRunError(RuntimeError):
    """Raised when a run fails before we can classify it as a normal loss."""

    def __init__(self, message: str, *, trace: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.trace = list(trace or [])


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    message: dict[str, Any]
    state: GameState | None = None
    final_response: dict[str, Any] | None = None
    lost_message: str | None = None


def get_api_key() -> str:
    api_key = get_env("AG3NTS_API_KEY")
    if not api_key:
        raise RuntimeError("Missing AG3NTS_API_KEY in the repository .env file.")
    return api_key


def looks_like_clear_scan(payload: str) -> bool:
    letters_only = "".join(character for character in payload.lower() if character.isalpha())
    return re.search(r"c+l+e+a+r+", letters_only) is not None


def extract_trap_data(payload: str) -> RadarTrap | None:
    frequency_match = FREQUENCY_PATTERN.search(payload)
    detection_code_match = DETECTION_CODE_PATTERN.search(payload)
    if frequency_match and detection_code_match:
        return RadarTrap(
            frequency=int(frequency_match.group("value")),
            detection_code=detection_code_match.group("value"),
        )

    fallback_numbers = [match.group("value") for match in FALLBACK_NUMBER_VALUE_PATTERN.finditer(payload)]
    fallback_strings = [
        match.group("value")
        for match in FALLBACK_STRING_VALUE_PATTERN.finditer(payload)
        if match.group("value").lower() not in {"true", "false", "null"}
    ]
    if not fallback_strings:
        fallback_strings = [
            match.group("value")
            for match in FALLBACK_BARE_TOKEN_VALUE_PATTERN.finditer(payload)
            if match.group("value").lower() not in {"true", "false", "null"}
        ]
    if not fallback_numbers or not fallback_strings:
        return None
    return RadarTrap(
        frequency=int(fallback_numbers[0]),
        detection_code=fallback_strings[-1],
    )


def is_retryable_ag3nts_error(exc: HttpRequestError) -> bool:
    if exc.status_code in {408, 425, 429, 500, 502, 503, 504}:
        return True
    payload = exc.body_as_json()
    if isinstance(payload, dict) and payload.get("code") in {-9999, -500}:
        return True
    return exc.status_code is None


def is_crash_http_error(exc: HttpRequestError) -> bool:
    payload = exc.body_as_json()
    return isinstance(payload, dict) and bool(payload.get("crashed"))


def should_apply_hub_spacing(request_url: str | None) -> bool:
    if not request_url:
        return False
    return urlsplit(request_url).netloc.lower() == HUB_HOST


def with_api_retry(
    action: str,
    operation: Callable[[], T],
    *,
    request_url: str | None = None,
) -> T:
    global _LAST_HUB_REQUEST_AT
    delay_seconds = max(0.1, API_RETRY_BASE_DELAY_SECONDS)
    last_error: Exception | None = None
    apply_spacing = should_apply_hub_spacing(request_url)

    for attempt in range(1, API_RETRY_ATTEMPTS + 1):
        try:
            if apply_spacing and HUB_REQUEST_SPACING_SECONDS > 0:
                wait_seconds = (
                    _LAST_HUB_REQUEST_AT + HUB_REQUEST_SPACING_SECONDS
                ) - time.monotonic()
                if wait_seconds > 0:
                    time.sleep(wait_seconds)
            return operation()
        except HttpRequestError as exc:
            last_error = exc
            should_retry = is_retryable_ag3nts_error(exc)
        except RetryablePayloadError as exc:
            last_error = exc
            should_retry = True
        finally:
            if apply_spacing:
                _LAST_HUB_REQUEST_AT = time.monotonic()

        if attempt >= API_RETRY_ATTEMPTS or not should_retry:
            break

        logger.warning(
            "{} failed on attempt {}/{}: {}. Retrying in {:.1f}s.",
            action,
            attempt,
            API_RETRY_ATTEMPTS,
            last_error,
            delay_seconds,
        )
        time.sleep(delay_seconds)
        delay_seconds = min(API_RETRY_MAX_DELAY_SECONDS, delay_seconds * 2)

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"{action} retry loop ended unexpectedly.")


def build_scanner_get_url(api_key: str) -> str:
    return f"{BASE_URL}/api/frequencyScanner?key={api_key}"


def parse_game_state(
    payload: dict[str, Any],
    *,
    current_base: Position | None = None,
) -> GameState:
    try:
        player = payload["player"]
        base = payload.get("base")
        if base is None:
            if current_base is None:
                raise KeyError("base")
            resolved_base = current_base
        else:
            resolved_base = Position(row=int(base["row"]), col=int(base["col"]))
        current_column = payload.get("currentColumn")
        stone_row: int | None = None
        if isinstance(current_column, dict):
            raw_stone_row = current_column.get("stoneRow")
            if isinstance(raw_stone_row, int):
                stone_row = raw_stone_row
        return GameState(
            player=Position(row=int(player["row"]), col=int(player["col"])),
            base=resolved_base,
            current_stone_row=stone_row,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RetryablePayloadError("Task response is missing player/base coordinates.") from exc


def valid_commands_for_row(row: int) -> list[str]:
    return [
        move
        for move in ("left", "go", "right")
        if 1 <= row + MOVE_DELTAS[move] <= 3
    ]


def state_snapshot(state: GameState) -> dict[str, Any]:
    return {
        "player": {
            "row": state.player.row,
            "col": state.player.col,
        },
        "base": {
            "row": state.base.row,
            "col": state.base.col,
        },
        "current_stone_row": state.current_stone_row,
        "valid_commands": valid_commands_for_row(state.player.row),
        "remaining_move_commands": max(0, state.base.col - state.player.col),
    }


def summarize_game_response(
    payload: dict[str, Any],
    *,
    current_base: Position | None = None,
) -> tuple[dict[str, Any], GameState]:
    state = parse_game_state(payload, current_base=current_base)
    return (
        {
            "code": payload.get("code"),
            "message": payload.get("message"),
            "state": state_snapshot(state),
            "current_column": payload.get("currentColumn"),
        },
        state,
    )


def submit_command(api_key: str, command: str) -> dict[str, Any]:
    def operation() -> dict[str, Any]:
        payload = post_json(
            VERIFY_URL,
            {
                "apikey": api_key,
                "task": TASK_NAME,
                "answer": {"command": command},
            },
            timeout_seconds=VERIFY_TIMEOUT_SECONDS,
            on_decode_error=RAW_TEXT,
        )
        if not isinstance(payload, dict):
            raise RetryablePayloadError(f"Command `{command}` returned a non-object payload.")
        if "code" not in payload:
            raise RetryablePayloadError(f"Command `{command}` response is missing a code.")
        return payload

    return with_api_retry(f"Command `{command}`", operation, request_url=VERIFY_URL)


def request_hint(api_key: str) -> str:
    def operation() -> str:
        payload = post_json(
            MESSAGE_URL,
            {"apikey": api_key},
            timeout_seconds=VERIFY_TIMEOUT_SECONDS,
            on_decode_error=RAW_TEXT,
        )
        if not isinstance(payload, dict) or not isinstance(payload.get("hint"), str):
            raise RetryablePayloadError("Hint response is malformed.")
        return str(payload["hint"])

    return with_api_retry("Radio hint", operation, request_url=MESSAGE_URL)


def read_scanner(api_key: str) -> RadarTrap | None:
    scanner_url = build_scanner_get_url(api_key)

    def operation() -> RadarTrap | None:
        payload = get_text(
            scanner_url,
            timeout_seconds=VERIFY_TIMEOUT_SECONDS,
        )
        if looks_like_clear_scan(payload):
            return None
        trap = extract_trap_data(payload)
        if trap is None:
            preview = payload.strip().replace("\n", "\\n")
            raise RetryablePayloadError(f"Scanner response could not be parsed: {preview}")
        return trap

    return with_api_retry("Frequency scan", operation, request_url=scanner_url)


def compute_disarm_hash(detection_code: str) -> str:
    return hashlib.sha1(f"{detection_code}disarm".encode("utf-8")).hexdigest()


def disarm_trap(api_key: str, trap: RadarTrap) -> Any:
    def operation() -> Any:
        return post_json(
            SCANNER_POST_URL,
            {
                "apikey": api_key,
                "frequency": trap.frequency,
                "disarmHash": compute_disarm_hash(trap.detection_code),
            },
            timeout_seconds=VERIFY_TIMEOUT_SECONDS,
            on_decode_error=RAW_TEXT,
        )

    return with_api_retry(
        f"Disarm frequency {trap.frequency}",
        operation,
        request_url=SCANNER_POST_URL,
    )


def ensure_safe_to_move(
    api_key: str,
    trace: list[dict[str, Any]],
    *,
    game_index: int,
) -> None:
    for cycle in range(1, MAX_DISARM_CYCLES + 1):
        trap = read_scanner(api_key)
        if trap is None:
            append_trace_event(
                trace,
                game_index=game_index,
                event={"kind": "scanner", "status": "clear"},
            )
            return

        append_trace_event(
            trace,
            game_index=game_index,
            event={
                "kind": "scanner",
                "status": "locked",
                "frequency": trap.frequency,
                "detection_code": trap.detection_code,
            },
        )
        logger.warning("Radar lock detected at frequency {}. Disarming.", trap.frequency)
        try:
            response = disarm_trap(api_key, trap)
        except HttpRequestError as exc:
            if is_crash_http_error(exc):
                crash_payload = exc.body_as_json()
                append_trace_event(
                    trace,
                    game_index=game_index,
                    event={
                        "kind": "crash",
                        "phase": "disarm",
                        "frequency": trap.frequency,
                        "response": crash_payload,
                    },
                )
                message = str(exc)
                if isinstance(crash_payload, dict):
                    message = str(
                        crash_payload.get("crashMessage")
                        or crash_payload.get("message")
                        or exc
                    )
                raise GameLostError(message, trace=trace) from exc
            raise
        append_trace_event(
            trace,
            game_index=game_index,
            event={
                "kind": "disarm",
                "frequency": trap.frequency,
                "response": response,
            },
        )

    raise RuntimeError("Could not clear the radar trap after repeated attempts.")


def is_flag_response(payload: dict[str, Any]) -> bool:
    return FLAG_PATTERN.search(json.dumps(payload, ensure_ascii=False)) is not None


def append_trace_event(
    trace: list[dict[str, Any]],
    *,
    game_index: int,
    event: dict[str, Any],
) -> None:
    trace.append(event)
    write_json(
        TRACE_PATH,
        serialize_trace(
            [
                {
                    "game": game_index,
                    "status": "running",
                    "trace": trace,
                }
            ]
        ),
    )


def build_game_agent_client() -> OpenRouterClient | None:
    global _GAME_AGENT_CLIENT
    if _GAME_AGENT_CLIENT is not False:
        return _GAME_AGENT_CLIENT if isinstance(_GAME_AGENT_CLIENT, OpenRouterClient) else None

    api_key = get_optional_env("OPENROUTER_API_KEY") or get_optional_env("LLM_API_KEY")
    base_url = get_optional_env("OPENROUTER_BASE_URL") or get_optional_env("LLM_BASE_URL")
    model = get_llm_model("GOINGTHERE_TOOL_MODEL", "GOINGTHERE_HINT_MODEL", default="")
    if not api_key or not base_url or not model:
        _GAME_AGENT_CLIENT = None
        return None

    _GAME_AGENT_CLIENT = build_task_openrouter_client(
        __file__,
        api_key=api_key,
        base_url=base_url,
        model=model,
        task_name=TASK_NAME,
        timeout_seconds=OPENROUTER_TIMEOUT_SECONDS,
    )
    return _GAME_AGENT_CLIENT


def build_game_agent_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "start_game",
                "description": "Start a fresh goingthere game and return the initial board state.",
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
                "name": "scan_frequency",
                "description": "Check whether the rocket is currently being tracked by the OKO radar.",
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
                "name": "disarm_trap",
                "description": "Disarm the currently detected radar trap using the frequency and detection code returned by scan_frequency.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "frequency": {"type": "integer"},
                        "detection_code": {"type": "string"},
                    },
                    "required": ["frequency", "detection_code"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_radio_hint",
                "description": "Read the radio hint that describes the rock in the next column.",
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
                "name": "move_rocket",
                "description": "Move the rocket one column forward. left=row-1, go=same row, right=row+1.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "enum": ["left", "go", "right"],
                        },
                    },
                    "required": ["command"],
                    "additionalProperties": False,
                },
            },
        },
    ]


def build_game_agent_system_prompt() -> str:
    try:
        return GAME_AGENT_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(
            f"Could not read system prompt from {GAME_AGENT_SYSTEM_PROMPT_PATH}."
        ) from exc


def recover_plaintext_tool_calls(response_text: str, *, step: int) -> list[ToolCall]:
    """Recover tool calls from malformed plaintext dumps like `OLCALL>[...]`."""

    normalized = response_text.strip()
    if '"name"' not in normalized or '"arguments"' not in normalized:
        return []

    payload_start = normalized.find("[")
    if payload_start < 0:
        payload_start = normalized.find("{")
    if payload_start < 0:
        return []

    try:
        parsed = parse_json_content(normalized[payload_start:])
    except OpenRouterError:
        return []

    raw_calls: list[dict[str, Any]]
    if isinstance(parsed, dict):
        raw_calls = [parsed]
    elif isinstance(parsed, list):
        raw_calls = [item for item in parsed if isinstance(item, dict)]
    else:
        return []

    recovered: list[ToolCall] = []
    for index, raw_call in enumerate(raw_calls, start=1):
        name = raw_call.get("name")
        arguments = raw_call.get("arguments", {})
        if not isinstance(name, str) or not isinstance(arguments, dict):
            continue
        if name not in GAME_AGENT_TOOL_NAMES:
            continue
        recovered.append(
            ToolCall(
                id=f"recovered_tool_call_{step}_{index}",
                name=name,
                arguments=arguments,
            )
        )
    return recovered


def execute_game_agent_tool(
    *,
    name: str,
    arguments: dict[str, Any],
    api_key: str,
    trace: list[dict[str, Any]],
    game_index: int,
    current_state: GameState | None,
) -> ToolExecutionResult:
    if name == "start_game":
        response = submit_command(api_key, "start")
        summary, state = summarize_game_response(response)
        append_trace_event(trace, game_index=game_index, event={"kind": "start", "response": response})
        logger.info("Game {} started. Target row: {}.", game_index, state.base.row)
        return ToolExecutionResult(
            message={"status": "started", **summary},
            state=state,
        )

    if name == "scan_frequency":
        trap = read_scanner(api_key)
        if trap is None:
            append_trace_event(
                trace,
                game_index=game_index,
                event={"kind": "scanner", "status": "clear"},
            )
            return ToolExecutionResult(message={"status": "clear"})

        append_trace_event(
            trace,
            game_index=game_index,
            event={
                "kind": "scanner",
                "status": "locked",
                "frequency": trap.frequency,
                "detection_code": trap.detection_code,
            },
        )
        return ToolExecutionResult(
            message={
                "status": "locked",
                "frequency": trap.frequency,
                "detection_code": trap.detection_code,
            }
        )

    if name == "disarm_trap":
        frequency = arguments.get("frequency")
        detection_code = arguments.get("detection_code")
        if not isinstance(frequency, int) or not isinstance(detection_code, str) or not detection_code:
            return ToolExecutionResult(
                message={
                    "status": "error",
                    "error": "disarm_trap requires integer frequency and non-empty detection_code.",
                }
            )
        trap = RadarTrap(frequency=frequency, detection_code=detection_code)
        logger.warning("Radar lock detected at frequency {}. Disarming.", trap.frequency)
        try:
            response = disarm_trap(api_key, trap)
        except HttpRequestError as exc:
            if is_crash_http_error(exc):
                crash_payload = exc.body_as_json()
                append_trace_event(
                    trace,
                    game_index=game_index,
                    event={
                        "kind": "crash",
                        "phase": "disarm",
                        "frequency": trap.frequency,
                        "response": crash_payload,
                    },
                )
                message = str(exc)
                if isinstance(crash_payload, dict):
                    message = str(
                        crash_payload.get("crashMessage")
                        or crash_payload.get("message")
                        or exc
                    )
                return ToolExecutionResult(
                    message={
                        "status": "crashed",
                        "phase": "disarm",
                        "frequency": trap.frequency,
                        "response": crash_payload if isinstance(crash_payload, dict) else {"error": str(exc)},
                    },
                    lost_message=message,
                )
            raise
        append_trace_event(
            trace,
            game_index=game_index,
            event={
                "kind": "disarm",
                "frequency": trap.frequency,
                "response": response,
            },
        )
        return ToolExecutionResult(
            message={
                "status": "disarmed",
                "frequency": trap.frequency,
                "response": response,
            }
        )

    if name == "get_radio_hint":
        hint = request_hint(api_key)
        append_trace_event(trace, game_index=game_index, event={"kind": "hint", "value": hint})
        logger.info("Game {} hint: {}", game_index, hint)
        return ToolExecutionResult(message={"status": "ok", "hint": hint})

    if name == "move_rocket":
        if current_state is None:
            return ToolExecutionResult(
                message={
                    "status": "error",
                    "error": "Call start_game before move_rocket.",
                }
            )
        command = arguments.get("command")
        if command not in ALL_MOVES:
            return ToolExecutionResult(
                message={
                    "status": "error",
                    "error": "move_rocket requires command left, go, or right.",
                }
            )
        valid_commands = valid_commands_for_row(current_state.player.row)
        if command not in valid_commands:
            return ToolExecutionResult(
                message={
                    "status": "error",
                    "error": "Requested move is invalid for the current row.",
                    "command": command,
                    "current_row": current_state.player.row,
                    "valid_commands": valid_commands,
                    "state": state_snapshot(current_state),
                }
            )

        append_trace_event(
            trace,
            game_index=game_index,
            event={"kind": "move_choice", "value": command},
        )
        logger.info(
            "Game {} at column {} row {} -> {}.",
            game_index,
            current_state.player.col,
            current_state.player.row,
            command,
        )
        try:
            response = submit_command(api_key, str(command))
        except HttpRequestError as exc:
            if is_crash_http_error(exc):
                crash_payload = exc.body_as_json()
                append_trace_event(
                    trace,
                    game_index=game_index,
                    event={
                        "kind": "crash",
                        "move": command,
                        "response": crash_payload,
                    },
                )
                message = str(exc)
                if isinstance(crash_payload, dict):
                    message = str(
                        crash_payload.get("crashMessage")
                        or crash_payload.get("message")
                        or exc
                    )
                return ToolExecutionResult(
                    message={
                        "status": "crashed",
                        "command": command,
                        "response": crash_payload if isinstance(crash_payload, dict) else {"error": str(exc)},
                    },
                    lost_message=message,
                )
            raise

        append_trace_event(
            trace,
            game_index=game_index,
            event={"kind": "move_response", "response": response},
        )

        if is_flag_response(response):
            return ToolExecutionResult(
                message={"status": "flag", "response": response},
                final_response=response,
            )

        summary, state = summarize_game_response(response, current_base=current_state.base)
        return ToolExecutionResult(
            message={"status": "moved", "command": command, **summary},
            state=state,
        )

    return ToolExecutionResult(
        message={
            "status": "error",
            "error": f"Unknown tool: {name}",
        }
    )


def run_single_game_with_tool_calling(
    api_key: str,
    *,
    game_index: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    client = build_game_agent_client()
    if client is None:
        raise GameRunError(
            "Tool-calling mode is enabled but no OpenRouter configuration is available."
        )

    trace: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": build_game_agent_system_prompt()},
        {
            "role": "user",
            "content": (
                "Start a fresh goingthere game and solve it completely. "
                "Keep using tools until you reach the flag or the rocket crashes."
            ),
        },
    ]
    current_state: GameState | None = None
    tools = build_game_agent_tools()

    try:
        for step in range(1, MAX_AGENT_STEPS + 1):
            completion = client.create_completion(
                messages,
                tools=tools,
                tool_choice="required",
                extra_payload={"temperature": 0},
            )
            append_trace_event(
                trace,
                game_index=game_index,
                event={
                    "kind": "agent_response",
                    "content": completion.content,
                    "tool_calls": [
                        {
                            "name": tool_call.name,
                            "arguments": tool_call.arguments,
                        }
                        for tool_call in completion.tool_calls
                    ],
                },
            )
            assistant_message: dict[str, Any] = {
                "role": "assistant",
            }
            if completion.content:
                assistant_message["content"] = completion.content
            if completion.tool_calls:
                assistant_message["tool_calls"] = [
                    tool_call.to_message_dict() for tool_call in completion.tool_calls
                ]
            messages.append(assistant_message)

            tool_calls = completion.tool_calls
            if not tool_calls:
                if not completion.content:
                    raise GameRunError(
                        "Tool-calling agent returned an empty non-tool response.",
                        trace=trace,
                    )

                logger.info(
                    "Tool-calling agent returned non-tool response on step {}: {}",
                    step,
                    completion.content,
                )
                tool_calls = recover_plaintext_tool_calls(completion.content, step=step)
                if tool_calls:
                    append_trace_event(
                        trace,
                        game_index=game_index,
                        event={
                            "kind": "recovered_tool_calls",
                            "tool_calls": [
                                {
                                    "name": tool_call.name,
                                    "arguments": tool_call.arguments,
                                }
                                for tool_call in tool_calls
                            ],
                        },
                    )
                else:
                    repair_prompt = (
                        "Use tools only. Your previous reply was not a valid tool call. "
                        "Continue the current goingthere game by calling exactly one or more "
                        "real tools with strict JSON arguments."
                    )
                    messages.append({"role": "user", "content": repair_prompt})
                    append_trace_event(
                        trace,
                        game_index=game_index,
                        event={"kind": "agent_reprompt", "content": repair_prompt},
                    )
                    continue

            for tool_call in tool_calls:
                outcome = execute_game_agent_tool(
                    name=tool_call.name,
                    arguments=tool_call.arguments,
                    api_key=api_key,
                    trace=trace,
                    game_index=game_index,
                    current_state=current_state,
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.name,
                        "content": json.dumps(outcome.message, ensure_ascii=False),
                    }
                )
                if outcome.state is not None:
                    current_state = outcome.state
                if outcome.final_response is not None:
                    return outcome.final_response, trace
                if outcome.lost_message is not None:
                    raise GameLostError(outcome.lost_message, trace=trace)

        raise GameRunError(
            f"Tool-calling agent exceeded {MAX_AGENT_STEPS} tool-call steps.",
            trace=trace,
        )
    except GameLostError:
        raise
    except (HttpRequestError, RetryablePayloadError, RuntimeError, OpenRouterError) as exc:
        raise GameRunError(str(exc), trace=trace) from exc


def serialize_trace(games: list[dict[str, Any]]) -> dict[str, Any]:
    return {"games": games}


def main() -> int:
    configure_logging(name="goingthere")
    api_key = get_api_key()
    if build_game_agent_client() is None:
        payload = {
            "error": "OpenRouter configuration is required because goingthere now uses only the tool-calling agent."
        }
        write_json(VERIFY_RESPONSE_PATH, payload)
        logger.error(payload["error"])
        return 1
    all_games: list[dict[str, Any]] = []

    for game_index in range(1, MAX_GAMES + 1):
        try:
            final_response, trace = run_single_game_with_tool_calling(
                api_key,
                game_index=game_index,
            )
            all_games.append(
                {
                    "game": game_index,
                    "status": "success",
                    "trace": trace,
                }
            )
            write_json(TRACE_PATH, serialize_trace(all_games))
            write_json(VERIFY_RESPONSE_PATH, final_response)
            logger.success("Task solved: {}", final_response)
            return 0
        except GameLostError as exc:
            logger.warning("Game {} ended unsuccessfully: {}", game_index, exc)
            all_games.append(
                {
                    "game": game_index,
                    "status": "lost",
                    "error": str(exc),
                    "trace": exc.trace,
                }
            )
            write_json(TRACE_PATH, serialize_trace(all_games))
            continue
        except GameRunError as exc:
            payload = {"error": str(exc)}
            write_json(VERIFY_RESPONSE_PATH, payload)
            write_json(
                TRACE_PATH,
                serialize_trace(
                    [
                        *all_games,
                        {
                            "game": game_index,
                            "status": "failed",
                            "error": str(exc),
                            "trace": exc.trace,
                        },
                    ]
                ),
            )
            logger.error("Solver failed: {}", exc)
            return 1

    payload = {"error": f"Failed to solve the task after {MAX_GAMES} games."}
    write_json(VERIFY_RESPONSE_PATH, payload)
    write_json(TRACE_PATH, serialize_trace(all_games))
    logger.error(payload["error"])
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
