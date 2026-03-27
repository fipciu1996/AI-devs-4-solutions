"""Solve the AG3NTS reactor task with a deterministic state-space planner."""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

REPO_ROOT_HINT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_HINT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_HINT))

from devs_utilities.ag3nts import AG3NTS_VERIFY_URL, submit_task_answer
from devs_utilities.bootstrap import bootstrap_repo
from devs_utilities.files import write_json
from devs_utilities.flags import extract_flag
from devs_utilities.http import HttpRequestError
from devs_utilities.logging import configure_logging, logger as shared_logger
from devs_utilities.openrouter import (
    OpenRouterClient,
    OpenRouterConfig,
    OpenRouterError,
    ToolCall,
)
from repo_env import get_env, get_int_env, get_optional_env


REPO_ROOT = bootstrap_repo(__file__)
logger = shared_logger.bind(component="reactor")


TASK_NAME = "reactor"
BOARD_COLUMNS = 7
BOARD_ROWS = 5
GOAL_COLUMN = 7
DEFAULT_MAX_STEPS = get_int_env("REACTOR_MAX_STEPS", 64) or 64
VERIFY_TIMEOUT_SECONDS = get_int_env("AG3NTS_TIMEOUT_SECONDS", 30) or 30
DEFAULT_MODEL = get_env("OPENROUTER_MODEL", "openai/gpt-4.1-mini") or "openai/gpt-4.1-mini"
DEFAULT_OPENROUTER_TIMEOUT_SECONDS = (
    get_int_env("OPENROUTER_TIMEOUT_SECONDS", 60) or 60
)
OUTPUT_DIR = Path(__file__).resolve().parent
LAST_RESPONSE_PATH = OUTPUT_DIR / "last_verify_response.json"
LAST_TRACE_PATH = OUTPUT_DIR / "last_trace.json"

MoveCommand = Literal["left", "wait", "right"]
ApiCommand = Literal["start", "reset", "left", "wait", "right"]
Direction = Literal["up", "down"]

ACTION_PRIORITY: tuple[MoveCommand, ...] = ("right", "wait", "left")
MODEL_SYSTEM_PROMPT = """You are the reasoning module for a reactor-navigation robot.

Goal:
- move the robot from column 1 to column 7 on the bottom lane
- never choose a move that can lead to an immediate collision
- prefer progress toward the goal and avoid unnecessary backtracking

Rules:
- Blocks move exactly one step after every command.
- You will receive only safe candidate moves.
- Use tool calling before the final answer.
- Choose the best candidate move for this turn.
- Return valid JSON only, following the provided schema.
- Keep the reason short and concrete.
"""
MODEL_MAX_STEPS = 4


@dataclass(frozen=True, slots=True)
class ReactorOption:
    """A safe move candidate together with its predicted aftermath."""

    command: MoveCommand
    next_state: ReactorState
    remaining_steps: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "remaining_steps_to_goal": self.remaining_steps,
            "next_player_col": self.next_state.player_col,
            "next_board": render_board(self.next_state),
        }


@dataclass(frozen=True, slots=True)
class ModelDecision:
    """A normalized OpenRouter decision payload."""

    command: MoveCommand
    reason: str

    def as_dict(self) -> dict[str, str]:
        return {
            "command": self.command,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class BlockState:
    """A single vertical reactor block in one board column."""

    col: int
    top_row: int
    bottom_row: int
    direction: Direction

    @classmethod
    def from_api_payload(cls, payload: Mapping[str, Any]) -> BlockState:
        return cls(
            col=int(payload["col"]),
            top_row=int(payload["top_row"]),
            bottom_row=int(payload["bottom_row"]),
            direction=str(payload["direction"]),
        )

    def advance(self) -> BlockState:
        """Advance the block by one turn using the API's next-direction convention."""

        if self.direction == "down":
            next_top = self.top_row + 1
            next_bottom = self.bottom_row + 1
            next_direction: Direction = "up" if next_bottom == 5 else "down"
        else:
            next_top = self.top_row - 1
            next_bottom = self.bottom_row - 1
            next_direction = "down" if next_top == 1 else "up"

        return BlockState(
            col=self.col,
            top_row=next_top,
            bottom_row=next_bottom,
            direction=next_direction,
        )

    def occupies_bottom_lane(self) -> bool:
        return self.bottom_row == 5

    def as_dict(self) -> dict[str, Any]:
        return {
            "col": self.col,
            "top_row": self.top_row,
            "bottom_row": self.bottom_row,
            "direction": self.direction,
        }


@dataclass(frozen=True, slots=True)
class ReactorState:
    """The full deterministic state needed to plan the next move."""

    player_col: int
    blocks: tuple[BlockState, ...]

    @classmethod
    def from_api_payload(cls, payload: Mapping[str, Any]) -> ReactorState:
        player = payload.get("player")
        blocks_payload = payload.get("blocks")
        if not isinstance(player, Mapping):
            raise ValueError("Missing player payload in reactor response.")
        if not isinstance(blocks_payload, list):
            raise ValueError("Missing blocks payload in reactor response.")

        blocks = tuple(
            sorted(
                (
                    BlockState.from_api_payload(block_payload)
                    for block_payload in blocks_payload
                    if isinstance(block_payload, Mapping)
                ),
                key=lambda block: block.col,
            )
        )
        if not blocks:
            raise ValueError("Reactor response did not contain any blocks.")

        return cls(player_col=int(player["col"]), blocks=blocks)

    def is_goal(self) -> bool:
        return self.player_col == GOAL_COLUMN

    def apply(self, command: MoveCommand) -> ReactorState | None:
        """Return the next safe state for the given move, or None on collision."""

        delta = {"left": -1, "wait": 0, "right": 1}[command]
        next_player_col = self.player_col + delta
        if not 1 <= next_player_col <= BOARD_COLUMNS:
            return None

        next_blocks = tuple(block.advance() for block in self.blocks)
        for block in next_blocks:
            if block.col == next_player_col and block.occupies_bottom_lane():
                return None

        return ReactorState(player_col=next_player_col, blocks=next_blocks)

    def matches_api_payload(self, payload: Mapping[str, Any]) -> bool:
        try:
            other = ReactorState.from_api_payload(payload)
        except (KeyError, TypeError, ValueError):
            return False
        return self == other

    def as_dict(self) -> dict[str, Any]:
        return {
            "player": {"col": self.player_col, "row": 5},
            "blocks": [block.as_dict() for block in self.blocks],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reset-first",
        action="store_true",
        help="Send reset before the required start command.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"Maximum move commands after start. Default: {DEFAULT_MAX_STEPS}.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "OpenRouter model override. Defaults to OPENROUTER_MODEL or "
            f"{DEFAULT_MODEL}."
        ),
    )
    return parser.parse_args()


def get_api_key() -> str:
    api_key = get_env("AG3NTS_API_KEY")
    if not api_key:
        raise SystemExit("Missing AG3NTS_API_KEY in .env.")
    return api_key


def send_command(api_key: str, command: ApiCommand) -> dict[str, Any]:
    try:
        response = submit_task_answer(
            AG3NTS_VERIFY_URL,
            api_key=api_key,
            task=TASK_NAME,
            answer={"command": command},
            timeout_seconds=VERIFY_TIMEOUT_SECONDS,
        )
    except HttpRequestError as exc:
        return exc.to_response_dict()

    if isinstance(response, dict):
        return dict(response)
    return {"raw": response}


def build_openrouter_client(model_override: str | None) -> OpenRouterClient:
    api_key = get_env("OPENROUTER_API_KEY")
    base_url = get_env("OPENROUTER_BASE_URL")
    model = (model_override or get_optional_env("OPENROUTER_MODEL") or DEFAULT_MODEL).strip()
    timeout_raw = get_optional_env("OPENROUTER_TIMEOUT_SECONDS") or str(
        DEFAULT_OPENROUTER_TIMEOUT_SECONDS
    )

    try:
        timeout_seconds = max(10, int(timeout_raw))
    except ValueError as exc:
        raise SystemExit(
            f"OPENROUTER_TIMEOUT_SECONDS must be an integer, got: {timeout_raw}"
        ) from exc

    missing: list[str] = []
    if not api_key:
        missing.append("OPENROUTER_API_KEY")
    if not base_url:
        missing.append("OPENROUTER_BASE_URL")
    if missing:
        raise SystemExit(f"Missing required OpenRouter settings: {', '.join(missing)}")

    return OpenRouterClient(
        OpenRouterConfig(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout_seconds=timeout_seconds,
            site_url=get_optional_env("OPENROUTER_SITE_URL")
            or get_optional_env("OPENROUTER_APP_URL"),
            site_name=get_optional_env("OPENROUTER_SITE_NAME")
            or get_optional_env("OPENROUTER_APP_TITLE"),
        )
    )


def plan_commands(initial_state: ReactorState) -> list[MoveCommand] | None:
    """Find the shortest safe command sequence from the current state to the goal."""

    queue: deque[ReactorState] = deque([initial_state])
    previous: dict[ReactorState, tuple[ReactorState | None, MoveCommand | None]] = {
        initial_state: (None, None)
    }

    while queue:
        state = queue.popleft()
        if state.is_goal():
            return reconstruct_plan(state, previous)

        for command in ACTION_PRIORITY:
            next_state = state.apply(command)
            if next_state is None or next_state in previous:
                continue
            previous[next_state] = (state, command)
            queue.append(next_state)

    return None


def reconstruct_plan(
    goal_state: ReactorState,
    previous: dict[ReactorState, tuple[ReactorState | None, MoveCommand | None]],
) -> list[MoveCommand]:
    commands: list[MoveCommand] = []
    current: ReactorState | None = goal_state

    while current is not None:
        parent, command = previous[current]
        if command is not None:
            commands.append(command)
        current = parent

    commands.reverse()
    return commands


def render_board(state: ReactorState) -> list[str]:
    board = [["." for _ in range(BOARD_COLUMNS)] for _ in range(BOARD_ROWS)]
    for block in state.blocks:
        board[block.top_row - 1][block.col - 1] = "B"
        board[block.bottom_row - 1][block.col - 1] = "B"
    board[BOARD_ROWS - 1][GOAL_COLUMN - 1] = "G"
    board[BOARD_ROWS - 1][state.player_col - 1] = "P"
    return ["".join(row) for row in board]


def build_safe_options(state: ReactorState) -> list[ReactorOption]:
    options: list[ReactorOption] = []
    for command in ACTION_PRIORITY:
        next_state = state.apply(command)
        if next_state is None:
            continue
        remaining_plan = plan_commands(next_state)
        if remaining_plan is None:
            continue
        options.append(
            ReactorOption(
                command=command,
                next_state=next_state,
                remaining_steps=len(remaining_plan),
            )
        )
    return options


def build_decision_schema(allowed_commands: list[MoveCommand]) -> dict[str, Any]:
    return {
        "name": "reactor_move_decision",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": allowed_commands,
                },
                "reason": {
                    "type": "string",
                },
            },
            "required": ["command", "reason"],
            "additionalProperties": False,
        },
    }


def build_decision_prompt(
    state: ReactorState,
    options: list[ReactorOption],
) -> str:
    payload = {
        "current_player_col": state.player_col,
        "goal_col": GOAL_COLUMN,
        "current_board": render_board(state),
        "blocks": [block.as_dict() for block in state.blocks],
        "safe_options": [option.as_dict() for option in options],
    }
    return (
        "Choose the best command for this turn.\n"
        "The board is represented as 5 strings from top row to bottom row.\n"
        "Return JSON only.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


REACTOR_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_turn_context",
            "description": "Get the current reactor board and all safe move options.",
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
            "name": "validate_command_choice",
            "description": "Validate that a proposed command is among the safe options for this turn.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": ["left", "wait", "right"],
                    }
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    },
]


def parse_model_decision(
    raw_content: str,
    *,
    allowed_commands: set[MoveCommand],
) -> ModelDecision:
    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise OpenRouterError("OpenRouter returned invalid JSON for reactor move.") from exc

    if not isinstance(parsed, dict):
        raise OpenRouterError("OpenRouter reactor decision must be an object.")

    command = parsed.get("command")
    reason = parsed.get("reason")
    if not isinstance(command, str) or command not in allowed_commands:
        raise OpenRouterError("OpenRouter selected an invalid reactor command.")
    if not isinstance(reason, str) or not reason.strip():
        raise OpenRouterError("OpenRouter reactor decision is missing a reason.")

    return ModelDecision(command=command, reason=reason.strip())


def build_reactor_tool_handlers(
    state: ReactorState,
    options: list[ReactorOption],
) -> dict[str, Any]:
    allowed_commands = {option.command for option in options}

    def get_turn_context(_: dict[str, Any]) -> dict[str, Any]:
        return {
            "prompt": build_decision_prompt(state, options),
            "allowed_commands": sorted(allowed_commands),
        }

    def validate_command_choice(arguments: dict[str, Any]) -> dict[str, Any]:
        command = str(arguments.get("command", "")).strip()
        return {
            "command": command,
            "is_safe": command in allowed_commands,
            "allowed_commands": sorted(allowed_commands),
        }

    return {
        "get_turn_context": get_turn_context,
        "validate_command_choice": validate_command_choice,
    }


def execute_reactor_tool_call(
    tool_call: ToolCall,
    handlers: dict[str, Any],
) -> dict[str, Any]:
    if tool_call.name not in handlers:
        raise OpenRouterError(f"Unknown reactor tool call: {tool_call.name!r}")
    result = handlers[tool_call.name](tool_call.arguments)
    return {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "name": tool_call.name,
        "content": json.dumps(result, ensure_ascii=False),
    }


def choose_command_with_openrouter(
    *,
    client: OpenRouterClient,
    state: ReactorState,
    options: list[ReactorOption],
) -> ModelDecision:
    allowed_commands = [option.command for option in options]
    handlers = build_reactor_tool_handlers(state, options)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": MODEL_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "Choose the best reactor move for this turn.",
        },
    ]
    for _ in range(MODEL_MAX_STEPS):
        completion = client.create_completion(messages, tools=REACTOR_TOOLS)
        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": completion.content or "",
        }
        if completion.tool_calls:
            assistant_message["tool_calls"] = [
                tool_call.to_message_dict() for tool_call in completion.tool_calls
            ]
            messages.append(assistant_message)
            for tool_call in completion.tool_calls:
                messages.append(execute_reactor_tool_call(tool_call, handlers))
            continue
        if not completion.content:
            raise OpenRouterError("OpenRouter returned no content for reactor move.")
        return parse_model_decision(
            completion.content,
            allowed_commands=set(allowed_commands),
        )
    raise OpenRouterError("OpenRouter tool calling did not finish for reactor move.")


def resolve_command_choice(
    options: list[ReactorOption],
    model_decision: ModelDecision | None,
) -> tuple[MoveCommand, dict[str, Any]]:
    if not options:
        raise ValueError("At least one safe reactor option is required.")

    best_option = min(options, key=lambda option: (option.remaining_steps, ACTION_PRIORITY.index(option.command)))
    by_command = {option.command: option for option in options}

    if model_decision is None:
        return best_option.command, {
            "source": "planner_fallback",
            "reason": "OpenRouter decision unavailable.",
            "fallback_command": best_option.command,
            "best_remaining_steps": best_option.remaining_steps,
        }

    selected_option = by_command.get(model_decision.command)
    if (
        selected_option is not None
        and selected_option.remaining_steps == best_option.remaining_steps
    ):
        return model_decision.command, {
            "source": "openrouter",
            "reason": model_decision.reason,
            "best_remaining_steps": best_option.remaining_steps,
        }

    return best_option.command, {
        "source": "planner_fallback",
        "reason": model_decision.reason,
        "requested_command": model_decision.command,
        "fallback_command": best_option.command,
        "best_remaining_steps": best_option.remaining_steps,
    }


def append_trace(
    trace: list[dict[str, Any]],
    *,
    command: ApiCommand,
    response: Mapping[str, Any],
    planned_path: list[MoveCommand] | None = None,
    predicted_state: ReactorState | None = None,
    safe_options: list[ReactorOption] | None = None,
    decision_meta: dict[str, Any] | None = None,
) -> None:
    entry: dict[str, Any] = {
        "command": command,
        "response": dict(response),
    }
    if planned_path is not None:
        entry["planned_path"] = list(planned_path)
    if predicted_state is not None:
        entry["predicted_state"] = predicted_state.as_dict()
        entry["prediction_matched_api"] = predicted_state.matches_api_payload(response)
    if safe_options is not None:
        entry["safe_options"] = [option.as_dict() for option in safe_options]
    if decision_meta is not None:
        entry["decision"] = dict(decision_meta)
    trace.append(entry)


def is_success_response(response: Mapping[str, Any]) -> bool:
    return bool(response.get("reached_goal")) or response.get("code") == 0 or bool(
        extract_flag(response)
    )


def main() -> int:
    configure_logging(name="reactor")
    args = parse_args()
    api_key = get_api_key()
    openrouter_client = build_openrouter_client(args.model)
    trace: list[dict[str, Any]] = []

    if args.reset_first:
        reset_response = send_command(api_key, "reset")
        append_trace(trace, command="reset", response=reset_response)
        logger.info("Reset response: {}", reset_response.get("message", "no message"))

    response = send_command(api_key, "start")
    append_trace(trace, command="start", response=response)
    write_json(LAST_RESPONSE_PATH, response)

    logger.info("Start response: {}", response.get("message", "no message"))

    if response.get("code") == -920:
        write_json(LAST_TRACE_PATH, trace)
        logger.error("Robot was crushed during initialization.")
        return 1

    for step_index in range(1, args.max_steps + 1):
        if is_success_response(response):
            break

        try:
            state = ReactorState.from_api_payload(response)
        except ValueError as exc:
            write_json(LAST_TRACE_PATH, trace)
            logger.error("Cannot parse reactor state: {}", exc)
            return 1

        options = build_safe_options(state)
        if not options:
            write_json(LAST_TRACE_PATH, trace)
            logger.error("No safe plan found from the current reactor state.")
            return 1

        plan = plan_commands(state) or []
        model_decision: ModelDecision | None = None
        try:
            model_decision = choose_command_with_openrouter(
                client=openrouter_client,
                state=state,
                options=options,
            )
        except OpenRouterError as exc:
            logger.warning("OpenRouter decision failed, using planner fallback: {}", exc)

        command, decision_meta = resolve_command_choice(options, model_decision)
        predicted_state = state.apply(command)
        logger.info(
            "Step {}: sending {} via {} (planned remaining steps: {}).",
            step_index,
            command,
            decision_meta["source"],
            len(plan),
        )

        response = send_command(api_key, command)
        append_trace(
            trace,
            command=command,
            response=response,
            planned_path=plan,
            predicted_state=predicted_state,
            safe_options=options,
            decision_meta=decision_meta,
        )
        write_json(LAST_RESPONSE_PATH, response)

        if response.get("code") == -920:
            write_json(LAST_TRACE_PATH, trace)
            logger.error("Robot was crushed after command {}.", command)
            return 1

        if not is_success_response(response) and predicted_state is not None:
            if not predicted_state.matches_api_payload(response):
                logger.warning("API state diverged from the local model after {}.", command)

        if is_success_response(response):
            break

    write_json(LAST_TRACE_PATH, trace)

    if not is_success_response(response):
        logger.error("Goal not reached within {} steps.", args.max_steps)
        return 1

    flag = extract_flag(response)
    if flag:
        logger.success("Flag: {}", flag)
    else:
        logger.success("Goal reached.")

    logger.info("Final response:\n{}", json.dumps(response, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
