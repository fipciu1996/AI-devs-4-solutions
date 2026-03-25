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
from repo_env import get_env, get_int_env


REPO_ROOT = bootstrap_repo(__file__)
logger = shared_logger.bind(component="reactor")


TASK_NAME = "reactor"
BOARD_COLUMNS = 7
GOAL_COLUMN = 7
DEFAULT_MAX_STEPS = get_int_env("REACTOR_MAX_STEPS", 64) or 64
VERIFY_TIMEOUT_SECONDS = get_int_env("AG3NTS_TIMEOUT_SECONDS", 30) or 30
OUTPUT_DIR = Path(__file__).resolve().parent
LAST_RESPONSE_PATH = OUTPUT_DIR / "last_verify_response.json"
LAST_TRACE_PATH = OUTPUT_DIR / "last_trace.json"

MoveCommand = Literal["left", "wait", "right"]
ApiCommand = Literal["start", "reset", "left", "wait", "right"]
Direction = Literal["up", "down"]

ACTION_PRIORITY: tuple[MoveCommand, ...] = ("right", "wait", "left")


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


def append_trace(
    trace: list[dict[str, Any]],
    *,
    command: ApiCommand,
    response: Mapping[str, Any],
    planned_path: list[MoveCommand] | None = None,
    predicted_state: ReactorState | None = None,
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
    trace.append(entry)


def is_success_response(response: Mapping[str, Any]) -> bool:
    return bool(response.get("reached_goal")) or response.get("code") == 0 or bool(
        extract_flag(response)
    )


def main() -> int:
    configure_logging(name="reactor")
    args = parse_args()
    api_key = get_api_key()
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

        plan = plan_commands(state)
        if not plan:
            write_json(LAST_TRACE_PATH, trace)
            logger.error("No safe plan found from the current reactor state.")
            return 1

        command = plan[0]
        predicted_state = state.apply(command)
        logger.info(
            "Step {}: sending {} (planned remaining steps: {}).",
            step_index,
            command,
            len(plan),
        )

        response = send_command(api_key, command)
        append_trace(
            trace,
            command=command,
            response=response,
            planned_path=plan,
            predicted_state=predicted_state,
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
