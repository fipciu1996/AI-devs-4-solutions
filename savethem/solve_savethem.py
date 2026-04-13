"""Solve the AG3NTS `savethem` task by probing preview state and planning a route."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from heapq import heappop, heappush
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

REPO_ROOT_HINT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_HINT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_HINT))

from devs_utilities.bootstrap import bootstrap_repo
from devs_utilities.files import write_json
from devs_utilities.http import HttpRequestError, post_json, request_bytes
from devs_utilities.logging import configure_logging, logger as shared_logger
from devs_utilities.openrouter import (
    build_task_openrouter_client,
    build_task_site_name,
    OpenRouterClient,
    OpenRouterError,
    parse_json_content,
    ToolCall,
)
from devs_utilities.prompts import load_prompt_text
from devs_utilities.repo_env import get_env, get_int_env, get_llm_model


REPO_ROOT = bootstrap_repo(__file__)
logger = shared_logger.bind(component="savethem")


TASK_NAME = "savethem"
BASE_URL = get_env("AG3NTS_BASE_URL").rstrip("/")
VERIFY_URL = f"{BASE_URL}/verify"
TOOLSEARCH_URL = f"{BASE_URL}/api/toolsearch"
PREVIEW_URL = f"{BASE_URL}/savethem_backend.php"
PREVIEW_MISSING_CODE = -980
TREE_FUEL_PENALTY = 0.3
VERIFY_TIMEOUT_SECONDS = get_int_env("AG3NTS_TIMEOUT_SECONDS", 30) or 30
OPENROUTER_TIMEOUT_SECONDS = min(
    get_int_env("OPENROUTER_TIMEOUT_SECONDS", 120) or 120,
    20,
)
API_RETRY_ATTEMPTS = get_int_env("SAVETHEM_API_RETRY_ATTEMPTS", 4) or 4
API_RETRY_BASE_DELAY_SECONDS = float(get_int_env("SAVETHEM_API_RETRY_BASE_DELAY_SECONDS", 2) or 2)
API_RETRY_MAX_DELAY_SECONDS = float(get_int_env("SAVETHEM_API_RETRY_MAX_DELAY_SECONDS", 20) or 20)
OUTPUT_DIR = Path(__file__).resolve().parent
PREVIEW_PATH = OUTPUT_DIR / "last_preview_state.json"
VERIFY_RESPONSE_PATH = OUTPUT_DIR / "last_verify_response.json"
ANSWER_PATH = OUTPUT_DIR / "solution.json"
MODEL_ROUTE_PATH = OUTPUT_DIR / "model_route.json"
DEFAULT_PROBE_ANSWER = ["walk"]
EPSILON = 1e-9
MODEL_MAX_STEPS = 6
SYSTEM_PROMPT = load_prompt_text(__file__, "system_prompt.txt")

DIRS: tuple[tuple[int, int, str], ...] = (
    (-1, 0, "up"),
    (1, 0, "down"),
    (0, -1, "left"),
    (0, 1, "right"),
)


@dataclass(frozen=True, slots=True)
class VehicleSpec:
    name: str
    fuel_per_move: float
    food_per_move: float


@dataclass(frozen=True, slots=True)
class Position:
    row: int
    col: int


@dataclass(frozen=True, slots=True)
class PreviewState:
    terrain: tuple[tuple[str, ...], ...]
    start: Position
    goal: Position


@dataclass(frozen=True, slots=True)
class SearchState:
    row: int
    col: int
    mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Submit the computed answer to the AG3NTS verify endpoint.",
    )
    parser.add_argument(
        "--refresh-preview",
        action="store_true",
        help="Force a fresh probe before loading preview state.",
    )
    return parser.parse_args()


def build_payload(answer: list[str], *, api_key: str) -> dict[str, Any]:
    return {
        "apikey": api_key,
        "task": TASK_NAME,
        "answer": answer,
    }


def build_openrouter_client() -> OpenRouterClient:
    api_key = get_env("OPENROUTER_API_KEY")
    base_url = get_env("OPENROUTER_BASE_URL")
    model = get_llm_model("SAVETHEM_MODEL")
    if not api_key or not base_url:
        raise RuntimeError("Missing OPENROUTER_API_KEY or OPENROUTER_BASE_URL in .env.")

    return build_task_openrouter_client(
        __file__,
        api_key=api_key,
        base_url=base_url,
        model=model,
        task_name=TASK_NAME,
        timeout_seconds=OPENROUTER_TIMEOUT_SECONDS,
    )


def is_retryable_ag3nts_error(exc: HttpRequestError) -> bool:
    """Return True for transient AG3NTS failures worth retrying."""

    if exc.status_code in {408, 425, 429, 500, 502, 503, 504}:
        return True
    payload = exc.body_as_json()
    if isinstance(payload, dict):
        code = payload.get("code")
        if code in {-9999, -500}:
            return True
    return exc.status_code is None


def with_ag3nts_retry(action: str, operation: Any) -> Any:
    """Retry transient AG3NTS API calls with exponential backoff."""

    delay_seconds = max(0.1, API_RETRY_BASE_DELAY_SECONDS)
    last_error: HttpRequestError | None = None
    for attempt in range(1, API_RETRY_ATTEMPTS + 1):
        try:
            return operation()
        except HttpRequestError as exc:
            last_error = exc
            if attempt >= API_RETRY_ATTEMPTS or not is_retryable_ag3nts_error(exc):
                raise
            logger.warning(
                "{} transient failure on attempt {}/{}: {}. Retrying in {:.1f}s.",
                action,
                attempt,
                API_RETRY_ATTEMPTS,
                exc,
                delay_seconds,
            )
            time.sleep(delay_seconds)
            delay_seconds = min(API_RETRY_MAX_DELAY_SECONDS, delay_seconds * 2)
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"{action} retry loop ended unexpectedly.")


def call_tool(url: str, *, api_key: str, query: str) -> Any:
    return with_ag3nts_retry(
        f"Tool call `{query}`",
        lambda: post_json(
            url,
            {"apikey": api_key, "query": query},
            timeout_seconds=VERIFY_TIMEOUT_SECONDS,
            on_decode_error="raw_text",
        ),
    )


def discover_tool(api_key: str, query: str, expected_name: str) -> str:
    response = call_tool(TOOLSEARCH_URL, api_key=api_key, query=query)
    if not isinstance(response, dict):
        raise RuntimeError("Toolsearch returned an unexpected payload.")

    for tool in response.get("tools", []):
        if isinstance(tool, dict) and tool.get("name") == expected_name:
            return f"{BASE_URL}{tool['url']}"

    raise RuntimeError(f"Toolsearch did not return the required tool: {expected_name}")


def fetch_vehicle_specs(api_key: str) -> dict[str, VehicleSpec]:
    wehicles_url = discover_tool(
        api_key,
        query="vehicle fuel food transport",
        expected_name="wehicles",
    )
    specs: dict[str, VehicleSpec] = {}
    for vehicle_name in ("walk", "horse", "car", "rocket"):
        payload = call_tool(wehicles_url, api_key=api_key, query=vehicle_name)
        if not isinstance(payload, dict):
            raise RuntimeError(f"Vehicle payload for {vehicle_name} is invalid.")
        consumption = payload.get("consumption")
        if not isinstance(consumption, dict):
            raise RuntimeError(f"Vehicle payload for {vehicle_name} is missing consumption.")
        specs[vehicle_name] = VehicleSpec(
            name=vehicle_name,
            fuel_per_move=float(consumption["fuel"]),
            food_per_move=float(consumption["food"]),
        )
    return specs


def send_probe_answer(api_key: str) -> None:
    payload = build_payload(DEFAULT_PROBE_ANSWER, api_key=api_key)
    try:
        response = post_json(
            VERIFY_URL,
            payload,
            timeout_seconds=VERIFY_TIMEOUT_SECONDS,
            on_decode_error="raw_text",
        )
    except HttpRequestError as exc:
        response = exc.to_response_dict()
    write_json(VERIFY_RESPONSE_PATH, response)


def fetch_preview_state(api_key: str) -> dict[str, Any]:
    body = urlencode({"key": api_key}).encode("utf-8")
    raw = with_ag3nts_retry(
        "Fetch preview state",
        lambda: request_bytes(
            PREVIEW_URL,
            method="POST",
            data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout_seconds=VERIFY_TIMEOUT_SECONDS,
        ),
    )
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Preview returned an unexpected payload.")
    write_json(PREVIEW_PATH, payload)
    return payload


def ensure_preview_state(api_key: str, *, refresh_preview: bool) -> dict[str, Any]:
    if refresh_preview:
        send_probe_answer(api_key)

    try:
        payload = fetch_preview_state(api_key)
    except HttpRequestError as exc:
        payload = exc.to_response_dict()

    if payload.get("code") == PREVIEW_MISSING_CODE:
        logger.info("Preview missing, sending a probe route to initialize it.")
        send_probe_answer(api_key)
        payload = fetch_preview_state(api_key)
    return payload


def parse_position(payload: Any, *, key: str) -> Position:
    if not isinstance(payload, dict):
        raise RuntimeError(f"Missing {key} position in preview payload.")
    return Position(row=int(payload["row"]) - 1, col=int(payload["col"]) - 1)


def parse_preview_state(payload: dict[str, Any]) -> PreviewState:
    raw_map = payload.get("map")
    if not isinstance(raw_map, list) or not raw_map:
        raise RuntimeError("Preview payload does not contain a valid map.")

    terrain_rows: list[tuple[str, ...]] = []
    for row in raw_map:
        if not isinstance(row, list):
            raise RuntimeError("Preview payload contains a malformed map row.")
        terrain_rows.append(tuple(str(cell) for cell in row))

    return PreviewState(
        terrain=tuple(terrain_rows),
        start=parse_position(payload.get("start"), key="start"),
        goal=parse_position(payload.get("goal"), key="goal"),
    )


def preview_to_prompt(preview: PreviewState) -> str:
    return "\n".join("".join(row) for row in preview.terrain)


def task_context_payload(
    preview: PreviewState,
    specs: dict[str, VehicleSpec],
) -> dict[str, Any]:
    return {
        "map": preview_to_prompt(preview),
        "start": {"row": preview.start.row + 1, "col": preview.start.col + 1},
        "goal": {"row": preview.goal.row + 1, "col": preview.goal.col + 1},
        "resources": {"fuel": 10, "food": 10},
        "vehicles": {
            name: {
                "fuel_per_move": spec.fuel_per_move,
                "food_per_move": spec.food_per_move,
            }
            for name, spec in specs.items()
        },
        "rules": {
            "initial_mode_only_once": True,
            "dismount_switches_to_walk": True,
            "rocks_impassable": True,
            "car_and_rocket_cannot_enter_water": True,
            "tree_extra_fuel_non_walk": TREE_FUEL_PENALTY,
            "goal_requires_fuel_and_food_above_zero": True,
            "objective": "minimize food use first, then fuel use",
        },
    }


def can_enter(cell: str, *, mode: str) -> bool:
    if cell == "R":
        return False
    if cell == "W" and mode in {"car", "rocket"}:
        return False
    return True


def move_cost(cell: str, *, mode: str, specs: dict[str, VehicleSpec]) -> tuple[float, float]:
    spec = specs[mode]
    fuel = spec.fuel_per_move
    food = spec.food_per_move
    if cell == "T" and mode != "walk":
        fuel += TREE_FUEL_PENALTY
    return fuel, food


def within_budget(*, fuel_used: float, food_used: float) -> bool:
    remaining_fuel = 10.0 - fuel_used
    remaining_food = 10.0 - food_used
    return remaining_fuel > EPSILON and remaining_food > EPSILON


def simulate_route(
    preview: PreviewState,
    specs: dict[str, VehicleSpec],
    answer: list[str],
) -> tuple[bool, str, float, float]:
    if not answer:
        return False, "Empty route.", 10.0, 10.0

    mode = answer[0]
    if mode not in specs:
        return False, f"Unknown initial mode: {mode}", 10.0, 10.0

    row = preview.start.row
    col = preview.start.col
    fuel_used = 0.0
    food_used = 0.0

    for operation in answer[1:]:
        if operation == "dismount":
            if mode == "walk":
                return False, "Cannot dismount while already walking.", 10.0 - fuel_used, 10.0 - food_used
            mode = "walk"
            continue

        move = next((item for item in DIRS if item[2] == operation), None)
        if move is None:
            return False, f"Unknown operation: {operation}", 10.0 - fuel_used, 10.0 - food_used

        next_row = row + move[0]
        next_col = col + move[1]
        if not (0 <= next_row < len(preview.terrain) and 0 <= next_col < len(preview.terrain[0])):
            return False, "Route leaves the map.", 10.0 - fuel_used, 10.0 - food_used

        cell = preview.terrain[next_row][next_col]
        if not can_enter(cell, mode=mode):
            return False, f"Mode {mode} cannot enter {cell}.", 10.0 - fuel_used, 10.0 - food_used

        step_fuel, step_food = move_cost(cell, mode=mode, specs=specs)
        fuel_used += step_fuel
        food_used += step_food
        if not within_budget(fuel_used=fuel_used, food_used=food_used):
            return False, "Route exceeds resource budget.", max(0.0, 10.0 - fuel_used), max(0.0, 10.0 - food_used)

        row = next_row
        col = next_col

    reached_goal = row == preview.goal.row and col == preview.goal.col
    return reached_goal, "Goal reached." if reached_goal else "Route does not reach the goal.", 10.0 - fuel_used, 10.0 - food_used


def normalize_answer(answer: Any) -> list[str]:
    if not isinstance(answer, list):
        raise OpenRouterError("Answer must be a list of strings.")
    normalized: list[str] = []
    for item in answer:
        if not isinstance(item, str):
            raise OpenRouterError("Each answer item must be a string.")
        stripped = item.strip()
        if stripped:
            normalized.append(stripped)
    if not normalized:
        raise OpenRouterError("Answer cannot be empty.")
    return normalized


def find_best_route(
    preview: PreviewState,
    specs: dict[str, VehicleSpec],
) -> list[str]:
    height = len(preview.terrain)
    width = len(preview.terrain[0])
    start = preview.start
    goal = preview.goal

    queue: list[tuple[float, float, int, int, SearchState, list[str]]] = []
    best_costs: dict[SearchState, tuple[float, float, int]] = {}
    push_order = 0

    for mode in ("walk", "horse", "car", "rocket"):
        state = SearchState(row=start.row, col=start.col, mode=mode)
        initial_path = [mode]
        score = (0.0, 0.0, len(initial_path))
        best_costs[state] = score
        heappush(queue, (0.0, 0.0, len(initial_path), push_order, state, initial_path))
        push_order += 1

    while queue:
        food_used, fuel_used, operations, _, state, path = heappop(queue)
        current_score = (food_used, fuel_used, operations)
        if best_costs.get(state) != current_score:
            continue

        if state.row == goal.row and state.col == goal.col:
            return path

        current_mode = state.mode
        current_cell = preview.terrain[state.row][state.col]
        if current_mode != "walk":
            dismounted_state = SearchState(row=state.row, col=state.col, mode="walk")
            dismounted_score = (food_used, fuel_used, operations + 1)
            if dismounted_score < best_costs.get(
                dismounted_state,
                (float("inf"), float("inf"), sys.maxsize),
            ):
                best_costs[dismounted_state] = dismounted_score
                heappush(
                    queue,
                    (
                        food_used,
                        fuel_used,
                        operations + 1,
                        push_order,
                        dismounted_state,
                        [*path, "dismount"],
                    ),
                )
                push_order += 1

        for row_delta, col_delta, step_name in DIRS:
            next_row = state.row + row_delta
            next_col = state.col + col_delta
            if not (0 <= next_row < height and 0 <= next_col < width):
                continue

            next_cell = preview.terrain[next_row][next_col]
            if not can_enter(next_cell, mode=current_mode):
                continue

            step_fuel, step_food = move_cost(next_cell, mode=current_mode, specs=specs)
            next_fuel = fuel_used + step_fuel
            next_food = food_used + step_food
            if not within_budget(fuel_used=next_fuel, food_used=next_food):
                continue

            next_state = SearchState(row=next_row, col=next_col, mode=current_mode)
            next_score = (next_food, next_fuel, operations + 1)
            if next_score >= best_costs.get(
                next_state,
                (float("inf"), float("inf"), sys.maxsize),
            ):
                continue

            best_costs[next_state] = next_score
            heappush(
                queue,
                (
                    next_food,
                    next_fuel,
                    operations + 1,
                    push_order,
                    next_state,
                    [*path, step_name],
                ),
            )
            push_order += 1

    raise RuntimeError("No feasible route reaches the goal within the resource limits.")


SAVE_THEM_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_task_context",
            "description": "Return the map, start, goal, resource limits, vehicle specs and movement rules.",
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
            "name": "get_deterministic_candidate",
            "description": "Return the best route found by the deterministic solver.",
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
            "name": "validate_candidate_route",
            "description": "Validate a proposed route and report whether it reaches the goal and how many resources remain.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Route beginning with vehicle name followed by moves and optional dismount.",
                    }
                },
                "required": ["answer"],
                "additionalProperties": False,
            },
        },
    },
]


def build_model_tool_handlers(
    preview: PreviewState,
    specs: dict[str, VehicleSpec],
    deterministic_answer: list[str],
) -> dict[str, Any]:
    def get_task_context(_: dict[str, Any]) -> dict[str, Any]:
        return task_context_payload(preview, specs)

    def get_deterministic_candidate(_: dict[str, Any]) -> dict[str, Any]:
        reached_goal, message, remaining_fuel, remaining_food = simulate_route(
            preview,
            specs,
            deterministic_answer,
        )
        return {
            "answer": deterministic_answer,
            "reached_goal": reached_goal,
            "message": message,
            "remaining_fuel": remaining_fuel,
            "remaining_food": remaining_food,
        }

    def validate_candidate_route(arguments: dict[str, Any]) -> dict[str, Any]:
        answer = normalize_answer(arguments.get("answer"))
        reached_goal, message, remaining_fuel, remaining_food = simulate_route(
            preview,
            specs,
            answer,
        )
        return {
            "answer": answer,
            "reached_goal": reached_goal,
            "message": message,
            "remaining_fuel": remaining_fuel,
            "remaining_food": remaining_food,
        }

    return {
        "get_task_context": get_task_context,
        "get_deterministic_candidate": get_deterministic_candidate,
        "validate_candidate_route": validate_candidate_route,
    }


def execute_model_tool_call(
    tool_call: ToolCall,
    handlers: dict[str, Any],
) -> dict[str, Any]:
    if tool_call.name not in handlers:
        raise OpenRouterError(f"Model called unknown tool: {tool_call.name!r}")

    result = handlers[tool_call.name](tool_call.arguments)
    return {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "name": tool_call.name,
        "content": json.dumps(result, ensure_ascii=False),
    }


def choose_route_with_openrouter(
    preview: PreviewState,
    specs: dict[str, VehicleSpec],
    deterministic_answer: list[str],
) -> list[str]:
    client = build_openrouter_client()
    handlers = build_model_tool_handlers(preview, specs, deterministic_answer)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "Find the best valid route for the current savethem map.",
        },
    ]

    for _ in range(MODEL_MAX_STEPS):
        completion = client.create_completion(messages, tools=SAVE_THEM_TOOLS)
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
                messages.append(execute_model_tool_call(tool_call, handlers))
            continue

        if not completion.content:
            raise OpenRouterError("OpenRouter returned neither content nor tool calls.")

        payload = parse_json_content(completion.content)
        write_json(MODEL_ROUTE_PATH, payload)
        if isinstance(payload, list):
            return normalize_answer(payload)
        if isinstance(payload, dict):
            return normalize_answer(payload.get("answer"))
        raise OpenRouterError("OpenRouter returned an unsupported route payload.")

    raise OpenRouterError("OpenRouter tool calling did not finish within the step limit.")


def submit_answer(answer: list[str], *, api_key: str) -> Any:
    payload = build_payload(answer, api_key=api_key)
    response = with_ag3nts_retry(
        "Submit savethem answer",
        lambda: post_json(
            VERIFY_URL,
            payload,
            timeout_seconds=VERIFY_TIMEOUT_SECONDS,
            on_decode_error="raw_text",
        ),
    )
    write_json(VERIFY_RESPONSE_PATH, response)
    return response


def main() -> int:
    configure_logging(name="savethem")
    args = parse_args()
    api_key = get_env("AG3NTS_API_KEY")
    if not api_key:
        logger.error("Missing AG3NTS_API_KEY in .env.")
        return 1

    specs = fetch_vehicle_specs(api_key)
    preview_payload = ensure_preview_state(api_key, refresh_preview=args.refresh_preview)
    preview = parse_preview_state(preview_payload)
    deterministic_answer = find_best_route(preview, specs)
    answer = deterministic_answer
    try:
        model_answer = choose_route_with_openrouter(preview, specs, deterministic_answer)
        is_valid, message, remaining_fuel, remaining_food = simulate_route(
            preview,
            specs,
            model_answer,
        )
        deterministic_valid, _, deterministic_fuel, deterministic_food = simulate_route(
            preview,
            specs,
            deterministic_answer,
        )
        if (
            is_valid
            and deterministic_valid
            and (
                remaining_food > deterministic_food + EPSILON
                or (
                    abs(remaining_food - deterministic_food) <= EPSILON
                    and remaining_fuel >= deterministic_fuel - EPSILON
                )
            )
        ):
            answer = model_answer
            logger.info("Using OpenRouter-selected route.")
        else:
            logger.warning(
                "OpenRouter route rejected: {} Remaining fuel {:.1f}, food {:.1f}. Using deterministic fallback.",
                message,
                remaining_fuel,
                remaining_food,
            )
    except (OpenRouterError, RuntimeError, json.JSONDecodeError) as exc:
        logger.warning("OpenRouter route selection failed: {}. Using deterministic fallback.", exc)

    solution_payload = build_payload(answer, api_key=api_key)
    write_json(ANSWER_PATH, solution_payload)
    logger.info("Computed route: {}", answer)

    if not args.verify:
        return 0

    try:
        response = submit_answer(answer, api_key=api_key)
    except HttpRequestError as exc:
        write_json(VERIFY_RESPONSE_PATH, exc.to_response_dict())
        logger.error("Verify failed: {}", exc)
        return 1

    logger.success("Verify response: {}", response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
