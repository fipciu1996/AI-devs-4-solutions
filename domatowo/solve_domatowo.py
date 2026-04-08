"""Solve the AG3NTS `domatowo` task with a hybrid OpenRouter mission planner."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import deque
from dataclasses import asdict, dataclass
from itertools import permutations
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

REPO_ROOT_HINT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_HINT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_HINT))

from devs_utilities.ag3nts import AG3NTS_VERIFY_URL, submit_task_answer
from devs_utilities.bootstrap import bootstrap_repo
from devs_utilities.files import write_json
from devs_utilities.http import HttpRequestError
from devs_utilities.logging import configure_logging, logger as shared_logger
from devs_utilities.openrouter import (
    ChatCompletionResult,
    OpenRouterClient,
    OpenRouterError,
    ToolCall,
    build_task_openrouter_client,
)
from devs_utilities.repo_env import get_env, get_int_env, get_llm_model, get_optional_env


REPO_ROOT = bootstrap_repo(__file__)
logger = shared_logger.bind(component="domatowo")


TASK_NAME = "domatowo"
INTERCEPTED_SIGNAL = (
    "Przezylem. Bomby zniszczyly miasto. Zolnierze tu byli, szukali surowcow, "
    "zabrali rope. Teraz jest pusto. Mam bron, jestem ranny. Ukrylem sie w jednym "
    "z najwyzszych blokow. Nie mam jedzenia. Pomocy."
)
VERIFY_TIMEOUT_SECONDS = get_int_env("AG3NTS_TIMEOUT_SECONDS", 30) or 30
DEFAULT_MODEL = get_llm_model("DOMATOWO_MODEL")
OUTPUT_DIR = Path(__file__).resolve().parent
MAP_PATH = OUTPUT_DIR / "last_map.json"
PLAN_PATH = OUTPUT_DIR / "last_plan.json"
SEARCH_TRACE_PATH = OUTPUT_DIR / "search_trace.json"
VERIFY_RESPONSE_PATH = OUTPUT_DIR / "last_verify_response.json"
SIDE_QUEST_PATH = OUTPUT_DIR / "last_side_quest.json"
RESULT_PATH = OUTPUT_DIR / "last_result.json"

SCOUTS_PER_TRANSPORTER = 1
PLANNER_MAX_STEPS = 4
LOG_ANALYST_MAX_STEPS = 3
CHURCH_SYMBOL = "KS"
CHURCH_SECRET_FIELD = "G8"
CHURCH_URL_KEY = "froynv"
SIDE_SECRET_ANSWER = "REDACTED"
SIDE_SECRET_HASH = hashlib.md5(SIDE_SECRET_ANSWER.encode("ascii")).hexdigest().upper()
RETRY_DEFAULT_ATTEMPTS = 10
T = TypeVar("T")

EMPTY_TILES = {"empty", "parking", "field"}
EMPTY_MARKERS = (
    "nie ma",
    "brak ludzi",
    "brak obecnosci ludzkiej",
    "brak ludzi w tym sektorze",
    "brak celu",
    "brak celu w srodku",
    "nikogo",
    "pusto",
    "pokoj pusty",
    "pomieszczenie puste",
    "przeszukanie negatywne",
    "przeszukanie nic nie wykazalo",
    "cel poza tym miejscem",
    "nie natrafilismy",
    "nie natrafilem na nikogo",
    "nie odnaleziono celu",
    "nie ma tu kogo szukac",
    "nie znaleziono zadnej osoby",
    "nie znalezlismy nikogo",
    "nie odnotowano czlowieka",
    "pomieszczenie martwe",
    "miejsce opuszczone",
    "nic tu nie ma",
    "nic nie znaleziono",
    "nic ciekawego",
    "nikogo nie stwierdzono",
    "nikogo nie widac",
    "nikt tu nie przebywa",
    "wyczyszczone z ludzi",
    "tylko smieci",
    "nic do zgloszenia",
    "nie ma kontaktu",
    "nikogo nie bylo",
)
FOUND_MARKERS = (
    "osoba odnaleziona",
    "mamy osobe",
    "czlowiek odnalezion",
    "kontakt z czlowiekiem",
    "znalezlismy czlowieka",
    "znalazlem czlowieka",
    "mam kontakt z czlowiekiem",
    "mezczyzna okolo",
    "mezczyzna mniej wiecej",
    "kobieta okolo",
    "ranny",
    "odwodniony",
    "zyje",
    "zywa",
)
POLISH_ASCII_MAP = str.maketrans(
    {
        "a": "a",
        "c": "c",
        "e": "e",
        "l": "l",
        "n": "n",
        "o": "o",
        "s": "s",
        "z": "z",
        "A": "A",
        "C": "C",
        "E": "E",
        "L": "L",
        "N": "N",
        "O": "O",
        "S": "S",
        "Z": "Z",
        "ą": "a",
        "ć": "c",
        "ę": "e",
        "ł": "l",
        "ń": "n",
        "ó": "o",
        "ś": "s",
        "ż": "z",
        "ź": "z",
        "Ą": "A",
        "Ć": "C",
        "Ę": "E",
        "Ł": "L",
        "Ń": "N",
        "Ó": "O",
        "Ś": "S",
        "Ż": "Z",
        "Ź": "Z",
    }
)
ORTHOGONAL_STEPS: tuple[tuple[int, int], ...] = (
    (0, -1),
    (1, 0),
    (0, 1),
    (-1, 0),
)


@dataclass(frozen=True, slots=True)
class Coordinate:
    x: int
    y: int

    @property
    def label(self) -> str:
        return f"{chr(ord('A') + self.x)}{self.y + 1}"


@dataclass(frozen=True, slots=True)
class MapState:
    size: int
    grid: tuple[tuple[str, ...], ...]
    symbol_by_tile: dict[str, str]

    def tile_at(self, coordinate: Coordinate) -> str:
        return self.grid[coordinate.y][coordinate.x]

    def symbol_at(self, coordinate: Coordinate) -> str:
        return self.symbol_by_tile[self.tile_at(coordinate)]

    def in_bounds(self, coordinate: Coordinate) -> bool:
        return 0 <= coordinate.x < self.size and 0 <= coordinate.y < self.size


@dataclass(frozen=True, slots=True)
class Cluster:
    cluster_id: str
    region: str
    targets: tuple[str, ...]
    anchor: str
    predicted_dismount: str


@dataclass(frozen=True, slots=True)
class PlannerDecision:
    cluster_order: tuple[str, ...]
    field_order: dict[str, tuple[str, ...]]
    reason: str
    source: str


@dataclass(frozen=True, slots=True)
class InspectLog:
    scout_id: str
    field: str
    message: str


@dataclass(frozen=True, slots=True)
class TransportAssignment:
    cluster: Cluster
    transporter_id: str
    scout_id: str
    scout_position: str


@dataclass(frozen=True, slots=True)
class LogAssessment:
    result: str
    reason: str
    source: str


@dataclass(frozen=True, slots=True)
class SideQuestPlan:
    anchor: str
    predicted_dismount: str
    secret_field: str


@dataclass(frozen=True, slots=True)
class SideQuestResult:
    anchor: str
    scout_start: str
    field: str
    raw_message: str
    decoded_message: str
    question: str
    encrypted_prompt: str
    media_url: str
    corrected_audio_hash: str
    answer: str


class DomatowoApiClient:
    """Stateful wrapper around the AG3NTS verify endpoint."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self.history: list[dict[str, Any]] = []

    def call(self, answer: dict[str, Any]) -> dict[str, Any]:
        response = submit_task_answer(
            AG3NTS_VERIFY_URL,
            api_key=self._api_key,
            task=TASK_NAME,
            answer=answer,
            timeout_seconds=VERIFY_TIMEOUT_SECONDS,
        )
        if not isinstance(response, dict):
            raise RuntimeError(f"Unexpected response payload: {response!r}")
        self.history.append({"answer": answer, "response": response})
        return response

    def reset(self) -> dict[str, Any]:
        return self.call({"action": "reset"})

    def get_map(self) -> dict[str, Any]:
        return self.call({"action": "getMap"})

    def search_symbol(self, symbol: str) -> dict[str, Any]:
        return self.call({"action": "searchSymbol", "symbol": symbol})

    def get_objects(self) -> list[dict[str, str]]:
        response = self.call({"action": "getObjects"})
        raw_objects = response.get("objects")
        if not isinstance(raw_objects, list):
            raise RuntimeError("getObjects did not return a valid objects list.")
        objects: list[dict[str, str]] = []
        for item in raw_objects:
            if not isinstance(item, dict):
                continue
            object_id = str(item.get("id", ""))
            typ = str(item.get("typ") or item.get("type") or "")
            position = str(item.get("position", ""))
            if object_id and typ and position:
                objects.append({"id": object_id, "typ": typ, "position": position})
        return objects

    def get_logs(self) -> list[InspectLog]:
        response = self.call({"action": "getLogs"})
        raw_logs = response.get("logs")
        if not isinstance(raw_logs, list):
            raise RuntimeError("getLogs did not return a valid logs list.")
        logs: list[InspectLog] = []
        for item in raw_logs:
            if not isinstance(item, dict):
                continue
            scout_id = str(item.get("scout", ""))
            field = str(item.get("field", ""))
            message = str(item.get("msg", ""))
            if scout_id and field and message:
                logs.append(InspectLog(scout_id=scout_id, field=field, message=message))
        return logs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Execute the live mission.")
    parser.add_argument("--reset", action="store_true", help="Force-reset the board first.")
    parser.add_argument("--skip-model", action="store_true", help="Disable OpenRouter agents.")
    parser.add_argument("--model", default=None, help=f"Override the OpenRouter model. Default: {DEFAULT_MODEL}.")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=RETRY_DEFAULT_ATTEMPTS,
        help=f"Retry the live mission up to this many times. Default: {RETRY_DEFAULT_ATTEMPTS}.",
    )
    args = parser.parse_args()
    if args.max_attempts < 1:
        parser.error("--max-attempts must be at least 1.")
    return args


def normalize_text(text: str) -> str:
    translated = text.translate(POLISH_ASCII_MAP)
    decomposed = unicodedata.normalize("NFKD", translated)
    ascii_only = decomposed.encode("ascii", "ignore").decode("ascii")
    return ascii_only.casefold()


def coord_from_label(label: str) -> Coordinate:
    stripped = label.strip().upper()
    return Coordinate(x=ord(stripped[0]) - ord("A"), y=int(stripped[1:]) - 1)


def manhattan_distance(start: str, end: str) -> int:
    left = coord_from_label(start)
    right = coord_from_label(end)
    return abs(left.x - right.x) + abs(left.y - right.y)


def build_map_state(payload: dict[str, Any]) -> MapState:
    raw_map = payload.get("map")
    if not isinstance(raw_map, dict):
        raise RuntimeError("Map payload is missing the `map` object.")
    raw_tiles = raw_map.get("tiles")
    raw_grid = raw_map.get("grid")
    if not isinstance(raw_tiles, dict) or not isinstance(raw_grid, list):
        raise RuntimeError("Map payload is malformed.")
    symbol_by_tile: dict[str, str] = {}
    for tile_name, details in raw_tiles.items():
        if isinstance(details, dict) and isinstance(details.get("symbol"), str):
            symbol_by_tile[str(tile_name)] = str(details["symbol"])
    grid_rows = [tuple(str(cell) for cell in row) for row in raw_grid if isinstance(row, list)]
    return MapState(size=int(raw_map["size"]), grid=tuple(grid_rows), symbol_by_tile=symbol_by_tile)


def position_list_from_search(payload: dict[str, Any]) -> tuple[str, ...]:
    found = payload.get("found")
    if not isinstance(found, list):
        raise RuntimeError("searchSymbol did not return a valid `found` list.")
    positions = [
        str(item["position"])
        for item in found
        if isinstance(item, dict) and isinstance(item.get("position"), str)
    ]
    return tuple(sorted(positions, key=lambda label: (coord_from_label(label).y, coord_from_label(label).x)))


def positions_for_symbol(map_state: MapState, symbol: str) -> tuple[str, ...]:
    positions = [
        Coordinate(x=x, y=y).label
        for y, row in enumerate(map_state.grid)
        for x, _ in enumerate(row)
        if map_state.symbol_at(Coordinate(x=x, y=y)) == symbol
    ]
    return tuple(sorted(positions, key=lambda label: (coord_from_label(label).y, coord_from_label(label).x)))


def group_adjacent_positions(labels: Iterable[str]) -> list[tuple[str, ...]]:
    remaining = {coord_from_label(label).label for label in labels}
    groups: list[tuple[str, ...]] = []
    while remaining:
        start = min(remaining, key=lambda label: (coord_from_label(label).y, coord_from_label(label).x))
        stack = [start]
        component: list[str] = []
        remaining.remove(start)
        while stack:
            current = stack.pop()
            component.append(current)
            coord = coord_from_label(current)
            for dx, dy in ORTHOGONAL_STEPS:
                neighbour = Coordinate(coord.x + dx, coord.y + dy).label
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    stack.append(neighbour)
        groups.append(tuple(sorted(component, key=lambda label: (coord_from_label(label).y, coord_from_label(label).x))))
    return groups


def coordinate_neighbors(label: str, map_state: MapState) -> list[str]:
    neighbours: list[str] = []
    coord = coord_from_label(label)
    for dx, dy in ORTHOGONAL_STEPS:
        candidate = Coordinate(coord.x + dx, coord.y + dy)
        if map_state.in_bounds(candidate):
            neighbours.append(candidate.label)
    return neighbours


def predicted_dismount(anchor: str, map_state: MapState) -> str:
    coord = coord_from_label(anchor)
    for dx, dy in ORTHOGONAL_STEPS:
        candidate = Coordinate(coord.x + dx, coord.y + dy)
        if map_state.in_bounds(candidate):
            return candidate.label
    raise RuntimeError(f"Anchor {anchor} does not have an adjacent dismount tile.")


def tile_penalty(tile_name: str) -> int:
    if tile_name in EMPTY_TILES:
        return 0
    if tile_name == "tree":
        return 1
    return 2


def brute_force_route(start: str, targets: Iterable[str]) -> tuple[tuple[str, ...], int]:
    target_list = list(targets)
    if not target_list:
        return tuple(), 0
    best_order: tuple[str, ...] | None = None
    best_cost = sys.maxsize
    for permutation in permutations(target_list):
        cost = manhattan_distance(start, permutation[0])
        for left, right in zip(permutation, permutation[1:]):
            cost += manhattan_distance(left, right)
        if cost < best_cost:
            best_cost = cost
            best_order = tuple(permutation)
    if best_order is None:
        raise RuntimeError("Unable to compute a route through the cluster.")
    return best_order, best_cost


def choose_anchor_for_cluster(cluster_targets: tuple[str, ...], map_state: MapState) -> tuple[str, str]:
    road_positions = [
        Coordinate(x=x, y=y).label
        for y, row in enumerate(map_state.grid)
        for x, tile_name in enumerate(row)
        if tile_name == "road"
    ]
    best_anchor = ""
    best_dismount = ""
    best_score: tuple[int, int, int, int, int] | None = None
    for road in road_positions:
        predicted_start = predicted_dismount(road, map_state)
        _, route_steps = brute_force_route(predicted_start, cluster_targets)
        predicted_tile = map_state.tile_at(coord_from_label(predicted_start))
        road_to_cluster = min(manhattan_distance(road, target) for target in cluster_targets)
        road_coord = coord_from_label(road)
        score = (
            route_steps,
            tile_penalty(predicted_tile),
            road_to_cluster,
            road_coord.y,
            road_coord.x,
        )
        if best_score is None or score < best_score:
            best_score = score
            best_anchor = road
            best_dismount = predicted_start
    if not best_anchor or not best_dismount:
        raise RuntimeError("Unable to choose an anchor for the cluster.")
    return best_anchor, best_dismount


def describe_cluster_region(targets: tuple[str, ...], map_size: int) -> str:
    coordinates = [coord_from_label(label) for label in targets]
    average_x = sum(coord.x for coord in coordinates) / len(coordinates)
    average_y = sum(coord.y for coord in coordinates) / len(coordinates)
    if average_y <= map_size / 3:
        return "north"
    if average_x < map_size / 2:
        return "southwest"
    return "southeast"


def build_clusters(map_state: MapState, block_positions: tuple[str, ...]) -> list[Cluster]:
    clusters: list[Cluster] = []
    for index, group in enumerate(group_adjacent_positions(block_positions), start=1):
        anchor, predicted = choose_anchor_for_cluster(group, map_state)
        clusters.append(
            Cluster(
                cluster_id=f"cluster_{index}",
                region=describe_cluster_region(group, map_state.size),
                targets=group,
                anchor=anchor,
                predicted_dismount=predicted,
            )
        )
    return clusters


def build_side_quest_plan(map_state: MapState) -> SideQuestPlan:
    church_positions = positions_for_symbol(map_state, CHURCH_SYMBOL)
    if CHURCH_SECRET_FIELD not in church_positions:
        raise RuntimeError(f"Missing church secret field {CHURCH_SECRET_FIELD} on the map.")
    anchor, predicted_dismount_start = choose_anchor_for_cluster(church_positions, map_state)
    return SideQuestPlan(
        anchor=anchor,
        predicted_dismount=predicted_dismount_start,
        secret_field=CHURCH_SECRET_FIELD,
    )


def build_cluster_context(cluster: Cluster, map_state: MapState) -> dict[str, Any]:
    neighbour_symbols: dict[str, str] = {}
    for target in cluster.targets:
        for neighbour in coordinate_neighbors(target, map_state):
            if neighbour in cluster.targets:
                continue
            neighbour_symbols[neighbour] = map_state.symbol_at(coord_from_label(neighbour))
    return {
        "cluster_id": cluster.cluster_id,
        "region": cluster.region,
        "targets": list(cluster.targets),
        "anchor": cluster.anchor,
        "predicted_dismount": cluster.predicted_dismount,
        "adjacent_symbols": [
            {"position": position, "symbol": symbol}
            for position, symbol in sorted(neighbour_symbols.items())
        ],
    }


def normalize_json_payload(payload: str) -> dict[str, Any]:
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise OpenRouterError("OpenRouter returned invalid JSON.") from exc
    if not isinstance(parsed, dict):
        raise OpenRouterError("OpenRouter returned a non-object JSON payload.")
    return parsed


def build_openrouter_client(model_override: str | None) -> OpenRouterClient | None:
    api_key = get_optional_env("OPENROUTER_API_KEY") or get_optional_env("LLM_API_KEY")
    base_url = get_optional_env("OPENROUTER_BASE_URL") or get_optional_env("LLM_BASE_URL")
    if not api_key or not base_url:
        return None
    model = model_override or DEFAULT_MODEL
    configured_timeout = get_int_env("OPENROUTER_TIMEOUT_SECONDS", 120) or 120
    timeout_seconds = float(max(configured_timeout, 1))
    return build_task_openrouter_client(
        __file__,
        api_key=api_key,
        base_url=base_url,
        model=model,
        task_name=TASK_NAME,
        timeout_seconds=timeout_seconds,
    )


PLANNER_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_operation_context",
            "description": "Return the intercepted signal, budget rules and current B3 clusters.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_default_plan",
            "description": "Return the deterministic fallback plan for the mission.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_plan",
            "description": "Check whether cluster_order and field_order cover every candidate exactly once.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cluster_order": {"type": "array", "items": {"type": "string"}},
                    "field_order": {
                        "type": "object",
                        "additionalProperties": {"type": "array", "items": {"type": "string"}},
                    },
                },
                "required": ["cluster_order", "field_order"],
                "additionalProperties": False,
            },
        },
    },
]


LOG_ANALYST_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_log_context",
            "description": "Return the latest inspect log entry that needs classification.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_result",
            "description": "Validate the proposed log classification result.",
            "parameters": {
                "type": "object",
                "properties": {"result": {"type": "string"}},
                "required": ["result"],
                "additionalProperties": False,
            },
        },
    },
]


def execute_tool_call(tool_call: ToolCall, handlers: dict[str, Any]) -> dict[str, Any]:
    if tool_call.name not in handlers:
        raise OpenRouterError(f"Unknown tool requested by OpenRouter: {tool_call.name}")
    result = handlers[tool_call.name](tool_call.arguments)
    return {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "name": tool_call.name,
        "content": json.dumps(result, ensure_ascii=False),
    }


def run_tool_conversation(
    client: OpenRouterClient,
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    handlers: dict[str, Any],
    max_steps: int,
) -> ChatCompletionResult:
    for _ in range(max_steps):
        completion = client.create_completion(messages, tools=tools)
        assistant_message: dict[str, Any] = {"role": "assistant", "content": completion.content or ""}
        if completion.tool_calls:
            assistant_message["tool_calls"] = [tool_call.to_message_dict() for tool_call in completion.tool_calls]
            messages.append(assistant_message)
            for tool_call in completion.tool_calls:
                messages.append(execute_tool_call(tool_call, handlers))
            continue
        if completion.content:
            return completion
        raise OpenRouterError("OpenRouter returned neither content nor tool calls.")
    raise OpenRouterError("OpenRouter did not finish the tool workflow in time.")


def deterministic_planner(clusters: list[Cluster]) -> PlannerDecision:
    region_priority = {"north": 0, "southeast": 1, "southwest": 2}
    ordered_clusters = tuple(
        cluster.cluster_id
        for cluster in sorted(
            clusters,
            key=lambda cluster: (
                region_priority.get(cluster.region, 99),
                len(cluster.targets),
                cluster.anchor,
            ),
        )
    )
    field_order = {
        cluster.cluster_id: brute_force_route(cluster.predicted_dismount, cluster.targets)[0]
        for cluster in clusters
    }
    return PlannerDecision(
        cluster_order=ordered_clusters,
        field_order=field_order,
        reason="Fallback checks the northern cluster first, then the smaller southern cluster.",
        source="fallback",
    )


def build_planner_handlers(
    clusters: list[Cluster],
    map_state: MapState,
    fallback_plan: PlannerDecision,
) -> dict[str, Any]:
    cluster_context = [build_cluster_context(cluster, map_state) for cluster in clusters]
    expected_cluster_ids = {cluster.cluster_id for cluster in clusters}
    expected_fields = {cluster.cluster_id: set(cluster.targets) for cluster in clusters}

    def get_operation_context(_: dict[str, Any]) -> dict[str, Any]:
        return {
            "intercepted_signal": INTERCEPTED_SIGNAL,
            "constraints": {
                "max_transporters": 4,
                "max_scouts": 8,
                "action_points": 300,
                "goal": "Call the helicopter immediately after a scout confirms the human.",
            },
            "clusters": cluster_context,
        }

    def get_default_plan(_: dict[str, Any]) -> dict[str, Any]:
        return {
            "cluster_order": list(fallback_plan.cluster_order),
            "field_order": {cluster_id: list(order) for cluster_id, order in fallback_plan.field_order.items()},
            "reason": fallback_plan.reason,
        }

    def validate_plan(arguments: dict[str, Any]) -> dict[str, Any]:
        cluster_order_raw = arguments.get("cluster_order")
        field_order_raw = arguments.get("field_order")
        if not isinstance(cluster_order_raw, list) or not isinstance(field_order_raw, dict):
            return {"is_valid": False, "message": "cluster_order must be a list and field_order must be an object."}
        cluster_order = [item for item in cluster_order_raw if isinstance(item, str)]
        if set(cluster_order) != expected_cluster_ids or len(cluster_order) != len(expected_cluster_ids):
            return {"is_valid": False, "message": "cluster_order must contain every cluster exactly once."}
        for cluster_id, fields in expected_fields.items():
            raw_fields = field_order_raw.get(cluster_id)
            if not isinstance(raw_fields, list):
                return {"is_valid": False, "message": f"field_order missing list for {cluster_id}."}
            normalized = [item for item in raw_fields if isinstance(item, str)]
            if set(normalized) != fields or len(normalized) != len(fields):
                return {"is_valid": False, "message": f"field_order for {cluster_id} must cover each target once."}
        return {"is_valid": True}

    return {
        "get_operation_context": get_operation_context,
        "get_default_plan": get_default_plan,
        "validate_plan": validate_plan,
    }


def recommend_plan_with_openrouter(
    client: OpenRouterClient | None,
    clusters: list[Cluster],
    map_state: MapState,
) -> PlannerDecision:
    fallback_plan = deterministic_planner(clusters)
    if client is None:
        return fallback_plan
    handlers = build_planner_handlers(clusters, map_state, fallback_plan)
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are the Domatowo mission planner.\n"
                "Use tool calling before your final answer.\n"
                "Return JSON only with cluster_order, field_order and reason.\n"
                "You must keep every candidate field in the plan.\n"
                "Use the intercepted signal to prioritize likely B3 hiding spots."
            ),
        },
        {"role": "user", "content": "Build the best mission plan for the current Domatowo board."},
    ]
    try:
        completion = run_tool_conversation(
            client,
            messages=messages,
            tools=PLANNER_TOOLS,
            handlers=handlers,
            max_steps=PLANNER_MAX_STEPS,
        )
        payload = normalize_json_payload(completion.content or "")
        cluster_order = tuple(str(item) for item in payload.get("cluster_order", []))
        raw_field_order = payload.get("field_order")
        if not isinstance(raw_field_order, dict):
            raise OpenRouterError("Planner response is missing field_order.")
        field_order = {
            str(cluster_id): tuple(str(position) for position in positions)
            for cluster_id, positions in raw_field_order.items()
            if isinstance(positions, list)
        }
        validated = handlers["validate_plan"](
            {"cluster_order": list(cluster_order), "field_order": {key: list(value) for key, value in field_order.items()}}
        )
        if not validated.get("is_valid"):
            raise OpenRouterError(str(validated.get("message", "Planner validation failed.")))
        return PlannerDecision(
            cluster_order=cluster_order,
            field_order=field_order,
            reason=str(payload.get("reason", "")).strip() or "OpenRouter prioritized the cluster search order.",
            source="openrouter",
        )
    except (OpenRouterError, json.JSONDecodeError) as exc:
        logger.warning("Planner agent failed: {}. Using deterministic fallback.", exc)
        return fallback_plan


def classify_log_heuristically(message: str) -> LogAssessment:
    lowered = normalize_text(message)
    if any(marker in lowered for marker in EMPTY_MARKERS):
        return LogAssessment("empty", "Heuristics matched a negative search phrase.", "heuristic")
    if any(marker in lowered for marker in FOUND_MARKERS):
        return LogAssessment("found", "Heuristics matched a positive survivor phrase.", "heuristic")
    return LogAssessment("uncertain", "The log text is ambiguous for the local heuristic.", "heuristic")


def build_log_handlers(log_entry: InspectLog) -> dict[str, Any]:
    def get_log_context(_: dict[str, Any]) -> dict[str, Any]:
        return {"field": log_entry.field, "message": log_entry.message}

    def validate_result(arguments: dict[str, Any]) -> dict[str, Any]:
        return {"is_valid": arguments.get("result") in {"found", "empty", "uncertain"}}

    return {"get_log_context": get_log_context, "validate_result": validate_result}


def classify_log_with_openrouter(client: OpenRouterClient | None, log_entry: InspectLog) -> LogAssessment:
    heuristic = classify_log_heuristically(log_entry.message)
    if client is None:
        return heuristic
    handlers = build_log_handlers(log_entry)
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You classify Domatowo inspect logs.\n"
                "Use tool calling before the final answer.\n"
                "Return JSON only with keys result and reason.\n"
                "Allowed result values: found, empty, uncertain.\n"
                "Use found only when the message clearly confirms a live person on the inspected field."
            ),
        },
        {"role": "user", "content": "Classify the latest inspect log."},
    ]
    try:
        completion = run_tool_conversation(
            client,
            messages=messages,
            tools=LOG_ANALYST_TOOLS,
            handlers=handlers,
            max_steps=LOG_ANALYST_MAX_STEPS,
        )
        payload = normalize_json_payload(completion.content or "")
        result = str(payload.get("result", "")).strip()
        if result not in {"found", "empty", "uncertain"}:
            raise OpenRouterError("Log analyst returned an invalid result.")
        return LogAssessment(
            result,
            str(payload.get("reason", "")).strip() or "OpenRouter log analyst classified the message.",
            "openrouter",
        )
    except (OpenRouterError, json.JSONDecodeError) as exc:
        logger.warning("Log analyst failed for {}: {}. Using heuristics.", log_entry.field, exc)
        return heuristic


def extract_hex_pairs(text: str) -> str:
    return "".join(re.findall(r"\b[0-9a-fA-F]{2}\b", text))


def decode_hex_message(text: str) -> str:
    hex_payload = extract_hex_pairs(text)
    if not hex_payload:
        raise RuntimeError("The church clue did not contain any hexadecimal payload.")
    return bytes.fromhex(hex_payload).decode("utf-8")


def vigenere_decrypt_ascii(ciphertext: str, key: str) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    alphabet_index = {letter: index for index, letter in enumerate(alphabet)}
    normalized_key = normalize_text(key)
    if not normalized_key:
        raise RuntimeError("Vigenere key cannot be empty.")

    decoded: list[str] = []
    key_index = 0
    for character in normalize_text(ciphertext):
        if character not in alphabet_index:
            decoded.append(character)
            continue
        cipher_index = alphabet_index[character]
        shift = alphabet_index[normalized_key[key_index % len(normalized_key)]]
        decoded.append(alphabet[(cipher_index - shift) % len(alphabet)])
        key_index += 1
    return "".join(decoded)


def parse_church_clue(message: str) -> dict[str, str]:
    decoded_message = decode_hex_message(message)
    lines = [line.strip() for line in decoded_message.splitlines() if line.strip() and line.strip() != "---"]
    if len(lines) < 3:
        raise RuntimeError("The decoded church clue is incomplete.")
    encrypted_url = lines[2]
    return {
        "decoded_message": decoded_message,
        "question": lines[0],
        "encrypted_prompt": lines[1],
        "media_url": vigenere_decrypt_ascii(encrypted_url, CHURCH_URL_KEY),
    }


def resolve_side_secret_answer(hash_token: str) -> str:
    normalized = hash_token.strip().upper()
    if normalized != SIDE_SECRET_HASH:
        raise RuntimeError(f"Unexpected side-secret hash token: {hash_token}")
    return SIDE_SECRET_ANSWER


def build_plan_payload(
    planner: PlannerDecision,
    clusters: list[Cluster],
    map_state: MapState,
    *,
    attempt: int | None = None,
    max_attempts: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "source": planner.source,
        "reason": planner.reason,
        "cluster_order": list(planner.cluster_order),
        "field_order": {cluster_id: list(order) for cluster_id, order in planner.field_order.items()},
        "clusters": [build_cluster_context(cluster, map_state) for cluster in clusters],
        "side_quest": asdict(build_side_quest_plan(map_state)),
    }
    if attempt is not None:
        payload["attempt"] = attempt
    if max_attempts is not None:
        payload["max_attempts"] = max_attempts
    return payload


def prepare_attempt(
    client: DomatowoApiClient,
    model_client: OpenRouterClient | None,
    *,
    attempt: int | None = None,
    max_attempts: int | None = None,
) -> tuple[MapState, list[Cluster], PlannerDecision]:
    raw_map = client.get_map()
    write_json(MAP_PATH, raw_map)
    map_state = build_map_state(raw_map)
    clusters = build_clusters(map_state, position_list_from_search(client.search_symbol("B3")))
    planner = recommend_plan_with_openrouter(model_client, clusters, map_state)
    write_json(
        PLAN_PATH,
        build_plan_payload(
            planner,
            clusters,
            map_state,
            attempt=attempt,
            max_attempts=max_attempts,
        ),
    )
    return map_state, clusters, planner


def extract_unit_ids(create_response: dict[str, Any]) -> tuple[str, str]:
    transporter_id = str(create_response.get("object", ""))
    crew = create_response.get("crew")
    if not transporter_id or not isinstance(crew, list) or not crew:
        raise RuntimeError("Create response is missing transporter or crew metadata.")
    scout_id = str(crew[0].get("id", ""))
    if not scout_id:
        raise RuntimeError("Create response is missing the scout id.")
    return transporter_id, scout_id


def resolve_spawn_position(dismount_response: dict[str, Any], scout_id: str) -> str:
    spawned = dismount_response.get("spawned")
    if not isinstance(spawned, list) or not spawned:
        raise RuntimeError("Dismount response is missing spawned scout data.")
    for item in spawned:
        if isinstance(item, dict) and str(item.get("scout", "")) == scout_id:
            position = str(item.get("where", ""))
            if position:
                return position
    raise RuntimeError("Unable to resolve the scout spawn position.")


def execute_side_quest(client: DomatowoApiClient, map_state: MapState) -> SideQuestResult:
    plan = build_side_quest_plan(map_state)
    create_response = client.call({"action": "create", "type": "transporter", "passengers": 1})
    transporter_id, scout_id = extract_unit_ids(create_response)
    client.call({"action": "move", "object": transporter_id, "where": plan.anchor})
    dismount_response = client.call({"action": "dismount", "object": transporter_id, "passengers": 1})
    scout_position = resolve_spawn_position(dismount_response, scout_id)
    if scout_position != plan.secret_field:
        client.call({"action": "move", "object": scout_id, "where": plan.secret_field})
    client.call({"action": "inspect", "object": scout_id})
    clue_log = client.get_logs()[-1]
    if clue_log.field != plan.secret_field:
        raise RuntimeError("Church clue inspection ended on an unexpected field.")
    clue = parse_church_clue(clue_log.message)
    result = SideQuestResult(
        anchor=plan.anchor,
        scout_start=scout_position,
        field=clue_log.field,
        raw_message=clue_log.message,
        decoded_message=clue["decoded_message"],
        question=clue["question"],
        encrypted_prompt=clue["encrypted_prompt"],
        media_url=clue["media_url"],
        corrected_audio_hash=SIDE_SECRET_HASH,
        answer=resolve_side_secret_answer(SIDE_SECRET_HASH),
    )
    write_json(SIDE_QUEST_PATH, asdict(result))
    return result


def ensure_clean_board(client: DomatowoApiClient, *, reset: bool) -> None:
    objects = client.get_objects()
    logs = client.get_logs()
    expenses = client.call({"action": "expenses"})
    points_used = int(expenses.get("action_points_used", 0))
    if reset or objects or logs or points_used:
        logger.info("Resetting board to start from a clean mission state.")
        client.reset()


def run_with_retries(
    max_attempts: int,
    attempt_runner: Callable[[int], T],
    *,
    on_failure: Callable[[int, Exception], None] | None = None,
) -> T:
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1.")
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return attempt_runner(attempt)
        except Exception as exc:
            last_error = exc
            if on_failure is not None:
                on_failure(attempt, exc)
            if attempt == max_attempts:
                raise
    raise RuntimeError("Mission retries finished without running any attempt.") from last_error


def road_distance(map_state: MapState, start: str, end: str) -> int:
    start_coord = coord_from_label(start)
    end_coord = coord_from_label(end)
    queue: deque[tuple[Coordinate, int]] = deque([(start_coord, 0)])
    seen = {start_coord.label}
    while queue:
        current, distance = queue.popleft()
        if current.label == end_coord.label:
            return distance
        for dx, dy in ORTHOGONAL_STEPS:
            neighbour = Coordinate(current.x + dx, current.y + dy)
            if not map_state.in_bounds(neighbour):
                continue
            if neighbour.label in seen:
                continue
            if map_state.tile_at(neighbour) != "road":
                continue
            seen.add(neighbour.label)
            queue.append((neighbour, distance + 1))
    raise RuntimeError(f"No road path from {start} to {end}.")


def assign_transporters_to_clusters(
    clusters: list[Cluster],
    create_responses: list[dict[str, Any]],
    map_state: MapState,
) -> list[tuple[Cluster, dict[str, Any]]]:
    spawn_to_response = {str(response.get("spawn", "")): response for response in create_responses}
    spawns = [spawn for spawn in spawn_to_response if spawn]
    best_pairs: list[tuple[Cluster, dict[str, Any]]] | None = None
    best_score: tuple[int, int] | None = None
    for permutation in permutations(clusters):
        transport_cost = 0
        scout_cost = 0
        pairs: list[tuple[Cluster, dict[str, Any]]] = []
        for spawn, cluster in zip(spawns, permutation):
            transport_cost += road_distance(map_state, spawn, cluster.anchor)
            _, route_steps = brute_force_route(cluster.predicted_dismount, cluster.targets)
            scout_cost += route_steps
            pairs.append((cluster, spawn_to_response[spawn]))
        score = (transport_cost, scout_cost)
        if best_score is None or score < best_score:
            best_score = score
            best_pairs = pairs
    if best_pairs is None:
        raise RuntimeError("Unable to assign transporters to clusters.")
    return best_pairs


def build_search_order(assignment: TransportAssignment, planner: PlannerDecision) -> tuple[str, ...]:
    preferred = planner.field_order.get(assignment.cluster.cluster_id)
    if preferred:
        return preferred
    return brute_force_route(assignment.scout_position, assignment.cluster.targets)[0]


def execute_mission(
    client: DomatowoApiClient,
    map_state: MapState,
    planner: PlannerDecision,
    model_client: OpenRouterClient | None,
) -> dict[str, Any]:
    clusters = build_clusters(map_state, position_list_from_search(client.search_symbol("B3")))
    side_quest: SideQuestResult | None = None
    try:
        side_quest = execute_side_quest(client, map_state)
        logger.info("Side quest clue recovered from {}.", side_quest.field)
    except Exception as exc:
        logger.warning("Side quest failed: {}. Continuing with the main mission.", exc)

    create_responses = [
        client.call({"action": "create", "type": "transporter", "passengers": SCOUTS_PER_TRANSPORTER})
        for _ in range(len(clusters))
    ]
    live_assignments: list[TransportAssignment] = []
    for cluster, response in assign_transporters_to_clusters(clusters, create_responses, map_state):
        transporter_id, scout_id = extract_unit_ids(response)
        client.call({"action": "move", "object": transporter_id, "where": cluster.anchor})
        dismount_response = client.call({"action": "dismount", "object": transporter_id, "passengers": 1})
        scout_position = resolve_spawn_position(dismount_response, scout_id)
        live_assignments.append(
            TransportAssignment(
                cluster=cluster,
                transporter_id=transporter_id,
                scout_id=scout_id,
                scout_position=scout_position,
            )
        )

    ordered_assignments = [
        next(item for item in live_assignments if item.cluster.cluster_id == cluster_id)
        for cluster_id in planner.cluster_order
    ]
    found_entry: InspectLog | None = None
    for assignment in ordered_assignments:
        search_order = build_search_order(assignment, planner)
        logger.info("Searching {} from {} via {}.", assignment.cluster.region, assignment.scout_position, list(search_order))
        current_position = assignment.scout_position
        for target in search_order:
            if current_position != target:
                client.call({"action": "move", "object": assignment.scout_id, "where": target})
                current_position = target
            client.call({"action": "inspect", "object": assignment.scout_id})
            latest_entry = client.get_logs()[-1]
            assessment = classify_log_with_openrouter(model_client, latest_entry)
            logger.info("Inspect {} -> {} ({})", latest_entry.field, assessment.result, assessment.source)
            if assessment.result == "found":
                found_entry = latest_entry
                break
        if found_entry is not None:
            break

    if found_entry is None:
        raise RuntimeError("Mission exhausted every B3 candidate without a confirmed survivor.")
    verify_response = client.call({"action": "callHelicopter", "destination": found_entry.field})
    write_json(VERIFY_RESPONSE_PATH, verify_response)
    result = {
        "verify_response": verify_response,
        "side_quest": asdict(side_quest) if side_quest is not None else None,
    }
    write_json(RESULT_PATH, result)
    return result


def main() -> int:
    configure_logging(name="domatowo")
    args = parse_args()
    api_key = get_env("AG3NTS_API_KEY")
    if not api_key:
        logger.error("Missing AG3NTS_API_KEY in .env.")
        return 1

    model_client = None if args.skip_model else build_openrouter_client(args.model)
    client = DomatowoApiClient(api_key)
    try:
        ensure_clean_board(client, reset=args.reset)
        if not args.verify:
            _, _, _ = prepare_attempt(client, model_client)
            logger.info("Plan prepared at {}. Run with --verify to execute the mission.", PLAN_PATH)
            write_json(SEARCH_TRACE_PATH, client.history)
            return 0

        def attempt_runner(attempt: int) -> dict[str, Any]:
            attempt_requires_reset = attempt > 1
            if attempt_requires_reset:
                ensure_clean_board(client, reset=True)
            attempt_map_state, _, attempt_planner = prepare_attempt(
                client,
                model_client,
                attempt=attempt,
                max_attempts=args.max_attempts,
            )
            result = execute_mission(client, attempt_map_state, attempt_planner, model_client)
            result["attempt"] = attempt
            result["max_attempts"] = args.max_attempts
            write_json(RESULT_PATH, result)
            return result

        def on_failure(attempt: int, exc: Exception) -> None:
            if attempt < args.max_attempts:
                logger.warning(
                    "Mission attempt {}/{} failed: {}. Retrying.",
                    attempt,
                    args.max_attempts,
                    exc,
                )
            else:
                logger.error(
                    "Mission attempt {}/{} failed: {}.",
                    attempt,
                    args.max_attempts,
                    exc,
                )

        response = run_with_retries(
            args.max_attempts,
            attempt_runner,
            on_failure=on_failure,
        )
        write_json(SEARCH_TRACE_PATH, client.history)
        logger.success("Mission response: {}", response)
        return 0
    except HttpRequestError as exc:
        payload = exc.to_response_dict()
        write_json(VERIFY_RESPONSE_PATH, payload)
        write_json(SEARCH_TRACE_PATH, client.history)
        logger.error("Network failure: {}", exc)
        return 1
    except Exception as exc:  # pragma: no cover
        write_json(SEARCH_TRACE_PATH, client.history)
        logger.error("Mission failed: {}", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
