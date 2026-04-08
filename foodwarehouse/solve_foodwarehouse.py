"""Solve the AG3NTS `foodwarehouse` task by creating the required city orders."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO_ROOT_HINT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_HINT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_HINT))

from devs_utilities.ag3nts import (
    AG3NTS_VERIFY_URL,
    build_ag3nts_public_data_url,
    submit_task_answer,
)
from devs_utilities.bootstrap import bootstrap_repo
from devs_utilities.files import write_json
from devs_utilities.http import HttpRequestError, get_json
from devs_utilities.logging import configure_logging, logger as shared_logger
from devs_utilities.repo_env import get_env, get_int_env


REPO_ROOT = bootstrap_repo(__file__)
logger = shared_logger.bind(component="foodwarehouse")


TASK_NAME = "foodwarehouse"
PUBLIC_DATA_URL = build_ag3nts_public_data_url("food4cities.json")
VERIFY_TIMEOUT_SECONDS = get_int_env("AG3NTS_TIMEOUT_SECONDS", 30) or 30

OUTPUT_DIR = Path(__file__).resolve().parent
PLAN_PATH = OUTPUT_DIR / "last_plan.json"
VERIFY_RESPONSE_PATH = OUTPUT_DIR / "last_verify_response.json"


@dataclass(frozen=True, slots=True)
class Creator:
    user_id: int
    login: str
    birthday: str


@dataclass(frozen=True, slots=True)
class CityDemand:
    city: str
    items: dict[str, int]


@dataclass(frozen=True, slots=True)
class DeliveryRequirement:
    city: str
    destination: int
    items: dict[str, int]


@dataclass(frozen=True, slots=True)
class PlannedOrder:
    city: str
    destination: int
    title: str
    creator: Creator
    items: dict[str, int]


def get_api_key() -> str:
    """Read the AG3NTS key from the current repository environment."""

    api_key = get_env("AG3NTS_API_KEY") or get_env("COURSE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing AG3NTS_API_KEY (or legacy COURSE_API_KEY) in .env.")
    return api_key


def normalize_city_name(raw_city: str) -> str:
    """Normalize a city label coming from the public JSON payload."""

    city = raw_city.strip()
    if not city:
        raise ValueError("City name cannot be empty.")
    return city[:1].upper() + city[1:].lower()


def coerce_positive_int(value: Any, *, context: str) -> int:
    """Validate quantities loaded from JSON or API responses."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} must be an integer, got: {value!r}")
    if value <= 0:
        raise ValueError(f"{context} must be positive, got: {value!r}")
    return value


def parse_city_demands(payload: Any) -> list[CityDemand]:
    """Parse the public `food4cities.json` file into a normalized structure."""

    if not isinstance(payload, dict):
        raise ValueError("City demand payload must be a JSON object.")

    demands: list[CityDemand] = []
    for raw_city, raw_items in payload.items():
        if not isinstance(raw_city, str):
            raise ValueError(f"Unexpected city key type: {type(raw_city)!r}")
        if not isinstance(raw_items, dict) or not raw_items:
            raise ValueError(f"City {raw_city!r} must contain a non-empty object of items.")

        items: dict[str, int] = {}
        for raw_name, raw_count in raw_items.items():
            if not isinstance(raw_name, str) or not raw_name.strip():
                raise ValueError(f"City {raw_city!r} has an invalid item name: {raw_name!r}")
            items[raw_name.strip()] = coerce_positive_int(
                raw_count,
                context=f"Quantity for {raw_city}/{raw_name}",
            )

        demands.append(CityDemand(city=normalize_city_name(raw_city), items=items))

    if not demands:
        raise ValueError("No city demands found in the public JSON payload.")
    return demands


def parse_missing_requirements(rows: Any) -> list[DeliveryRequirement]:
    """Normalize the `missing` block returned by the `done` validator."""

    if not isinstance(rows, list):
        raise ValueError("Missing-requirements payload must be a list.")

    requirements: list[DeliveryRequirement] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"Unexpected requirement row: {row!r}")

        city = normalize_city_name(str(row.get("city", "")))
        destination = coerce_positive_int(
            row.get("destination"),
            context=f"Destination for {city}",
        )
        raw_items = row.get("items")
        if not isinstance(raw_items, dict) or not raw_items:
            raise ValueError(f"Requirement for {city} is missing its items map.")
        items = {
            str(name).strip(): coerce_positive_int(
                quantity,
                context=f"Quantity for {city}/{name}",
            )
            for name, quantity in raw_items.items()
        }
        requirements.append(
            DeliveryRequirement(
                city=city,
                destination=destination,
                items=items,
            )
        )

    return requirements


def requirement_index(requirements: Iterable[DeliveryRequirement]) -> dict[str, dict[str, Any]]:
    """Build a stable city-indexed view for equality checks and JSON dumps."""

    index: dict[str, dict[str, Any]] = {}
    for requirement in requirements:
        if requirement.city in index:
            raise ValueError(f"Duplicate requirement for city {requirement.city}.")
        index[requirement.city] = {
            "destination": requirement.destination,
            "items": dict(sorted(requirement.items.items())),
        }
    return index


def build_planned_orders(
    demands: Iterable[CityDemand],
    destination_by_city: Mapping[str, int],
    creator: Creator,
) -> list[PlannedOrder]:
    """Attach destination codes and a creator to every city demand."""

    planned_orders: list[PlannedOrder] = []
    for demand in demands:
        try:
            destination = destination_by_city[demand.city]
        except KeyError as exc:
            raise ValueError(f"Missing destination code for city {demand.city}.") from exc
        planned_orders.append(
            PlannedOrder(
                city=demand.city,
                destination=destination,
                title=f"Dostawa dla {demand.city}",
                creator=creator,
                items=dict(demand.items),
            )
        )
    return planned_orders


def sql_string_literal(value: str) -> str:
    """Escape a Python string for a simple SQL string literal."""

    return "'" + value.replace("'", "''") + "'"


class FoodwarehouseClient:
    """Small typed wrapper around the task-specific `/verify` tool API."""

    def __init__(self, api_key: str, *, timeout_seconds: float) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def call_tool(
        self,
        answer: dict[str, Any],
        *,
        allow_http_error: bool = False,
    ) -> dict[str, Any]:
        try:
            response = submit_task_answer(
                AG3NTS_VERIFY_URL,
                api_key=self.api_key,
                task=TASK_NAME,
                answer=answer,
                timeout_seconds=self.timeout_seconds,
            )
        except HttpRequestError as exc:
            if allow_http_error:
                payload = exc.body_as_json()
                if isinstance(payload, dict):
                    if exc.status_code is not None:
                        payload.setdefault("http_status", exc.status_code)
                    return payload
            raise RuntimeError(str(exc)) from exc

        if not isinstance(response, dict):
            raise RuntimeError(f"Unexpected response shape for tool call: {response!r}")
        return response

    def reset(self) -> dict[str, Any]:
        return self.call_tool({"tool": "reset"})

    def get_orders(self) -> dict[str, Any]:
        return self.call_tool({"tool": "orders", "action": "get"})

    def database_query(self, query: str) -> dict[str, Any]:
        return self.call_tool({"tool": "database", "query": query})

    def generate_signature(self, *, login: str, birthday: str, destination: int) -> str:
        response = self.call_tool(
            {
                "tool": "signatureGenerator",
                "action": "generate",
                "login": login,
                "birthday": birthday,
                "destination": destination,
            }
        )
        signature = response.get("hash")
        if not isinstance(signature, str) or not signature:
            raise RuntimeError(f"Unexpected signature response: {response!r}")
        return signature

    def create_order(self, planned_order: PlannedOrder, signature: str) -> dict[str, Any]:
        response = self.call_tool(
            {
                "tool": "orders",
                "action": "create",
                "title": planned_order.title,
                "creatorID": planned_order.creator.user_id,
                "destination": planned_order.destination,
                "signature": signature,
            }
        )
        order = response.get("order")
        if not isinstance(order, dict):
            raise RuntimeError(f"Unexpected create-order response: {response!r}")
        return order

    def append_items(self, order_id: str, items: Mapping[str, int]) -> dict[str, Any]:
        return self.call_tool(
            {
                "tool": "orders",
                "action": "append",
                "id": order_id,
                "items": dict(items),
            }
        )

    def done(self, *, allow_http_error: bool = False) -> dict[str, Any]:
        return self.call_tool({"tool": "done"}, allow_http_error=allow_http_error)


def fetch_default_creator(client: FoodwarehouseClient) -> Creator:
    """Pick the first active transport operator from the database."""

    role_response = client.database_query(
        "select role_id from roles where name = 'Obsługa transportów'"
    )
    role_rows = role_response.get("rows")
    if not isinstance(role_rows, list) or len(role_rows) != 1:
        raise RuntimeError(f"Could not resolve transport role id: {role_response!r}")

    role_id = role_rows[0].get("role_id")
    role_id = coerce_positive_int(role_id, context="Transport role id")

    user_response = client.database_query(
        "select user_id, login, birthday from users "
        f"where role = {role_id} and is_active = 1 "
        "order by user_id limit 1"
    )
    user_rows = user_response.get("rows")
    if not isinstance(user_rows, list) or len(user_rows) != 1:
        raise RuntimeError(f"Could not resolve a default creator: {user_response!r}")

    row = user_rows[0]
    return Creator(
        user_id=coerce_positive_int(row.get("user_id"), context="Creator user_id"),
        login=str(row.get("login", "")).strip(),
        birthday=str(row.get("birthday", "")).strip(),
    )


def fetch_destination_map(
    client: FoodwarehouseClient,
    cities: Iterable[str],
) -> dict[str, int]:
    """Load destination ids for the requested city set."""

    unique_cities = list(dict.fromkeys(cities))
    in_clause = ", ".join(sql_string_literal(city) for city in unique_cities)
    response = client.database_query(
        "select destination_id, name from destinations "
        f"where name in ({in_clause})"
    )
    rows = response.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError(f"Unexpected destination query response: {response!r}")

    destination_by_city = {
        normalize_city_name(str(row.get("name", ""))): coerce_positive_int(
            row.get("destination_id"),
            context=f"Destination for {row.get('name')}",
        )
        for row in rows
    }
    missing_cities = [city for city in unique_cities if city not in destination_by_city]
    if missing_cities:
        missing_text = ", ".join(missing_cities)
        raise RuntimeError(f"Missing destination codes for: {missing_text}")
    return destination_by_city


def fetch_city_demands() -> list[CityDemand]:
    """Download and parse the public city-needs file."""

    payload = get_json(PUBLIC_DATA_URL, timeout_seconds=VERIFY_TIMEOUT_SECONDS)
    return parse_city_demands(payload)


def fetch_backend_requirements(client: FoodwarehouseClient) -> list[DeliveryRequirement]:
    """Ask the validator what is still missing after a clean reset."""

    response = client.done(allow_http_error=True)
    missing = response.get("missing")
    if response.get("code") == -655 and isinstance(missing, list):
        return parse_missing_requirements(missing)
    raise RuntimeError(f"Unexpected validator baseline response: {response!r}")


def build_plan(client: FoodwarehouseClient) -> tuple[Creator, list[PlannedOrder]]:
    """Assemble the full order plan from public data and the read-only database."""

    creator = fetch_default_creator(client)
    city_demands = fetch_city_demands()
    destination_by_city = fetch_destination_map(
        client,
        (demand.city for demand in city_demands),
    )
    return creator, build_planned_orders(city_demands, destination_by_city, creator)


def execute_plan(
    client: FoodwarehouseClient,
    planned_orders: Iterable[PlannedOrder],
) -> list[dict[str, Any]]:
    """Create and fill all required city orders."""

    created_orders: list[dict[str, Any]] = []
    for planned_order in planned_orders:
        signature = client.generate_signature(
            login=planned_order.creator.login,
            birthday=planned_order.creator.birthday,
            destination=planned_order.destination,
        )
        created_order = client.create_order(planned_order, signature)
        order_id = str(created_order.get("id", "")).strip()
        if not order_id:
            raise RuntimeError(f"Created order is missing its id: {created_order!r}")
        client.append_items(order_id, planned_order.items)
        created_orders.append(
            {
                "city": planned_order.city,
                "id": order_id,
                "destination": planned_order.destination,
                "title": planned_order.title,
                "items": dict(planned_order.items),
            }
        )
    return created_orders


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Reset the task state, create all missing orders, and run the final `done` check.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(verbose=args.verbose, name="foodwarehouse")

    client = FoodwarehouseClient(get_api_key(), timeout_seconds=VERIFY_TIMEOUT_SECONDS)
    creator, planned_orders = build_plan(client)

    expected_requirements = [
        DeliveryRequirement(
            city=planned_order.city,
            destination=planned_order.destination,
            items=dict(planned_order.items),
        )
        for planned_order in planned_orders
    ]
    plan_payload = {
        "creator": asdict(creator),
        "requirements": requirement_index(expected_requirements),
        "orders": [asdict(planned_order) for planned_order in planned_orders],
    }
    write_json(PLAN_PATH, plan_payload)
    logger.info("Prepared plan for {} cities. Saved to {}", len(planned_orders), PLAN_PATH)

    if not args.verify:
        print(json.dumps(plan_payload, ensure_ascii=False, indent=2))
        return 0

    client.reset()
    backend_requirements = fetch_backend_requirements(client)
    if requirement_index(expected_requirements) != requirement_index(backend_requirements):
        raise RuntimeError(
            "Public JSON + database plan does not match the validator baseline.\n"
            f"Expected: {json.dumps(requirement_index(expected_requirements), ensure_ascii=False, indent=2)}\n"
            f"Actual: {json.dumps(requirement_index(backend_requirements), ensure_ascii=False, indent=2)}"
        )

    orders_before = client.get_orders()
    logger.info("Task reset complete. Current seeded order count: {}", orders_before.get("count"))
    created_orders = execute_plan(client, planned_orders)
    final_response = client.done()

    verify_payload = {
        "creator": asdict(creator),
        "created_orders": created_orders,
        "final_response": final_response,
    }
    write_json(VERIFY_RESPONSE_PATH, verify_payload)
    print(json.dumps(final_response, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
