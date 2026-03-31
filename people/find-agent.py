"""Find the best suspect match and prepare the access-level verification payload."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT_HINT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_HINT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_HINT))

from devs_utilities.ag3nts import (
    AG3NTS_ACCESS_LEVEL_URL,
    AG3NTS_LOCATION_URL,
    AG3NTS_VERIFY_URL,
    build_ag3nts_task_data_url,
    submit_task_answer,
)
from devs_utilities.bootstrap import bootstrap_repo
from devs_utilities.files import resolve_path, write_json
from devs_utilities.http import get_json as http_get_json
from devs_utilities.http import post_json as http_post_json
from devs_utilities.logging import configure_logging, logger as shared_logger
from devs_utilities.openrouter import (
    build_task_openrouter_client,
    build_task_site_name,
    OpenRouterClient,
    OpenRouterError,
    ToolCall,
)
from repo_env import (
    get_course_api_key,
    get_env,
    get_int_env,
    get_llm_api_key,
    get_llm_base_url,
    get_optional_env,
)


REPO_ROOT = bootstrap_repo(__file__)
logger = shared_logger.bind(component="people.find")


OPENROUTER_API_URL = get_llm_base_url()
TASK_NAME = "findhim"
DEFAULT_OPENROUTER_MODEL = get_env("OPENROUTER_MODEL", "openai/gpt-4.1-mini") or "openai/gpt-4.1-mini"
API_TIMEOUT_SECONDS = get_int_env("PEOPLE_TIMEOUT_SECONDS", 120) or 120
OPENROUTER_TIMEOUT_SECONDS = get_int_env("OPENROUTER_TIMEOUT_SECONDS", 120) or 120
DEFAULT_PLANTS_PATH = "findhim_locations.json"
DEFAULT_SUSPECTS_PATH = "people_result.json"
DEFAULT_OUTPUT_PATH = "findhim_result.json"
MODEL_MAX_STEPS = 4


@dataclass(slots=True)
class AppConfig:
    course_api_key: str
    llm_api_key: str
    openrouter_model: str
    site_url: str | None
    site_name: str | None


@dataclass(slots=True)
class Suspect:
    name: str
    surname: str
    birth_year: int


@dataclass(slots=True)
class Coordinate:
    latitude: float
    longitude: float


@dataclass(slots=True)
class PowerPlant:
    city: str
    code: str
    coordinate: Coordinate


@dataclass(slots=True)
class CandidateMatch:
    suspect: Suspect
    power_plant: PowerPlant
    distance_km: float
    access_level: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rozwiązanie zadania findhim: znajduje podejrzanego najbliżej elektrowni, "
            "pobiera access level i przygotowuje payload do /verify."
        )
    )
    parser.add_argument(
        "--config",
        default="people_config.json",
        help="Ścieżka do pliku JSON z kluczami i ustawieniami OpenRouter/HUB.",
    )
    parser.add_argument(
        "--csv",
        default="people.csv",
        help="Ścieżka do people.csv z pełnymi danymi źródłowymi.",
    )
    parser.add_argument(
        "--suspects",
        default=DEFAULT_SUSPECTS_PATH,
        help="Ścieżka do wyniku poprzedniego zadania z listą podejrzanych.",
    )
    parser.add_argument(
        "--plants",
        default=DEFAULT_PLANTS_PATH,
        help="Lokalny cache dla listy elektrowni z hubu.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="Plik wynikowy z payloadem do wysłania.",
    )
    parser.add_argument(
        "--model",
        help="Nadpisuje openrouter_model z configu.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Wyślij wynik na wbudowany endpoint /verify.",
    )
    parser.add_argument(
        "--refresh-plants",
        action="store_true",
        help="Pobierz świeżą listę elektrowni z hubu nawet jeśli lokalny plik już istnieje.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Nie wysyłaj wyniku do /verify. Pozostałe requesty są wykonywane normalnie.",
    )
    return parser.parse_args()


def load_config(path: Path, args: argparse.Namespace) -> AppConfig:
    payload: dict[str, Any] = {}
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))

    hub_api_key = str(
        payload.get("course_api_key") or get_course_api_key() or ""
    ).strip()
    openrouter_api_key = str(
        payload.get("llm_api_key")
        or get_llm_api_key()
        or ""
    ).strip()
    openrouter_model = str(payload.get("openrouter_model") or DEFAULT_OPENROUTER_MODEL).strip()
    site_url = (
        str(payload.get("site_url") or get_optional_env("OPENROUTER_SITE_URL") or "").strip()
        or None
    )
    site_name = build_task_site_name(__file__)

    if args.model:
        openrouter_model = args.model

    if not hub_api_key:
        raise ValueError("Brakuje COURSE_API_KEY lub course_api_key w configu.")
    if not openrouter_api_key:
        raise ValueError("Brakuje LLM_API_KEY lub llm_api_key w configu.")

    return AppConfig(
        course_api_key=hub_api_key,
        llm_api_key=openrouter_api_key,
        openrouter_model=openrouter_model,
        site_url=site_url,
        site_name=site_name,
    )


def normalize_name(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    ascii_only = "".join(char for char in decomposed if not unicodedata.combining(char))
    return ascii_only.strip().casefold()


def load_birth_years(csv_path: Path) -> dict[tuple[str, str], int]:
    people: dict[tuple[str, str], int] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (row["name"].strip(), row["surname"].strip())
            people[key] = int(row["birthDate"][:4])
    return people


def load_suspects(path: Path, birth_years: dict[tuple[str, str], int]) -> list[Suspect]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    suspects: list[Suspect] = []
    for item in payload["answer"]:
        name = str(item["name"]).strip()
        surname = str(item["surname"]).strip()
        birth_year = int(item.get("born") or birth_years[(name, surname)])
        suspects.append(Suspect(name=name, surname=surname, birth_year=birth_year))
    return suspects


def fetch_power_plants(config: AppConfig, path: Path, refresh: bool) -> dict[str, Any]:
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))
    url = build_ag3nts_task_data_url(config.course_api_key, "findhim_locations.json")
    payload = http_get_json(url, timeout_seconds=API_TIMEOUT_SECONDS)
    write_json(path, payload)
    return payload


def build_geocode_schema() -> dict[str, Any]:
    return {
        "name": "power_plant_coordinates",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "plants": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "city": {"type": "string"},
                            "latitude": {"type": "number"},
                            "longitude": {"type": "number"},
                        },
                        "required": ["city", "latitude", "longitude"],
                    },
                }
            },
            "required": ["plants"],
        },
    }


FIND_AGENT_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_power_plant_city_context",
            "description": "Return the list of Polish cities that need approximate coordinates.",
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
            "name": "validate_geocode_results",
            "description": "Validate that the proposed coordinates cover all requested cities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "plants": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"},
                                "latitude": {"type": "number"},
                                "longitude": {"type": "number"},
                            },
                            "required": ["city", "latitude", "longitude"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["plants"],
                "additionalProperties": False,
            },
        },
    },
]


def build_find_agent_handlers(city_names: list[str]) -> dict[str, Any]:
    expected = {normalize_name(city) for city in city_names}

    def get_power_plant_city_context(_: dict[str, Any]) -> dict[str, Any]:
        prompt = (
            "Dla podanych polskich miast zwrĂłÄ‡ przybliĹĽone wspĂłĹ‚rzÄ™dne geograficzne "
            "centrĂłw tych miejscowoĹ›ci. Nie podawaj objaĹ›nieĹ„. Miasta:\n"
            + "\n".join(f"- {city}" for city in city_names)
        )
        return {"prompt": prompt, "cities": city_names}

    def validate_geocode_results(arguments: dict[str, Any]) -> dict[str, Any]:
        raw_plants = arguments.get("plants")
        if not isinstance(raw_plants, list):
            return {"is_valid": False, "message": "plants must be a list"}
        received = {
            normalize_name(str(item.get("city")))
            for item in raw_plants
            if isinstance(item, dict) and item.get("city") is not None
        }
        return {
            "is_valid": received == expected,
            "expected_cities": sorted(expected),
            "received_cities": sorted(received),
        }

    return {
        "get_power_plant_city_context": get_power_plant_city_context,
        "validate_geocode_results": validate_geocode_results,
    }


def execute_find_agent_tool_call(
    tool_call: ToolCall,
    handlers: dict[str, Any],
) -> dict[str, Any]:
    if tool_call.name not in handlers:
        raise OpenRouterError(f"Unknown people.find tool call: {tool_call.name!r}")
    result = handlers[tool_call.name](tool_call.arguments)
    return {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "name": tool_call.name,
        "content": json.dumps(result, ensure_ascii=False),
    }


def geocode_power_plants_legacy(
    raw_plants: dict[str, Any],
    config: AppConfig,
    openrouter_client: OpenRouterClient,
) -> list[PowerPlant]:
    plants_section = raw_plants["power_plants"]
    city_names = list(plants_section.keys())
    prompt = (
        "Dla podanych polskich miast zwróć przybliżone współrzędne geograficzne "
        "centrów tych miejscowości. Nie podawaj objaśnień. Miasta:\n"
        + "\n".join(f"- {city}" for city in city_names)
    )
    response = openrouter_client.create_raw_completion_legacy(
        [
            {
                "role": "system",
                "content": (
                    "Zwracasz tylko poprawny JSON zgodny ze schematem. "
                    "Używaj współrzędnych miast w Polsce."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        extra_payload={
            "schema_legacy": {
                "type": "json_schema",
                "json_schema": build_geocode_schema(),
            }
        },
    )
    completion = extract_completion_result(response)
    content = str(completion.content or "").strip()
    if not content:
        raise RuntimeError(f"Brak treści w odpowiedzi geokodowania: {response}")

    parsed = json.loads(content)
    coordinates_by_city = {
        normalize_name(str(item["city"])): Coordinate(
            latitude=float(item["latitude"]),
            longitude=float(item["longitude"]),
        )
        for item in parsed["plants"]
    }

    plants: list[PowerPlant] = []
    for city, details in plants_section.items():
        coordinate = coordinates_by_city.get(normalize_name(city))
        if coordinate is None:
            raise RuntimeError(f"Brak współrzędnych dla elektrowni w mieście {city}")
        plants.append(
            PowerPlant(
                city=city,
                code=str(details["code"]),
                coordinate=coordinate,
            )
        )
    return plants


def geocode_power_plants(
    raw_plants: dict[str, Any],
    config: AppConfig,
    openrouter_client: OpenRouterClient,
) -> list[PowerPlant]:
    """Tool-calling wrapper for approximate power-plant geocoding."""

    plants_section = raw_plants["power_plants"]
    city_names = list(plants_section.keys())
    handlers = build_find_agent_handlers(city_names)
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "Zwracasz tylko poprawny JSON zgodny ze schematem. "
                "Użyj tool callingu przed finalną odpowiedzią. "
                "Używaj współrzędnych miast w Polsce."
            ),
        },
        {
            "role": "user",
            "content": "Przygotuj współrzędne dla miast z elektrowniami.",
        },
    ]
    content = ""
    for _ in range(MODEL_MAX_STEPS):
        completion = openrouter_client.create_completion(
            messages,
            tools=FIND_AGENT_TOOLS,
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
            for tool_call in completion.tool_calls:
                messages.append(execute_find_agent_tool_call(tool_call, handlers))
            continue
        content = str(completion.content or "").strip()
        if content:
            break
    if not content:
        raise RuntimeError("Brak treści w odpowiedzi geokodowania.")

    parsed = json.loads(content)
    coordinates_by_city = {
        normalize_name(str(item["city"])): Coordinate(
            latitude=float(item["latitude"]),
            longitude=float(item["longitude"]),
        )
        for item in parsed["plants"]
    }

    plants: list[PowerPlant] = []
    for city, details in plants_section.items():
        coordinate = coordinates_by_city.get(normalize_name(city))
        if coordinate is None:
            raise RuntimeError(f"Brak współrzędnych dla elektrowni w mieście {city}")
        plants.append(
            PowerPlant(
                city=city,
                code=str(details["code"]),
                coordinate=coordinate,
            )
        )
    return plants


def normalize_location_entry(entry: Any) -> Coordinate | None:
    if isinstance(entry, dict):
        lat = entry.get("latitude", entry.get("lat"))
        lon = entry.get("longitude", entry.get("lon", entry.get("lng")))
        if lat is None or lon is None:
            return None
        return Coordinate(latitude=float(lat), longitude=float(lon))
    if isinstance(entry, list) and len(entry) >= 2:
        return Coordinate(latitude=float(entry[0]), longitude=float(entry[1]))
    if isinstance(entry, str):
        parts = [part.strip() for part in entry.split(",")]
        if len(parts) >= 2:
            return Coordinate(latitude=float(parts[0]), longitude=float(parts[1]))
    return None


def fetch_person_locations(config: AppConfig, suspect: Suspect) -> list[Coordinate]:
    payload = {
        "apikey": config.course_api_key,
        "name": suspect.name,
        "surname": suspect.surname,
    }
    response = http_post_json(
        AG3NTS_LOCATION_URL,
        payload,
        timeout_seconds=API_TIMEOUT_SECONDS,
    )
    if isinstance(response, dict):
        raw_locations = (
            response.get("locations")
            or response.get("answer")
            or response.get("coordinates")
            or []
        )
    elif isinstance(response, list):
        raw_locations = response
    else:
        raise RuntimeError(f"Nieobsługiwany format odpowiedzi location: {response}")

    locations: list[Coordinate] = []
    for entry in raw_locations:
        normalized = normalize_location_entry(entry)
        if normalized is not None:
            locations.append(normalized)
    if not locations:
        raise RuntimeError(
            f"Brak poprawnych współrzędnych dla osoby {suspect.name} {suspect.surname}: {response}"
        )
    return locations


def fetch_access_level(config: AppConfig, suspect: Suspect) -> int:
    payload = {
        "apikey": config.course_api_key,
        "name": suspect.name,
        "surname": suspect.surname,
        "birthYear": suspect.birth_year,
    }
    response = http_post_json(
        AG3NTS_ACCESS_LEVEL_URL,
        payload,
        timeout_seconds=API_TIMEOUT_SECONDS,
    )
    access_level = response.get("accessLevel") if isinstance(response, dict) else None
    if access_level is None:
        raise RuntimeError(
            f"Brak accessLevel dla osoby {suspect.name} {suspect.surname}: {response}"
        )
    return int(access_level)


def haversine_km(point_a: Coordinate, point_b: Coordinate) -> float:
    radius_km = 6371.0
    lat_1 = math.radians(point_a.latitude)
    lon_1 = math.radians(point_a.longitude)
    lat_2 = math.radians(point_b.latitude)
    lon_2 = math.radians(point_b.longitude)
    delta_lat = lat_2 - lat_1
    delta_lon = lon_2 - lon_1
    a_value = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat_1) * math.cos(lat_2) * math.sin(delta_lon / 2) ** 2
    )
    c_value = 2 * math.atan2(math.sqrt(a_value), math.sqrt(1 - a_value))
    return radius_km * c_value


def find_best_match(suspects: list[Suspect], plants: list[PowerPlant], config: AppConfig) -> CandidateMatch:
    best_match: CandidateMatch | None = None
    for suspect in suspects:
        locations = fetch_person_locations(config, suspect)
        for location in locations:
            for plant in plants:
                distance_km = haversine_km(location, plant.coordinate)
                if best_match is None or distance_km < best_match.distance_km:
                    best_match = CandidateMatch(
                        suspect=suspect,
                        power_plant=plant,
                        distance_km=distance_km,
                    )
    if best_match is None:
        raise RuntimeError("Nie udało się wyznaczyć żadnego kandydata.")
    best_match.access_level = fetch_access_level(config, best_match.suspect)
    return best_match


def build_verify_payload(match: CandidateMatch, config: AppConfig) -> dict[str, Any]:
    if match.access_level is None:
        raise ValueError("Brak access_level w najlepszym dopasowaniu.")
    return {
        "apikey": config.course_api_key,
        "task": TASK_NAME,
        "answer": {
            "name": match.suspect.name,
            "surname": match.suspect.surname,
            "accessLevel": match.access_level,
            "powerPlant": match.power_plant.code,
        },
    }


def main() -> None:
    configure_logging(name="people.find")
    if not OPENROUTER_API_URL:
        raise ValueError("Missing LLM_BASE_URL in the local repository config.")
    args = parse_args()
    people_dir = REPO_ROOT / "people"
    config = load_config(resolve_path(args.config, people_dir), args)
    birth_years = load_birth_years(resolve_path(args.csv, people_dir))
    suspects = load_suspects(resolve_path(args.suspects, people_dir), birth_years)
    raw_plants = fetch_power_plants(
        config,
        resolve_path(args.plants, people_dir),
        args.refresh_plants,
    )
    openrouter_client = build_task_openrouter_client(
        __file__,
        api_key=config.llm_api_key,
        base_url=OPENROUTER_API_URL,
        model=config.openrouter_model,
        timeout_seconds=OPENROUTER_TIMEOUT_SECONDS,
        site_url=config.site_url,
        site_name=config.site_name,
    )
    plants = geocode_power_plants(raw_plants, config, openrouter_client)
    best_match = find_best_match(suspects, plants, config)
    payload = build_verify_payload(best_match, config)

    write_json(resolve_path(args.output, people_dir), payload)
    preview = {
        "bestMatch": {
            "name": best_match.suspect.name,
            "surname": best_match.suspect.surname,
            "birthYear": best_match.suspect.birth_year,
            "accessLevel": best_match.access_level,
            "powerPlant": best_match.power_plant.code,
            "powerPlantCity": best_match.power_plant.city,
            "distanceKm": round(best_match.distance_km, 3),
        },
        "payload": payload,
    }
    logger.info("Preview:\n{}", json.dumps(preview, ensure_ascii=False, indent=2))

    if args.verify and not args.dry_run:
        response = submit_task_answer(
            AG3NTS_VERIFY_URL,
            api_key=config.hub_api_key,
            task=TASK_NAME,
            answer=payload["answer"],
            timeout_seconds=API_TIMEOUT_SECONDS,
        )
        logger.info("Verify response:\n{}", json.dumps(response, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
