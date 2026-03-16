"""Find the best suspect match and prepare the access-level verification payload."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, parse, request

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from repo_env import get_env, get_optional_env, load_repo_env


load_repo_env(__file__)


OPENROUTER_API_URL = get_env("OPENROUTER_BASE_URL")
HUB_VERIFY_URL = get_env("AG3NTS_VERIFY_URL")
HUB_LOCATION_URL = get_env("AG3NTS_LOCATION_URL")
HUB_ACCESS_URL = get_env("AG3NTS_ACCESS_URL")
AG3NTS_DATA_BASE_URL = get_env("AG3NTS_DATA_BASE_URL")
TASK_NAME = "findhim"
DEFAULT_PLANTS_PATH = "findhim_locations.json"
DEFAULT_SUSPECTS_PATH = "people_result.json"
DEFAULT_OUTPUT_PATH = "findhim_result.json"


@dataclass(slots=True)
class AppConfig:
    hub_api_key: str
    openrouter_api_key: str
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
        help="Wyślij wynik na endpoint z AG3NTS_VERIFY_URL.",
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
        payload.get("hub_api_key") or os.environ.get("AG3NTS_API_KEY") or ""
    ).strip()
    openrouter_api_key = str(
        payload.get("openrouter_api_key")
        or os.environ.get("OPENROUTER_API_KEY")
        or ""
    ).strip()
    openrouter_model = str(
        payload.get("openrouter_model") or get_env("OPENROUTER_MODEL", "openai/gpt-4.1-mini")
    ).strip()
    site_url = (
        str(payload.get("site_url") or get_optional_env("OPENROUTER_SITE_URL") or "").strip()
        or None
    )
    site_name = (
        str(payload.get("site_name") or get_optional_env("OPENROUTER_SITE_NAME") or "").strip()
        or None
    )

    if args.model:
        openrouter_model = args.model

    if not hub_api_key:
        raise ValueError("Brakuje AG3NTS_API_KEY lub hub_api_key w configu.")
    if not openrouter_api_key:
        raise ValueError("Brakuje OPENROUTER_API_KEY lub openrouter_api_key w configu.")

    return AppConfig(
        hub_api_key=hub_api_key,
        openrouter_api_key=openrouter_api_key,
        openrouter_model=openrouter_model,
        site_url=site_url,
        site_name=site_name,
    )


def post_json(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(url=url, data=body, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=120) as response:
            raw = response.read().decode("utf-8")
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} dla {url}: {details}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Błąd połączenia z {url}: {exc.reason}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Odpowiedź z {url} nie jest JSON-em: {raw}") from exc


def get_json(url: str) -> dict[str, Any]:
    req = request.Request(url=url, method="GET")
    try:
        with request.urlopen(req, timeout=120) as response:
            raw = response.read().decode("utf-8")
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} dla {url}: {details}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Błąd połączenia z {url}: {exc.reason}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Odpowiedź z {url} nie jest JSON-em: {raw}") from exc


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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
    if not AG3NTS_DATA_BASE_URL:
        raise RuntimeError("Missing AG3NTS_DATA_BASE_URL in .env.")
    url = (
        f"{AG3NTS_DATA_BASE_URL.rstrip('/')}/"
        f"{parse.quote(config.hub_api_key)}/findhim_locations.json"
    )
    payload = get_json(url)
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


def geocode_power_plants(raw_plants: dict[str, Any], config: AppConfig) -> list[PowerPlant]:
    plants_section = raw_plants["power_plants"]
    city_names = list(plants_section.keys())
    prompt = (
        "Dla podanych polskich miast zwróć przybliżone współrzędne geograficzne "
        "centrów tych miejscowości. Nie podawaj objaśnień. Miasta:\n"
        + "\n".join(f"- {city}" for city in city_names)
    )
    payload = {
        "model": config.openrouter_model,
        "messages": [
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
        "response_format": {
            "type": "json_schema",
            "json_schema": build_geocode_schema(),
        },
    }
    headers = {
        "Authorization": f"Bearer {config.openrouter_api_key}",
        "Content-Type": "application/json",
    }
    if config.site_url:
        headers["HTTP-Referer"] = config.site_url
    if config.site_name:
        headers["X-Title"] = config.site_name

    response = post_json(OPENROUTER_API_URL, payload, headers)
    choices = response.get("choices") or []
    message = choices[0].get("message", {}) if choices else {}
    content = str(message.get("content") or "").strip()
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
        "apikey": config.hub_api_key,
        "name": suspect.name,
        "surname": suspect.surname,
    }
    response = post_json(HUB_LOCATION_URL, payload, {"Content-Type": "application/json"})
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
        "apikey": config.hub_api_key,
        "name": suspect.name,
        "surname": suspect.surname,
        "birthYear": suspect.birth_year,
    }
    response = post_json(HUB_ACCESS_URL, payload, {"Content-Type": "application/json"})
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
        "apikey": config.hub_api_key,
        "task": TASK_NAME,
        "answer": {
            "name": match.suspect.name,
            "surname": match.suspect.surname,
            "accessLevel": match.access_level,
            "powerPlant": match.power_plant.code,
        },
    }


def main() -> None:
    required_urls = (
        OPENROUTER_API_URL,
        HUB_VERIFY_URL,
        HUB_LOCATION_URL,
        HUB_ACCESS_URL,
    )
    if not all(required_urls):
        raise ValueError(
            "Missing OPENROUTER_BASE_URL or AG3NTS_* URLs in .env."
        )
    args = parse_args()
    config = load_config(Path(args.config), args)
    birth_years = load_birth_years(Path(args.csv))
    suspects = load_suspects(Path(args.suspects), birth_years)
    raw_plants = fetch_power_plants(config, Path(args.plants), args.refresh_plants)
    plants = geocode_power_plants(raw_plants, config)
    best_match = find_best_match(suspects, plants, config)
    payload = build_verify_payload(best_match, config)

    write_json(Path(args.output), payload)
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
    print(json.dumps(preview, ensure_ascii=False, indent=2))

    if args.verify and not args.dry_run:
        response = post_json(HUB_VERIFY_URL, payload, {"Content-Type": "application/json"})
        print("\nVerify response:")
        print(json.dumps(response, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
