"""Catalog loading and item matching for the negotiations tool."""

from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.request import urlopen


STOPWORDS = {
    "a",
    "aby",
    "albo",
    "bede",
    "by",
    "byc",
    "chce",
    "do",
    "dla",
    "i",
    "jest",
    "kazdego",
    "kupic",
    "mi",
    "na",
    "nam",
    "o",
    "od",
    "oraz",
    "potrzeba",
    "potrzebuje",
    "potrzebujemy",
    "prosze",
    "sama",
    "sam",
    "sie",
    "szukam",
    "to",
    "uruchomienia",
    "wiatrowej",
    "z",
    "ze",
}

PHRASE_SYNONYMS = {
    "przetwornica": "inwerter",
    "przetwornice": "inwerter",
    "bateria": "akumulator",
    "baterie": "akumulator",
    "wirek": "turbina",
    "generator": "turbina",
}

TOKEN_SYNONYMS = {
    "agm": {"agm"},
    "ah": {"ah"},
    "akumulator": {"akumulator", "bateria"},
    "dc": {"dc"},
    "inwerter": {"inwerter", "przetwornica"},
    "kwasowy": {"kwasowy"},
    "wiatrowa": {"wiatrowa", "wiatrak", "wiatrowy"},
    "turbina": {"turbina", "generator", "wiatrak"},
}

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
UNIT_PATTERN = re.compile(r"(?P<value>\d+(?:[.,]\d+)?)\s*(?P<unit>v|w|ah|a)")
DOMAIN_STEMS = {
    "akumulator": "akumulator",
    "bateri": "akumulator",
    "inwerter": "inwerter",
    "przetwornic": "inwerter",
    "turbin": "turbina",
    "wiatr": "wiatrowa",
}


@dataclass(frozen=True, slots=True)
class ItemRecord:
    """Catalog entry enriched with normalized keywords."""

    code: str
    name: str
    keywords: frozenset[str]
    normalized_name: str


@dataclass(frozen=True, slots=True)
class CatalogMatch:
    """Single item match result."""

    item: ItemRecord
    cities: tuple[str, ...]
    score: float


@dataclass(frozen=True, slots=True)
class Catalog:
    """Complete in-memory catalog."""

    items: tuple[ItemRecord, ...]
    cities_by_item_code: dict[str, tuple[str, ...]]

    def find_best_match(self, query: str) -> CatalogMatch | None:
        """Return the best matching item for a natural-language query."""

        normalized_query = normalize_text(query)
        query_keywords = build_keywords(query)
        if not query_keywords:
            return None

        best_match: CatalogMatch | None = None
        for item in self.items:
            overlap = query_keywords & item.keywords
            if not overlap:
                continue
            score = score_keywords(query_keywords, item, overlap, normalized_query)
            cities = self.cities_by_item_code.get(item.code, ())
            candidate = CatalogMatch(item=item, cities=cities, score=score)
            if best_match is None or candidate.score > best_match.score:
                best_match = candidate
        return best_match


def normalize_text(value: str) -> str:
    """Lowercase text, remove diacritics, and collapse separators."""

    text = unicodedata.normalize("NFKD", value.casefold())
    ascii_text = "".join(char for char in text if not unicodedata.combining(char))
    ascii_text = ascii_text.replace("/", " ")
    ascii_text = re.sub(r"[^a-z0-9.,+\-\s]", " ", ascii_text)
    return re.sub(r"\s+", " ", ascii_text).strip()


def tokenize(value: str) -> list[str]:
    """Split normalized text into searchable tokens."""

    return TOKEN_PATTERN.findall(normalize_text(value))


def build_keywords(value: str) -> frozenset[str]:
    """Build a keyword set from raw item or query text."""

    normalized = normalize_text(value)
    for source, target in PHRASE_SYNONYMS.items():
        normalized = normalized.replace(source, target)

    keywords: set[str] = set()
    for token in TOKEN_PATTERN.findall(normalized):
        if token in STOPWORDS:
            continue
        keywords.add(token)
        if canonical := canonical_token(token):
            keywords.add(canonical)

    for match in UNIT_PATTERN.finditer(normalized):
        number = match.group("value").replace(",", ".")
        unit = canonical_token(match.group("unit"))
        keywords.add(f"{number}{unit}")
        keywords.add(number)
        keywords.add(unit)

    return frozenset(keywords)


def canonical_token(token: str) -> str:
    """Map a token to its canonical searchable form."""

    lowered = token.casefold()
    for stem, canonical in DOMAIN_STEMS.items():
        if lowered.startswith(stem):
            return canonical
    for canonical, variants in TOKEN_SYNONYMS.items():
        if lowered in variants:
            return canonical
    return lowered


def score_keywords(
    query_keywords: frozenset[str],
    item: ItemRecord,
    overlap: frozenset[str] | set[str],
    normalized_query: str,
) -> float:
    """Score a candidate item against the query keywords."""

    score = 0.0
    for token in overlap:
        if token in {"48v", "24v", "12v", "400w", "3000w", "150ah", "200ah"}:
            score += 5.0
            continue
        if token.isdigit():
            score += 0.8
            continue
        if re.search(r"\d", token):
            score += 4.0
            continue
        if token in {"inwerter", "akumulator", "turbina"}:
            score += 3.5
            continue
        score += 1.4

    if item.normalized_name in normalized_query:
        score += 8.0

    coverage = len(overlap) / max(1, len(query_keywords))
    return score + coverage


def format_match_output(match: CatalogMatch) -> str:
    """Format a concise tool response that stays under the task limit."""

    cities = ", ".join(match.cities) if match.cities else "brak"
    output = f"match={match.item.name} [{match.item.code}]; cities={cities}"
    if len(output.encode("utf-8")) <= 500:
        return output

    trimmed = []
    total = len(f"match={match.item.name} [{match.item.code}]; cities=".encode("utf-8"))
    for city in match.cities:
        city_bytes = city.encode("utf-8")
        separator = 2 if trimmed else 0
        if total + separator + len(city_bytes) > 480:
            break
        trimmed.append(city)
        total += separator + len(city_bytes)
    cities_part = ", ".join(trimmed) if trimmed else "brak"
    return f"match={match.item.name} [{match.item.code}]; cities={cities_part}"


def ensure_dataset(
    *,
    data_base_url: str,
    cache_dir: Path,
    timeout_seconds: float,
) -> tuple[Path, Path, Path]:
    """Download CSV files into the cache directory if needed."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    names = ("cities.csv", "items.csv", "connections.csv")
    paths: list[Path] = []
    for file_name in names:
        local_path = cache_dir / file_name
        if not local_path.exists():
            remote_url = f"{data_base_url}/{file_name}"
            with urlopen(remote_url, timeout=timeout_seconds) as response:
                local_path.write_bytes(response.read())
        paths.append(local_path)
    return paths[0], paths[1], paths[2]


def load_catalog_from_csv(
    cities_path: Path,
    items_path: Path,
    connections_path: Path,
) -> Catalog:
    """Load all CSV data into searchable structures."""

    cities_by_code = read_simple_csv(cities_path)
    items_by_code = read_simple_csv(items_path)
    city_codes_by_item = read_connections(connections_path)

    items: list[ItemRecord] = []
    cities_by_item_code: dict[str, tuple[str, ...]] = {}
    for item_code, item_name in items_by_code.items():
        item_cities = tuple(
            sorted(
                cities_by_code[city_code]
                for city_code in city_codes_by_item.get(item_code, set())
                if city_code in cities_by_code
            )
        )
        items.append(
            ItemRecord(
                code=item_code,
                name=item_name,
                keywords=build_keywords(item_name),
                normalized_name=normalize_text(item_name),
            )
        )
        cities_by_item_code[item_code] = item_cities

    return Catalog(items=tuple(items), cities_by_item_code=cities_by_item_code)


def read_simple_csv(path: Path) -> dict[str, str]:
    """Read `{name,code}` CSV files into a code->name mapping."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {row["code"]: row["name"] for row in reader if row.get("code") and row.get("name")}


def read_connections(path: Path) -> dict[str, set[str]]:
    """Read `{itemCode,cityCode}` CSV into a lookup map."""

    mapping: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            item_code = (row.get("itemCode") or "").strip()
            city_code = (row.get("cityCode") or "").strip()
            if not item_code or not city_code:
                continue
            mapping.setdefault(item_code, set()).add(city_code)
    return mapping


@lru_cache(maxsize=1)
def load_catalog(
    data_base_url: str,
    cache_dir: str,
    timeout_seconds: float,
) -> Catalog:
    """Load and cache the public catalog."""

    cities_path, items_path, connections_path = ensure_dataset(
        data_base_url=data_base_url,
        cache_dir=Path(cache_dir),
        timeout_seconds=timeout_seconds,
    )
    return load_catalog_from_csv(cities_path, items_path, connections_path)
