"""Build and upload the `filesystem` task structure for the AG3NTS hub."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import ZipFile

REPO_ROOT_HINT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_HINT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_HINT))

from devs_utilities.ag3nts import AG3NTS_VERIFY_URL, build_ag3nts_public_data_url
from devs_utilities.bootstrap import bootstrap_repo
from devs_utilities.files import write_json
from devs_utilities.http import HttpRequestError, RAW_TEXT, get_bytes, post_json
from devs_utilities.logging import configure_logging, logger as shared_logger
from repo_env import get_env, get_int_env


REPO_ROOT = bootstrap_repo(__file__)
logger = shared_logger.bind(component="filesystem")


TASK_NAME = "filesystem"
OUTPUT_DIR = Path(__file__).resolve().parent
LAST_BATCH_PATH = OUTPUT_DIR / "last_batch.json"
LAST_UPLOAD_RESPONSE_PATH = OUTPUT_DIR / "last_upload_response.json"
LAST_DONE_RESPONSE_PATH = OUTPUT_DIR / "last_done_response.json"
LAST_HELP_RESPONSE_PATH = OUTPUT_DIR / "last_help_response.json"

REQUEST_TIMEOUT_SECONDS = get_int_env("AG3NTS_TIMEOUT_SECONDS", 30) or 30
NOTES_ZIP_URL = build_ag3nts_public_data_url("natan_notes.zip")
NAME_PATTERN = re.compile(r"^[a-z0-9_]+$")

CITY_DISPLAY_NAMES = {
    "brudzewo": "Brudzewo",
    "celbowo": "Celbowo",
    "darzlubie": "Darzlubie",
    "domatowo": "Domatowo",
    "karlinkowo": "Karlinkowo",
    "mechowo": "Mechowo",
    "opalino": "Opalino",
    "puck": "Puck",
}
CITY_ALIASES = {
    "brudzewo": "brudzewo",
    "brudzewa": "brudzewo",
    "celbowo": "celbowo",
    "celbowa": "celbowo",
    "darzlubie": "darzlubie",
    "darzlubiem": "darzlubie",
    "darzlubiu": "darzlubie",
    "domatowo": "domatowo",
    "domatowa": "domatowo",
    "domatowie": "domatowo",
    "karlinkowo": "karlinkowo",
    "mechowo": "mechowo",
    "opalina": "opalino",
    "opalino": "opalino",
    "puck": "puck",
    "pucka": "puck",
}
GOOD_ALIASES = {
    "butelek wody": "woda",
    "chleb": "chleb",
    "chlebow": "chleb",
    "kapusta": "kapusta",
    "kapuste": "kapusta",
    "kilof": "kilof",
    "kilofow": "kilof",
    "kilofy": "kilof",
    "kurczak": "kurczak",
    "kurczaka": "kurczak",
    "lopata": "lopata",
    "lopat": "lopata",
    "lopaty": "lopata",
    "maka": "maka",
    "makaron": "makaron",
    "makaronu": "makaron",
    "marchew": "marchew",
    "mlotek": "mlotek",
    "mlotki": "mlotek",
    "mlotkow": "mlotek",
    "porcji kurczaka": "kurczak",
    "porcji wolowiny": "wolowina",
    "porcje wolowiny": "wolowina",
    "ryz": "ryz",
    "ryzu": "ryz",
    "workow ryzu": "ryz",
    "wiertarka": "wiertarka",
    "wiertarki": "wiertarka",
    "wiertarek": "wiertarka",
    "woda": "woda",
    "wolowina": "wolowina",
    "wolowiny": "wolowina",
    "ziemniak": "ziemniak",
    "ziemniaki": "ziemniak",
    "ziemniakow": "ziemniak",
}

CITY_NEED_PATTERNS = {
    "opalino": re.compile(
        r"Opalino, niech podrzuci (?P<chleb>\d+) chlebow, "
        r"(?P<woda>\d+) butelek wody i (?P<mlotek>\d+) mlotkow",
        re.IGNORECASE,
    ),
    "domatowo": re.compile(
        r"Do Domatowa trzeba dorzucic na transport (?P<makaron>\d+) makaronu, "
        r"(?P<woda>\d+) butelek wody i (?P<lopata>\d+) lopat",
        re.IGNORECASE,
    ),
    "brudzewo": re.compile(
        r"Brudzewo: ryz (?P<ryz>\d+) workow \+ (?P<woda>\d+) butelek wody "
        r"\+ (?P<wiertarka>\d+) wiertarek",
        re.IGNORECASE,
    ),
    "darzlubie": re.compile(
        r"w Darzlubiu schodzi towar szybko, potrzeba (?P<wolowina>\d+) porcji "
        r"wolowiny, (?P<woda>\d+) butelek wody i (?P<kilof>\d+) kilofow",
        re.IGNORECASE,
    ),
    "celbowo": re.compile(
        r"Celbowo pyta o (?P<kurczak>\d+) porcji kurczaka, (?P<woda>\d+) "
        r"butelek wody i (?P<mlotek>\d+) mlotkow",
        re.IGNORECASE,
    ),
    "mechowo": re.compile(
        r"Mechowo, tam duzy zrzut: ziemniaki (?P<ziemniak>\d+) kg, kapusta "
        r"(?P<kapusta>\d+), marchew (?P<marchew>\d+) kg, woda (?P<woda>\d+) "
        r"butelek, lopaty (?P<lopata>\d+)",
        re.IGNORECASE,
    ),
    "puck": re.compile(
        r"Puck, na ten moment potrzebuja (?P<chleb>\d+) chlebow, "
        r"(?P<ryz>\d+) workow ryzu, (?P<woda>\d+) butelek wody i "
        r"(?P<wiertarka>\d+) wiertarek",
        re.IGNORECASE,
    ),
    "karlinkowo": re.compile(
        r"Karlinkowo: do uzupelnienia (?P<makaron>\d+) makaronu, "
        r"(?P<wolowina>\d+) porcje wolowiny, (?P<ziemniak>\d+) kg ziemniakow, "
        r"(?P<woda>\d+) butelek wody i (?P<kilof>\d+) kilofow",
        re.IGNORECASE,
    ),
}

CITY_MANAGER_CUES = {
    "domatowo": ("Domatowie", "Natan Rams"),
    "opalino": ("Opalina", "Iga Kapecka"),
    "brudzewo": ("Brudzewa", "Kisiel", "Rafal"),
    "darzlubie": ("Darzlubiem", "Marta Frantz"),
    "celbowo": ("Celbowa", "Oskar Radtke"),
    "mechowo": ("Mechowo", "Eliza Redmann"),
    "puck": ("Pucka", "Damian Kroll"),
    "karlinkowo": ("Karlinkowo", "Konkel", "Lena"),
}
CITY_MANAGERS = {
    "domatowo": "Natan Rams",
    "opalino": "Iga Kapecka",
    "brudzewo": "Rafal Kisiel",
    "darzlubie": "Marta Frantz",
    "celbowo": "Oskar Radtke",
    "mechowo": "Eliza Redmann",
    "puck": "Damian Kroll",
    "karlinkowo": "Lena Konkel",
}


@dataclass(frozen=True, slots=True)
class NotesBundle:
    """Raw note files loaded from the archive or an extracted directory."""

    announcements: str
    conversations: str
    transactions: str


@dataclass(frozen=True, slots=True)
class MarketplaceData:
    """Normalized task data ready to serialize into the virtual filesystem."""

    city_needs: dict[str, dict[str, int]]
    city_managers: dict[str, str]
    goods_sources: dict[str, list[str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only build the batch locally and skip remote upload.",
    )
    parser.add_argument(
        "--notes-dir",
        type=Path,
        default=None,
        help="Optional path to an extracted notes directory.",
    )
    parser.add_argument(
        "--skip-done",
        action="store_true",
        help="Upload the filesystem batch but skip the final `done` validation step.",
    )
    parser.add_argument(
        "--help-api",
        action="store_true",
        help="Print the remote filesystem API manual from the hub.",
    )
    return parser.parse_args()


def to_ascii(text: str) -> str:
    """Strip Polish diacritics while keeping ASCII letters intact."""

    translation = str.maketrans(
        {
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
    return text.translate(translation)


def normalize_token(text: str) -> str:
    """Normalize free text into a lowercase ASCII token."""

    ascii_text = to_ascii(text).lower()
    ascii_text = re.sub(r"[^a-z0-9 ]+", " ", ascii_text)
    return " ".join(ascii_text.split())


def get_api_key() -> str:
    """Read the hub API key using the local repo naming variants."""

    api_key = get_env("COURSE_API_KEY") or get_env("AG3NTS_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing COURSE_API_KEY / AG3NTS_API_KEY in the repository config."
        )
    return api_key


def submit_answer(answer: Any) -> Any:
    """Send an answer payload to the AG3NTS verify endpoint."""

    payload = {
        "apikey": get_api_key(),
        "task": TASK_NAME,
        "answer": answer,
    }
    try:
        return post_json(
            AG3NTS_VERIFY_URL,
            payload,
            timeout_seconds=REQUEST_TIMEOUT_SECONDS,
            on_decode_error=RAW_TEXT,
        )
    except HttpRequestError as exc:
        raise RuntimeError(json.dumps(exc.to_response_dict(), ensure_ascii=False)) from exc


def read_note_text_from_directory(base_dir: Path, target_name: str) -> str:
    """Read a note file from an extracted directory using ASCII-safe matching."""

    target_token = normalize_token(target_name)
    for path in base_dir.iterdir():
        if not path.is_file():
            continue
        if normalize_token(path.name) == target_token:
            return path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Missing note file in {base_dir}: {target_name}")


def read_note_text_from_zip(zip_bytes: bytes, target_name: str) -> str:
    """Read one note file from the downloaded ZIP archive."""

    target_token = normalize_token(target_name)
    with ZipFile(BytesIO(zip_bytes)) as archive:
        for info in archive.infolist():
            if normalize_token(Path(info.filename).name) != target_token:
                continue
            return archive.read(info).decode("utf-8")
    raise FileNotFoundError(f"Missing note file in ZIP archive: {target_name}")


def load_notes(notes_dir: Path | None = None) -> NotesBundle:
    """Load the Natan notes either from a local directory or from the public ZIP."""

    if notes_dir is not None:
        base_dir = notes_dir.resolve()
        return NotesBundle(
            announcements=read_note_text_from_directory(base_dir, "ogłoszenia.txt"),
            conversations=read_note_text_from_directory(base_dir, "rozmowy.txt"),
            transactions=read_note_text_from_directory(base_dir, "transakcje.txt"),
        )

    zip_bytes = get_bytes(NOTES_ZIP_URL, timeout_seconds=REQUEST_TIMEOUT_SECONDS)
    return NotesBundle(
        announcements=read_note_text_from_zip(zip_bytes, "ogłoszenia.txt"),
        conversations=read_note_text_from_zip(zip_bytes, "rozmowy.txt"),
        transactions=read_note_text_from_zip(zip_bytes, "transakcje.txt"),
    )


def normalize_city_slug(raw_name: str) -> str:
    """Convert a city name from any Polish case form into the filesystem slug."""

    token = normalize_token(raw_name)
    try:
        return CITY_ALIASES[token]
    except KeyError as exc:
        raise ValueError(f"Unsupported city name: {raw_name!r}") from exc


def normalize_good_name(raw_name: str) -> str:
    """Convert an inflected good name into the required nominative singular slug."""

    token = normalize_token(raw_name)
    try:
        return GOOD_ALIASES[token]
    except KeyError as exc:
        raise ValueError(f"Unsupported good name: {raw_name!r}") from exc


def parse_city_needs(announcements_text: str) -> dict[str, dict[str, int]]:
    """Extract the per-city demand JSON payloads from the announcements note."""

    city_needs: dict[str, dict[str, int]] = {}
    for city_slug, pattern in CITY_NEED_PATTERNS.items():
        match = pattern.search(announcements_text)
        if match is None:
            raise ValueError(f"Could not parse demand entry for {city_slug}.")
        city_needs[city_slug] = {
            normalize_good_name(name): int(value)
            for name, value in match.groupdict().items()
        }
    return city_needs


def infer_city_managers(conversations_text: str) -> dict[str, str]:
    """Resolve the trading manager assigned to each city."""

    for city_slug, cues in CITY_MANAGER_CUES.items():
        missing = [cue for cue in cues if cue not in conversations_text]
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(f"Missing manager cues for {city_slug}: {missing_text}")
    return dict(CITY_MANAGERS)


def parse_goods_sources(transactions_text: str) -> dict[str, list[str]]:
    """Aggregate which cities offer each good for sale."""

    offered_by: defaultdict[str, set[str]] = defaultdict(set)
    for raw_line in transactions_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split("->")]
        if len(parts) != 3:
            raise ValueError(f"Unexpected transaction row: {line!r}")
        source_raw, good_raw, _target_raw = parts
        source_slug = normalize_city_slug(source_raw)
        good_slug = normalize_good_name(good_raw)
        offered_by[good_slug].add(source_slug)
    return {
        good_slug: sorted(city_slugs)
        for good_slug, city_slugs in sorted(offered_by.items())
    }


def build_marketplace_data(notes: NotesBundle) -> MarketplaceData:
    """Combine all parsed note sources into one normalized structure."""

    city_needs = parse_city_needs(notes.announcements)
    city_managers = infer_city_managers(notes.conversations)
    goods_sources = parse_goods_sources(notes.transactions)
    return MarketplaceData(
        city_needs=city_needs,
        city_managers=city_managers,
        goods_sources=goods_sources,
    )


def build_city_link(city_slug: str) -> str:
    """Render a markdown link to a city file in the virtual filesystem."""

    return f"[{CITY_DISPLAY_NAMES[city_slug]}](/miasta/{city_slug})"


def build_person_file_name(full_name: str) -> str:
    """Use the recommended person-file naming convention from the task."""

    return normalize_token(full_name).replace(" ", "_")


def validate_created_names(actions: list[dict[str, Any]]) -> None:
    """Check the batch against the hub limits before uploading it."""

    created_names: set[str] = set()
    for action in actions:
        path = str(action.get("path", ""))
        if not path:
            continue
        parts = [part for part in path.split("/") if part]
        for index, part in enumerate(parts):
            if not NAME_PATTERN.fullmatch(part):
                raise ValueError(f"Path part violates the allowed name pattern: {part}")
            limit = 30 if action["action"] == "createDirectory" and index == len(parts) - 1 else 20
            if len(part) > limit:
                raise ValueError(f"Path part exceeds its max length ({limit}): {part}")
        if action["action"] not in {"createDirectory", "createFile"}:
            continue
        created_name = parts[-1]
        if created_name in created_names:
            raise ValueError(f"Duplicate created name: {created_name}")
        created_names.add(created_name)


def build_batch_actions(data: MarketplaceData) -> list[dict[str, Any]]:
    """Create the batch of remote filesystem operations."""

    actions: list[dict[str, Any]] = [{"action": "reset"}]
    for directory in ("/miasta", "/osoby", "/towary"):
        actions.append({"action": "createDirectory", "path": directory})

    for city_slug in sorted(data.city_needs):
        actions.append(
            {
                "action": "createFile",
                "path": f"/miasta/{city_slug}",
                "content": json.dumps(
                    data.city_needs[city_slug],
                    ensure_ascii=True,
                    sort_keys=True,
                ),
            }
        )

    for city_slug in sorted(data.city_managers):
        manager_name = data.city_managers[city_slug]
        actions.append(
            {
                "action": "createFile",
                "path": f"/osoby/{build_person_file_name(manager_name)}",
                "content": f"{manager_name}\n{build_city_link(city_slug)}",
            }
        )

    for good_slug, city_slugs in sorted(data.goods_sources.items()):
        actions.append(
            {
                "action": "createFile",
                "path": f"/towary/{good_slug}",
                "content": "\n".join(
                    f"- {build_city_link(city_slug)}" for city_slug in city_slugs
                ),
            }
        )

    validate_created_names(actions)
    return actions


def dump_preview(data: MarketplaceData) -> str:
    """Build a compact summary for dry runs and logs."""

    lines = [
        f"cities={len(data.city_needs)}",
        f"people={len(data.city_managers)}",
        f"goods={len(data.goods_sources)}",
    ]
    return ", ".join(lines)


def main() -> int:
    configure_logging(name="filesystem")
    args = parse_args()

    if args.help_api:
        help_response = submit_answer({"action": "help"})
        write_json(LAST_HELP_RESPONSE_PATH, help_response)
        print(json.dumps(help_response, ensure_ascii=False, indent=2))
        return 0

    notes = load_notes(args.notes_dir)
    data = build_marketplace_data(notes)
    batch_actions = build_batch_actions(data)
    write_json(LAST_BATCH_PATH, batch_actions, ensure_ascii=False)

    logger.info("Prepared filesystem batch: {}", dump_preview(data))

    if args.dry_run:
        print(json.dumps(batch_actions, ensure_ascii=False, indent=2))
        return 0

    upload_response = submit_answer(batch_actions)
    write_json(LAST_UPLOAD_RESPONSE_PATH, upload_response, ensure_ascii=False)
    logger.info("Upload response: {}", upload_response)

    if args.skip_done:
        return 0

    done_response = submit_answer({"action": "done"})
    write_json(LAST_DONE_RESPONSE_PATH, done_response, ensure_ascii=False)
    logger.info("Done response: {}", done_response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
