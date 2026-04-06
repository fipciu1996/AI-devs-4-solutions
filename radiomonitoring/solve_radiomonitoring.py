"""Solve the AG3NTS `radiomonitoring` task with local-first data routing."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import re
import sys
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT_HINT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_HINT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_HINT))

from devs_utilities.ag3nts import AG3NTS_VERIFY_URL, submit_task_answer
from devs_utilities.bootstrap import bootstrap_repo
from devs_utilities.files import write_json
from devs_utilities.http import HttpRequestError
from devs_utilities.logging import configure_logging, logger as shared_logger
from devs_utilities.openrouter import (
    OpenRouterClient,
    OpenRouterConfig,
    OpenRouterError,
)
from repo_env import get_env, get_int_env, get_optional_env


REPO_ROOT = bootstrap_repo(__file__)
logger = shared_logger.bind(component="radiomonitoring")

TASK_NAME = "radiomonitoring"
VERIFY_TIMEOUT_SECONDS = get_int_env("AG3NTS_TIMEOUT_SECONDS", 30) or 30
OPENROUTER_TIMEOUT_SECONDS = get_int_env("OPENROUTER_TIMEOUT_SECONDS", 180) or 180
DEFAULT_VISION_MODEL = (
    get_optional_env("OPENROUTER_MODEL")
    or get_optional_env("LLM_MODEL")
    or "openai/gpt-4.1-nano"
)
AUDIO_MODEL = "openai/gpt-audio"

OUTPUT_DIR = Path(__file__).resolve().parent
CAPTURES_DIR = OUTPUT_DIR / "captures"
ANALYSIS_PATH = OUTPUT_DIR / "analysis.json"
FINAL_ANSWER_PATH = OUTPUT_DIR / "final_answer.json"
VERIFY_RESPONSE_PATH = OUTPUT_DIR / "last_verify_response.json"
MULTIMODAL_CACHE_PATH = OUTPUT_DIR / "multimodal_cache.json"

PHONE_PATTERN = re.compile(r"\b\d{3}[- ]?\d{3}[- ]?\d{3}\b")
WORD_PATTERN = re.compile(r"\b[^\W\d_]+\b", re.UNICODE)
POLISH_ASCII_MAP = str.maketrans(
    {
        "ą": "a",
        "ć": "c",
        "ę": "e",
        "ł": "l",
        "ń": "n",
        "ó": "o",
        "ś": "s",
        "ź": "z",
        "ż": "z",
        "Ą": "A",
        "Ć": "C",
        "Ę": "E",
        "Ł": "L",
        "Ń": "N",
        "Ó": "O",
        "Ś": "S",
        "Ź": "Z",
        "Ż": "Z",
    }
)

MORSE_SYMBOLS = {
    ".-": "A",
    "-...": "B",
    "-.-.": "C",
    "-..": "D",
    ".": "E",
    "..-.": "F",
    "--.": "G",
    "....": "H",
    "..": "I",
    ".---": "J",
    "-.-": "K",
    ".-..": "L",
    "--": "M",
    "-.": "N",
    "---": "O",
    ".--.": "P",
    "--.-": "Q",
    ".-.": "R",
    "...": "S",
    "-": "T",
    "..-": "U",
    "...-": "V",
    ".--": "W",
    "-..-": "X",
    "-.--": "Y",
    "--..": "Z",
    "-----": "0",
    ".----": "1",
    "..---": "2",
    "...--": "3",
    "....-": "4",
    ".....": "5",
    "-....": "6",
    "--...": "7",
    "---..": "8",
    "----.": "9",
}

POLISH_ORDINALS = {
    "pierwszy": 1,
    "drugi": 2,
    "trzeci": 3,
    "czwarty": 4,
    "piaty": 5,
    "szosty": 6,
    "siodmy": 7,
    "osmy": 8,
    "dziewiaty": 9,
    "dziesiaty": 10,
    "jedenasty": 11,
    "dwunasty": 12,
    "trzynasty": 13,
    "czternasty": 14,
    "pietnasty": 15,
    "szesnasty": 16,
    "siedemnasty": 17,
    "osiemnasty": 18,
    "dziewietnasty": 19,
    "dwudziesty": 20,
}

EXTENSION_BY_META = {
    "text/csv": ".csv",
    "application/json": ".json",
    "text/xml": ".xml",
    "audio/mpeg": ".mp3",
    "image/png": ".png",
    "image/jpeg": ".jpg",
}


@dataclass(frozen=True, slots=True)
class Capture:
    index: int
    response: dict[str, Any]
    attachment_path: str | None = None


@dataclass(frozen=True, slots=True)
class TradeEntry:
    city: str
    action: str
    goods: str
    quantity: int | None
    note: str


@dataclass(frozen=True, slots=True)
class TextClue:
    capture_index: int
    text: str
    mentioned_words: tuple[str, ...]
    phone_numbers: tuple[str, ...]
    decoded_morse: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Transmit the resolved answer back to the AG3NTS endpoint.",
    )
    return parser.parse_args()


def get_api_key() -> str:
    api_key = get_env("AG3NTS_API_KEY")
    if not api_key:
        raise RuntimeError("Missing AG3NTS_API_KEY in the repository .env file.")
    return api_key


def get_openrouter_api_key() -> str:
    api_key = get_optional_env("OPENROUTER_API_KEY") or get_optional_env("LLM_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY or LLM_API_KEY in .env.")
    return api_key


def get_openrouter_base_url() -> str:
    base_url = get_optional_env("OPENROUTER_BASE_URL") or get_optional_env("LLM_BASE_URL")
    if not base_url:
        raise RuntimeError("Missing OPENROUTER_BASE_URL or LLM_BASE_URL in .env.")
    return base_url


def request_task(answer: dict[str, Any]) -> dict[str, Any]:
    response = submit_task_answer(
        AG3NTS_VERIFY_URL,
        api_key=get_api_key(),
        task=TASK_NAME,
        answer=answer,
        timeout_seconds=VERIFY_TIMEOUT_SECONDS,
    )
    if not isinstance(response, dict):
        raise RuntimeError(f"Unexpected response payload: {response!r}")
    return response


def build_openrouter_client(model: str) -> OpenRouterClient:
    return OpenRouterClient(
        OpenRouterConfig(
            api_key=get_openrouter_api_key(),
            base_url=get_openrouter_base_url(),
            model=model,
            timeout_seconds=float(max(30, OPENROUTER_TIMEOUT_SECONDS)),
        )
    )


def normalize_text(text: str) -> str:
    translated = text.translate(POLISH_ASCII_MAP)
    decomposed = unicodedata.normalize("NFKD", translated)
    ascii_text = decomposed.encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_text.casefold().split())


def ensure_capture_dir() -> None:
    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)


def guess_attachment_extension(meta: str) -> str:
    return EXTENSION_BY_META.get(meta, ".bin")


def save_capture_response(index: int, payload: dict[str, Any]) -> None:
    write_json(CAPTURES_DIR / f"capture_{index:02d}.json", payload)


def save_attachment(index: int, meta: str, raw_bytes: bytes) -> Path:
    extension = guess_attachment_extension(meta)
    path = CAPTURES_DIR / f"capture_{index:02d}{extension}"
    path.write_bytes(raw_bytes)
    return path


def collect_session() -> list[Capture]:
    ensure_capture_dir()
    start_response = request_task({"action": "start"})
    save_capture_response(0, start_response)
    logger.info("Session started: {}", start_response.get("message", "ok"))

    captures: list[Capture] = []
    for index in range(1, 256):
        response = request_task({"action": "listen"})
        save_capture_response(index, response)
        code = int(response.get("code", 0))
        if code != 100:
            logger.info("Session ended after {} captures.", index - 1)
            break

        attachment_path: str | None = None
        attachment = response.get("attachment")
        if isinstance(attachment, str):
            meta = str(response.get("meta", "application/octet-stream"))
            raw_bytes = base64.b64decode(attachment)
            stored_path = save_attachment(index, meta, raw_bytes)
            attachment_path = str(stored_path)
            logger.info(
                "Capture {}: attachment {} ({} bytes)",
                index,
                meta,
                len(raw_bytes),
            )
        else:
            logger.info("Capture {}: transcription", index)

        captures.append(Capture(index=index, response=response, attachment_path=attachment_path))
    return captures


def decode_morse_transcription(text: str) -> str | None:
    if "(stop)" not in text:
        return None

    words: list[str] = []
    current_word: list[str] = []
    for token in text.replace("*", " ").split():
        lowered = token.strip().lower()
        if lowered == "(stop)":
            if current_word:
                words.append("".join(current_word))
                current_word = []
            continue

        symbol = lowered.replace("ti", ".").replace("ta", "-")
        if not symbol or any(character not in ".-" for character in symbol):
            continue
        current_word.append(MORSE_SYMBOLS.get(symbol, "?"))

    if current_word:
        words.append("".join(current_word))
    decoded = " ".join(part for part in words if part)
    return decoded or None


def extract_phone_numbers(text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(PHONE_PATTERN.findall(text)))


def normalize_phone_number(value: str) -> str:
    return "".join(character for character in value if character.isdigit())


def extract_words(text: str) -> tuple[str, ...]:
    words = []
    for word in WORD_PATTERN.findall(text):
        if not word[:1].isupper():
            continue
        words.append(word)
    return tuple(dict.fromkeys(words))


def build_text_clues(captures: Iterable[Capture]) -> list[TextClue]:
    clues: list[TextClue] = []
    for capture in captures:
        transcription = capture.response.get("transcription")
        if not isinstance(transcription, str):
            continue
        clues.append(
            TextClue(
                capture_index=capture.index,
                text=transcription,
                mentioned_words=extract_words(transcription),
                phone_numbers=extract_phone_numbers(transcription),
                decoded_morse=decode_morse_transcription(transcription),
            )
        )
    return clues


def parse_trade_entries(path: Path) -> list[TradeEntry]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        entries: list[TradeEntry] = []
        for row in reader:
            quantity_raw = (row.get("ilosc") or "").strip()
            entries.append(
                TradeEntry(
                    city=(row.get("miasto") or "").strip(),
                    action=(row.get("akcja") or "").strip(),
                    goods=(row.get("towar") or "").strip(),
                    quantity=int(quantity_raw) if quantity_raw else None,
                    note=(row.get("w_zamian") or "").strip(),
                )
            )
    return entries


def parse_json_attachment(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_xml_attachment(path: Path) -> list[dict[str, Any]]:
    root = ET.fromstring(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for element in root:
        row: dict[str, Any] = dict(element.attrib)
        for child in element:
            if list(child):
                row[child.tag] = [(grandchild.text or "").strip() for grandchild in child]
            else:
                row[child.tag] = (child.text or "").strip()
        if row:
            rows.append(row)
    return rows


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_attachment_summary(captures: Iterable[Capture]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "trade_entries": [],
        "json_payloads": [],
        "xml_rows": [],
        "binary_assets": [],
    }
    for capture in captures:
        if capture.attachment_path is None:
            continue
        path = Path(capture.attachment_path)
        meta = str(capture.response.get("meta", "application/octet-stream"))
        if meta == "text/csv":
            summary["trade_entries"] = [asdict(entry) for entry in parse_trade_entries(path)]
            continue
        if meta == "application/json":
            parsed = parse_json_attachment(path)
            if isinstance(parsed, list):
                summary["json_payloads"] = parsed
            else:
                summary["json_payloads"] = [parsed]
            continue
        if meta == "text/xml":
            summary["xml_rows"] = parse_xml_attachment(path)
            continue
        summary["binary_assets"].append(
            {
                "capture_index": capture.index,
                "meta": meta,
                "path": str(path),
                "sha256": sha256_file(path),
                "filesize": path.stat().st_size,
            }
        )
    return summary


def load_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def save_cache(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def parse_model_json(content: str) -> dict[str, Any]:
    normalized = strip_code_fences(content)
    try:
        parsed = json.loads(normalized)
    except json.JSONDecodeError as exc:
        raise OpenRouterError("OpenRouter did not return valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise OpenRouterError("OpenRouter returned a non-object JSON payload.")
    return parsed


def get_binary_asset(
    attachment_summary: dict[str, Any],
    *,
    meta: str,
) -> dict[str, Any] | None:
    assets = attachment_summary.get("binary_assets")
    if not isinstance(assets, list):
        return None
    for asset in assets:
        if isinstance(asset, dict) and asset.get("meta") == meta:
            return asset
    return None


def analyze_contact_image(
    path: Path,
    *,
    cache: dict[str, Any],
) -> dict[str, Any]:
    digest = sha256_file(path)
    cache_key = f"image:{digest}"
    cached = cache.get(cache_key)
    if isinstance(cached, dict):
        return cached

    image_b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    client = build_openrouter_client(DEFAULT_VISION_MODEL)
    completion = client.create_completion(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Read the note in this image. Return JSON only with keys: "
                            "contact_name, phone_number, purpose, confidence."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            }
        ]
    )
    if not completion.content:
        raise OpenRouterError("Image OCR returned no content.")
    parsed = parse_model_json(completion.content)
    cache[cache_key] = parsed
    save_cache(MULTIMODAL_CACHE_PATH, cache)
    return parsed


def analyze_audio_broadcast(
    path: Path,
    *,
    cache: dict[str, Any],
) -> dict[str, Any]:
    digest = sha256_file(path)
    cache_key = f"audio:{digest}"
    cached = cache.get(cache_key)
    if isinstance(cached, dict):
        return cached

    audio_b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    client = build_openrouter_client(AUDIO_MODEL)
    completion = client.create_completion(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Transcribe this Polish audio. Return JSON only with keys: "
                            "transcription, city_name, current_warehouses_count, "
                            "planned_warehouses_count, requested_goods, offered_goods, "
                            "confidence. Use null when unknown."
                        ),
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_b64, "format": "mp3"},
                    },
                ],
            }
        ]
    )
    if not completion.content:
        raise OpenRouterError("Audio transcription returned no content.")
    parsed = parse_model_json(completion.content)
    cache[cache_key] = parsed
    save_cache(MULTIMODAL_CACHE_PATH, cache)
    return parsed


def choose_phone_number(text_clues: Iterable[TextClue]) -> str | None:
    phones: list[str] = []
    for clue in text_clues:
        if "Syjon" not in clue.text:
            continue
        phones.extend(clue.phone_numbers)
    return normalize_phone_number(phones[0]) if phones else None


def canonical_good(good: str) -> str:
    normalized = normalize_text(good)
    if normalized in {"bydlo", "wolowina", "krowa"}:
        return "cattle"
    if normalized == "kilof":
        return "pickaxe"
    if normalized == "lopata":
        return "shovel"
    if normalized == "woda":
        return "water"
    return normalized


def build_city_trade_profiles(entries: Iterable[TradeEntry]) -> dict[str, dict[str, set[str]]]:
    profiles: dict[str, dict[str, set[str]]] = {}
    for entry in entries:
        city_profile = profiles.setdefault(entry.city, {"buys": set(), "sells": set()})
        target = "buys" if entry.action == "szuka" else "sells"
        city_profile[target].add(canonical_good(entry.goods))
    return profiles


def build_syjon_trade_signature(entries: Iterable[TradeEntry]) -> dict[str, set[str]]:
    profiles = build_city_trade_profiles(entries)
    syjon = profiles.get("Syjon")
    if syjon is None:
        raise RuntimeError("Missing Syjon trade entries in CSV data.")
    return syjon


def city_text_score(city_name: str, text_clues: Iterable[TextClue]) -> int:
    score = 0
    city_norm = normalize_text(city_name)
    for clue in text_clues:
        text_norm = normalize_text(clue.text)
        if city_norm not in text_norm:
            continue
        if "biblijn" in text_norm or "raj" in text_norm:
            score += 4
        if "miasto ocalalych" in text_norm:
            score += 3
        if "oczyszczaj" in text_norm or "technologi" in text_norm:
            score += 2
        if "bydlo" in text_norm:
            score += 1
    return score


def resolve_city_name(
    trade_entries: list[TradeEntry],
    text_clues: list[TextClue],
) -> str:
    profiles = build_city_trade_profiles(trade_entries)
    syjon = build_syjon_trade_signature(trade_entries)

    best_city = ""
    best_score = -1
    for city_name, profile in profiles.items():
        if city_name == "Syjon":
            continue
        score = 0
        score += 3 * len(syjon["buys"] & profile["buys"])
        score += 4 * len(syjon["sells"] & profile["sells"])
        score += city_text_score(city_name, text_clues)
        if score > best_score:
            best_city = city_name
            best_score = score

    if not best_city:
        raise RuntimeError("Failed to resolve the real city behind the Syjon alias.")
    return best_city


def format_area(value: Any) -> str:
    normalized = Decimal(str(value))
    quantized = normalized.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return format(quantized, ".2f")


def parse_ordinal_count(text: str) -> int | None:
    normalized = normalize_text(text)
    for ordinal, number in POLISH_ORDINALS.items():
        if f" {ordinal} magazyn" in f" {normalized}":
            return number
    return None


def resolve_warehouses_count(audio_analysis: dict[str, Any]) -> int:
    current = audio_analysis.get("current_warehouses_count")
    if isinstance(current, int):
        return current

    planned = audio_analysis.get("planned_warehouses_count")
    if isinstance(planned, int):
        return max(0, planned - 1)

    transcription = audio_analysis.get("transcription")
    if isinstance(transcription, str):
        ordinal_count = parse_ordinal_count(transcription)
        if ordinal_count is not None:
            if "wybud" in normalize_text(transcription):
                return max(0, ordinal_count - 1)
            return ordinal_count

    raise RuntimeError("Failed to resolve the current warehouse count.")


def find_keyword_rows(rows: Iterable[dict[str, Any]], keyword: str) -> list[dict[str, Any]]:
    keyword_norm = normalize_text(keyword)
    matches: list[dict[str, Any]] = []
    for row in rows:
        haystack = normalize_text(json.dumps(row, ensure_ascii=False))
        if keyword_norm in haystack:
            matches.append(row)
    return matches


def build_analysis(captures: list[Capture]) -> dict[str, Any]:
    text_clues = build_text_clues(captures)
    attachment_summary = collect_attachment_summary(captures)
    trade_entries = [TradeEntry(**entry) for entry in attachment_summary["trade_entries"]]

    multimodal_cache = load_cache(MULTIMODAL_CACHE_PATH)
    multimodal: dict[str, Any] = {}

    png_asset = get_binary_asset(attachment_summary, meta="image/png")
    if png_asset:
        multimodal["contact_note"] = analyze_contact_image(
            Path(str(png_asset["path"])),
            cache=multimodal_cache,
        )

    mp3_asset = get_binary_asset(attachment_summary, meta="audio/mpeg")
    if mp3_asset:
        multimodal["warehouse_broadcast"] = analyze_audio_broadcast(
            Path(str(mp3_asset["path"])),
            cache=multimodal_cache,
        )

    return {
        "capture_count": len(captures),
        "text_clues": [asdict(clue) for clue in text_clues],
        "attachment_summary": attachment_summary,
        "syjon_trade_signature": {
            key: sorted(values)
            for key, values in build_syjon_trade_signature(trade_entries).items()
        },
        "syjon_phone_number": choose_phone_number(text_clues),
        "syjon_mentions": [
            asdict(clue)
            for clue in text_clues
            if "Syjon" in clue.text or clue.decoded_morse is not None
        ],
        "xml_syjon_matches": find_keyword_rows(attachment_summary["xml_rows"], "syjon"),
        "multimodal": multimodal,
    }


def resolve_answer(analysis: dict[str, Any]) -> dict[str, Any]:
    attachment_summary = analysis.get("attachment_summary")
    if not isinstance(attachment_summary, dict):
        raise RuntimeError("Missing attachment summary in analysis.")

    trade_entries_raw = attachment_summary.get("trade_entries")
    json_payloads = attachment_summary.get("json_payloads")
    if not isinstance(trade_entries_raw, list) or not trade_entries_raw:
        raise RuntimeError("Missing CSV trade data.")
    if not isinstance(json_payloads, list) or not json_payloads:
        raise RuntimeError("Missing JSON city registry data.")

    trade_entries = [TradeEntry(**entry) for entry in trade_entries_raw]
    text_clues = [
        TextClue(**item)
        for item in analysis.get("text_clues", [])
        if isinstance(item, dict)
    ]
    city_name = resolve_city_name(trade_entries, text_clues)

    city_registry = {
        str(item["name"]): item
        for item in json_payloads
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    city_data = city_registry.get(city_name)
    if not isinstance(city_data, dict):
        raise RuntimeError(f"Missing registry data for city {city_name}.")

    multimodal = analysis.get("multimodal")
    if not isinstance(multimodal, dict):
        raise RuntimeError("Missing multimodal analysis data.")

    contact_note = multimodal.get("contact_note")
    if not isinstance(contact_note, dict):
        raise RuntimeError("Missing OCR result for the Syjon contact note.")
    contact_phone = contact_note.get("phone_number")
    if not isinstance(contact_phone, str) or not contact_phone.strip():
        contact_phone = analysis.get("syjon_phone_number")
    if not isinstance(contact_phone, str) or not contact_phone.strip():
        raise RuntimeError("Failed to resolve the Syjon contact phone number.")

    warehouse_broadcast = multimodal.get("warehouse_broadcast")
    if not isinstance(warehouse_broadcast, dict):
        raise RuntimeError("Missing audio analysis for warehouse count.")

    phone_number = normalize_phone_number(contact_phone)
    if len(phone_number) != 9:
        raise RuntimeError(f"Resolved phone number is invalid: {contact_phone!r}")

    return {
        "cityName": city_name,
        "cityArea": format_area(city_data["occupiedArea"]),
        "warehousesCount": resolve_warehouses_count(warehouse_broadcast),
        "phoneNumber": phone_number,
    }


def main() -> int:
    configure_logging(name="radiomonitoring")
    args = parse_args()
    try:
        captures = collect_session()
        analysis = build_analysis(captures)
        write_json(ANALYSIS_PATH, analysis)
        logger.info("Wrote analysis to {}", ANALYSIS_PATH)

        answer = resolve_answer(analysis)
        payload = {
            "apikey": get_api_key(),
            "task": TASK_NAME,
            "answer": {"action": "transmit", **answer},
        }
        write_json(FINAL_ANSWER_PATH, payload)
        logger.info("Prepared final answer payload: {}", payload["answer"])

        if not args.verify:
            return 0

        response = request_task(payload["answer"])
        write_json(VERIFY_RESPONSE_PATH, response)
        logger.success("Verify response: {}", response)
        return 0
    except HttpRequestError as exc:
        write_json(VERIFY_RESPONSE_PATH, exc.to_response_dict())
        logger.error("Network failure: {}", exc)
        return 1
    except (OpenRouterError, RuntimeError, ValueError) as exc:
        logger.error("Solver failed: {}", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
