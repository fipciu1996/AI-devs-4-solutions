"""Official `/verify` client and response helpers for timetravel."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any
import unicodedata

from devs_utilities.ag3nts import AG3NTS_VERIFY_URL, submit_task_answer
from devs_utilities.flags import extract_flag

from .models import DeviceConfig


GUIDANCE_KEY_FRAGMENTS = (
    "needconfig",
    "guidance",
    "hint",
    "recommend",
    "advice",
    "message",
)

POLISH_NUMBER_TOKENS = {
    "zero": 0,
    "jeden": 1,
    "jedna": 1,
    "jedno": 1,
    "dwa": 2,
    "dwie": 2,
    "trzy": 3,
    "cztery": 4,
    "pięć": 5,
    "szesc": 6,
    "sześć": 6,
    "siedem": 7,
    "osiem": 8,
    "dziewięć": 9,
    "dziewiec": 9,
    "dziesięć": 10,
    "dziesiec": 10,
    "jedenaście": 11,
    "jedenascie": 11,
    "dwanaście": 12,
    "dwanascie": 12,
    "trzynaście": 13,
    "trzynascie": 13,
    "czternaście": 14,
    "czternascie": 14,
    "piętnaście": 15,
    "pietnascie": 15,
    "szesnaście": 16,
    "szesnascie": 16,
    "siedemnaście": 17,
    "siedemnascie": 17,
    "osiemnaście": 18,
    "osiemnascie": 18,
    "dziewiętnaście": 19,
    "dziewietnascie": 19,
    "dwadzieścia": 20,
    "dwadziescia": 20,
    "trzydzieści": 30,
    "trzydziesci": 30,
    "czterdzieści": 40,
    "czterdziesci": 40,
    "pięćdziesiąt": 50,
    "piecdziesiat": 50,
    "pięćdziesiat": 50,
    "szesdziesiąt": 60,
    "sześćdziesiąt": 60,
    "szescdziesiat": 60,
    "siedemdziesiąt": 70,
    "siedemdziesiat": 70,
    "osiemdziesiąt": 80,
    "osiemdziesiat": 80,
    "dziewięćdziesiąt": 90,
    "dziewiecdziesiat": 90,
    "sto": 100,
    "dwieście": 200,
    "dwiescie": 200,
    "trzysta": 300,
    "czterysta": 400,
    "pięćset": 500,
    "piecset": 500,
    "sześćset": 600,
    "szescset": 600,
    "siedemset": 700,
    "osiemset": 800,
    "dziewięćset": 900,
    "dziewiecset": 900,
    "tysiąc": 1000,
    "tysiac": 1000,
}

NUMBER_WORD_PATTERN = "|".join(
    sorted((re.escape(token) for token in POLISH_NUMBER_TOKENS), key=len, reverse=True)
)
NUMBER_MENTION_PATTERN = re.compile(
    rf"\b\d+\b|\b(?:{NUMBER_WORD_PATTERN})(?:[\s-]+(?:{NUMBER_WORD_PATTERN}))*\b",
    flags=re.IGNORECASE,
)

ADD_KEYWORDS = ("dodac", "podniesc", "podniesienie", "zwiekszyc", "zwiekszenie")
SUBTRACT_KEYWORDS = ("odjac", "obnizyc", "obnizenie")


@dataclass(frozen=True, slots=True)
class NumberMention:
    value: int
    start: int
    end: int


class TimetravelApiClient:
    """Small typed wrapper around the public timetravel `/verify` API."""

    def __init__(
        self,
        *,
        api_key: str,
        url: str = AG3NTS_VERIFY_URL,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.api_key = api_key
        self.url = url
        self.timeout_seconds = timeout_seconds

    def help(self) -> dict[str, Any]:
        """Call the `help` action."""

        return self._call({"action": "help"})

    def get_config(self) -> DeviceConfig:
        """Return the normalized device configuration."""

        response = self._call({"action": "getConfig"})
        config_payload = response.get("config")
        if not isinstance(config_payload, dict):
            raise RuntimeError(f"Missing config payload in response: {response!r}")
        return DeviceConfig.from_payload(config_payload)

    def reset(self) -> dict[str, Any]:
        """Reset the device to a clean state."""

        return self._call({"action": "reset"})

    def configure(self, param: str, value: Any) -> dict[str, Any]:
        """Set a configurable backend parameter."""

        return self._call(
            {
                "action": "configure",
                "param": param,
                "value": value,
            }
        )

    def _call(self, answer: dict[str, Any]) -> dict[str, Any]:
        response = submit_task_answer(
            self.url,
            api_key=self.api_key,
            task="timetravel",
            answer=answer,
            timeout_seconds=self.timeout_seconds,
        )
        if isinstance(response, dict):
            return response
        return {"raw": response}


def extract_stabilization_value(payload: Any) -> int | None:
    """Find a stabilization value inside structured or text responses."""

    if isinstance(payload, str):
        return _extract_stabilization_from_text(payload)

    for guidance_text in _collect_guidance_texts(payload):
        extracted = _extract_stabilization_from_text(guidance_text)
        if extracted is not None:
            return extracted

    return _extract_guidance_stabilization(payload)


def extract_flag_from_ui_text(value: str | None) -> str | None:
    """Normalize a potential UI flag string."""

    if not value:
        return None
    return extract_flag(value.strip())


def _extract_guidance_stabilization(
    payload: Any,
    *,
    in_guidance_context: bool = False,
) -> int | None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            normalized_key = key.casefold().replace("_", "")
            next_guidance_context = in_guidance_context or any(
                fragment in normalized_key for fragment in GUIDANCE_KEY_FRAGMENTS
            )
            if next_guidance_context and "stabil" in normalized_key:
                if isinstance(value, int) and 0 <= value <= 1000:
                    return value
                if isinstance(value, float) and value.is_integer() and 0 <= value <= 1000:
                    return int(value)
                if isinstance(value, str):
                    parsed = _extract_stabilization_from_text(value)
                    if parsed is not None:
                        return parsed
            nested = _extract_guidance_stabilization(
                value,
                in_guidance_context=next_guidance_context,
            )
            if nested is not None:
                return nested
    if isinstance(payload, list):
        for item in payload:
            nested = _extract_guidance_stabilization(item, in_guidance_context=in_guidance_context)
            if nested is not None:
                return nested
    return None


def _collect_guidance_texts(payload: Any) -> list[str]:
    if isinstance(payload, str):
        return [payload]
    if isinstance(payload, dict):
        collected: list[str] = []
        for key, value in payload.items():
            normalized_key = key.casefold().replace("_", "")
            if any(fragment in normalized_key for fragment in GUIDANCE_KEY_FRAGMENTS):
                if isinstance(value, str):
                    collected.append(value)
                else:
                    collected.extend(_collect_guidance_texts(value))
        return collected
    if isinstance(payload, list):
        collected = []
        for item in payload:
            collected.extend(_collect_guidance_texts(item))
        return collected
    return []


def _extract_stabilization_from_text(payload: Any) -> int | None:
    if not isinstance(payload, str):
        return None

    normalized_payload = _normalize_text(payload)
    direct_patterns = (
        r"stabilization[^0-9]{0,24}(\d{1,4})",
        r"ustaw[^0-9]{0,24}stabilization[^0-9]{0,24}(\d{1,4})",
        r"set[^0-9]{0,24}stabilization[^0-9]{0,24}(\d{1,4})",
    )
    for pattern in direct_patterns:
        match = re.search(pattern, normalized_payload, flags=re.IGNORECASE)
        if not match:
            continue
        value = int(match.group(1))
        if 0 <= value <= 1000:
            return value

    mentions = _extract_number_mentions(normalized_payload)
    if not mentions:
        return None
    if len(mentions) == 1:
        return mentions[0].value

    first, second = mentions[0], mentions[1]
    between = normalized_payload[first.end : second.start]
    if any(keyword in between for keyword in ADD_KEYWORDS):
        return first.value + second.value
    if any(keyword in between for keyword in SUBTRACT_KEYWORDS):
        return first.value - second.value
    return None


def _extract_number_mentions(text: str) -> list[NumberMention]:
    mentions: list[NumberMention] = []
    for match in NUMBER_MENTION_PATTERN.finditer(text):
        value = _parse_number_phrase(match.group(0))
        if value is None or not (0 <= value <= 1000):
            continue
        mentions.append(NumberMention(value=value, start=match.start(), end=match.end()))
    return mentions


def _parse_number_phrase(phrase: str) -> int | None:
    stripped = phrase.strip()
    if not stripped:
        return None
    if stripped.isdigit():
        return int(stripped)

    total = 0
    for raw_token in re.split(r"[\s-]+", _normalize_text(stripped)):
        token = raw_token.strip(",.()")
        if not token:
            continue
        value = POLISH_NUMBER_TOKENS.get(token)
        if value is None:
            return None
        total += value
    return total if total else None


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(character for character in normalized if not unicodedata.combining(character)).casefold()
