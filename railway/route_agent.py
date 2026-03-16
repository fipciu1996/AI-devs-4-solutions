"""Control railway routes through OpenRouter tool calling and shared env config."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from repo_env import get_env, get_optional_env, load_repo_env


load_repo_env(__file__)


try:
    from loguru import logger
except ImportError as exc:
    raise SystemExit(
        "Brak zaleznosci 'loguru'. Zainstaluj ja poleceniem: pip install loguru"
    ) from exc


OPENROUTER_API_URL = get_env("OPENROUTER_BASE_URL")
DEFAULT_MODEL = "openrouter/healer-alpha"
DEFAULT_MAX_STEPS = 8
DEFAULT_RAILWAY_RETRY_ATTEMPTS = 3
DEFAULT_503_RETRY_DELAY_SECONDS = 30
RAILWAY_TASK_NAME = "railway"
RETRYABLE_STATUS_CODES = {429, 503}
ROUTE_PATTERN = re.compile(r"^[a-z]-[0-9]{1,2}$", re.IGNORECASE)
STATUS_VALUES = {"RTOPEN", "RTCLOSE"}

SYSTEM_PROMPT = """Jestes agentem do zmiany statusu tras kolejowych.
Masz dostep wylacznie do narzedzi API i zawsze uzywasz ich do sprawdzania oraz zmiany stanu.

Zasady pracy:
- Gdy uzytkownik prosi o zmiane statusu trasy, najpierw sprawdz aktualny stan.
- Aby zmienic status, zawsze wykonaj kolejnosc: reconfigure -> setstatus -> save.
- Nie wywoluj setstatus bez wczesniejszego reconfigure.
- Jesli status jest juz zgodny z celem uzytkownika, poinformuj o tym zamiast wykonywac zbedne zmiany.
- Jesli polecenie uzytkownika jest niejednoznaczne, najpierw pobierz dostepne informacje narzedziami i krotko opisz problem.
- Odpowiadaj po polsku.
"""


class RailwayError(RuntimeError):
    """Raised when the railway API or OpenRouter returns an error."""


@dataclass(slots=True)
class AppConfig:
    railway_api_url: str
    railway_api_key: str
    openrouter_api_key: str
    model: str
    site_url: str | None
    site_name: str | None
    max_steps: int


def configure_logging(show_tool_results: bool) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG" if show_tool_results else "INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | <level>{message}</level>",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Agent OpenRouter do zarzadzania statusami tras kolejowych za pomoca "
            "tool callingu."
        )
    )
    parser.add_argument(
        "prompt",
        nargs="+",
        help="Polecenie dla modelu, np. 'Zamknij trase a-3'.",
    )
    parser.add_argument(
        "--railway-api-url",
        help="Endpoint API tras. Mozna tez ustawic RAILWAY_API_URL.",
    )
    parser.add_argument(
        "--railway-api-key",
        help="Klucz API tras. Mozna tez ustawic RAILWAY_API_KEY.",
    )
    parser.add_argument(
        "--openrouter-api-key",
        help="Klucz OpenRouter. Mozna tez ustawic OPENROUTER_API_KEY.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"Model OpenRouter. Domyslnie: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--site-url",
        default=None,
        help="Opcjonalny naglowek HTTP-Referer dla OpenRouter.",
    )
    parser.add_argument(
        "--site-name",
        default=None,
        help="Opcjonalny naglowek X-Title dla OpenRouter.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"Maksymalna liczba iteracji tool callingu. Domyslnie: {DEFAULT_MAX_STEPS}",
    )
    parser.add_argument(
        "--show-tool-results",
        action="store_true",
        help="Wypisz po drodze wyniki narzedzi.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> AppConfig:
    def read_value(cli_value: str | None, env_name: str) -> str:
        return (cli_value or os.environ.get(env_name) or "").strip()

    railway_api_url = read_value(args.railway_api_url, "RAILWAY_API_URL")
    railway_api_key = read_value(args.railway_api_key, "RAILWAY_API_KEY")
    openrouter_api_key = read_value(args.openrouter_api_key, "OPENROUTER_API_KEY")
    model = (args.model or get_env("OPENROUTER_MODEL") or DEFAULT_MODEL).strip()
    site_url = (args.site_url or get_optional_env("OPENROUTER_SITE_URL") or "").strip() or None
    site_name = (args.site_name or get_optional_env("OPENROUTER_SITE_NAME") or "").strip() or None

    missing: list[str] = []
    if not railway_api_url:
        missing.append("RAILWAY_API_URL")
    if not railway_api_key:
        missing.append("RAILWAY_API_KEY")
    if not openrouter_api_key:
        missing.append("OPENROUTER_API_KEY")

    if missing:
        joined = ", ".join(missing)
        raise SystemExit(f"Brakuje wymaganych ustawien: {joined}")
    if args.max_steps < 1:
        raise SystemExit("--max-steps musi byc dodatnie.")

    return AppConfig(
        railway_api_url=railway_api_url,
        railway_api_key=railway_api_key,
        openrouter_api_key=openrouter_api_key,
        model=model,
        site_url=site_url,
        site_name=site_name,
        max_steps=args.max_steps,
    )


def require_route(route: str) -> str:
    candidate = route.strip()
    if not ROUTE_PATTERN.fullmatch(candidate):
        raise RailwayError(
            'Niepoprawny format trasy. Oczekiwano formatu "[a-z]-[0-9]{1,2}".'
        )
    return candidate.lower()


def require_status(value: str) -> str:
    candidate = value.strip().upper()
    if candidate not in STATUS_VALUES:
        raise RailwayError('Niepoprawny status. Dozwolone wartosci: "RTOPEN", "RTCLOSE".')
    return candidate


class RailwayApiClient:
    def __init__(
        self,
        api_url: str,
        api_key: str,
        *,
        retry_attempts: int = DEFAULT_RAILWAY_RETRY_ATTEMPTS,
    ) -> None:
        self.api_url = api_url
        self.api_key = api_key
        self.retry_attempts = retry_attempts

    def call(
        self,
        *,
        action: str,
        route: str | None = None,
        value: str | None = None,
    ) -> dict[str, Any]:
        answer: dict[str, str] = {
            "action": action,
        }
        if route is not None:
            answer["route"] = require_route(route)
        if value is not None:
            answer["value"] = require_status(value)

        payload: dict[str, Any] = {
            "apikey": self.api_key,
            "task": RAILWAY_TASK_NAME,
            "answer": answer,
        }

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        retries_left = self.retry_attempts

        while True:
            http_request = request.Request(
                url=self.api_url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            try:
                with request.urlopen(http_request, timeout=60) as response:
                    raw = response.read().decode("utf-8")
                break
            except error.HTTPError as exc:
                details = exc.read().decode("utf-8", errors="replace")
                if exc.code in RETRYABLE_STATUS_CODES and retries_left > 0:
                    retry_delay = self._extract_retry_delay(exc, details)
                    logger.warning(
                        "Railway API zwrocilo {} dla akcji {}. Czekam {} s i ponawiam "
                        "probe (pozostalo prob: {}).",
                        exc.code,
                        action,
                        retry_delay,
                        retries_left,
                    )
                    time.sleep(retry_delay)
                    retries_left -= 1
                    continue
                raise RailwayError(f"HTTP {exc.code} dla Railway API: {details}") from exc
            except error.URLError as exc:
                raise RailwayError(f"Blad polaczenia z Railway API: {exc.reason}") from exc

        try:
            result = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RailwayError(f"Railway API zwrocilo niepoprawny JSON: {raw}") from exc

        if not isinstance(result, dict):
            raise RailwayError(f"Railway API zwrocilo niepoprawny typ danych: {type(result)!r}")

        if result.get("ok") is False:
            message = result.get("error") or result.get("message") or raw
            raise RailwayError(f"Railway API zwrocilo blad: {message}")

        return result

    def _extract_retry_delay(self, http_error: error.HTTPError, details: str) -> int:
        retry_after_header = http_error.headers.get("Retry-After")
        if retry_after_header:
            try:
                return max(1, int(retry_after_header))
            except ValueError:
                logger.debug("Nie udalo sie sparsowac naglowka Retry-After: {}", retry_after_header)

        try:
            payload = json.loads(details)
        except json.JSONDecodeError:
            payload = None

        if isinstance(payload, dict):
            retry_after = payload.get("retry_after")
            if isinstance(retry_after, int) and retry_after > 0:
                return retry_after
            penalty_seconds = payload.get("penalty_seconds")
            if isinstance(penalty_seconds, int) and penalty_seconds > 0:
                return penalty_seconds

        if http_error.code == 503:
            return DEFAULT_503_RETRY_DELAY_SECONDS

        return 1


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "help",
            "description": "Pokazuje dostepne akcje Railway API i ich parametry.",
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
            "name": "reconfigure",
            "description": "Wlacza tryb rekonfiguracji dla wskazanej trasy.",
            "parameters": {
                "type": "object",
                "properties": {
                    "route": {
                        "type": "string",
                        "description": 'Trasa w formacie "[a-z]-[0-9]{1,2}".',
                    }
                },
                "required": ["route"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "getstatus",
            "description": "Pobiera aktualny status wskazanej trasy.",
            "parameters": {
                "type": "object",
                "properties": {
                    "route": {
                        "type": "string",
                        "description": 'Trasa w formacie "[a-z]-[0-9]{1,2}".',
                    }
                },
                "required": ["route"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "setstatus",
            "description": (
                "Ustawia status trasy w trybie rekonfiguracji. "
                'Dozwolone wartosci: "RTOPEN", "RTCLOSE".'
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "route": {
                        "type": "string",
                        "description": 'Trasa w formacie "[a-z]-[0-9]{1,2}".',
                    },
                    "value": {
                        "type": "string",
                        "enum": ["RTOPEN", "RTCLOSE"],
                        "description": "Docelowy status trasy.",
                    },
                },
                "required": ["route", "value"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save",
            "description": "Zapisuje zmiany i wychodzi z trybu rekonfiguracji.",
            "parameters": {
                "type": "object",
                "properties": {
                    "route": {
                        "type": "string",
                        "description": 'Trasa w formacie "[a-z]-[0-9]{1,2}".',
                    }
                },
                "required": ["route"],
                "additionalProperties": False,
            },
        },
    },
]


def build_tool_handlers(client: RailwayApiClient) -> dict[str, Any]:
    return {
        "help": lambda _: client.call(action="help"),
        "reconfigure": lambda args: client.call(action="reconfigure", route=str(args["route"])),
        "getstatus": lambda args: client.call(action="getstatus", route=str(args["route"])),
        "setstatus": lambda args: client.call(
            action="setstatus",
            route=str(args["route"]),
            value=str(args["value"]),
        ),
        "save": lambda args: client.call(action="save", route=str(args["route"])),
    }


def post_json(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    http_request = request.Request(
        url=url,
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with request.urlopen(http_request, timeout=120) as response:
            raw = response.read().decode("utf-8")
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RailwayError(f"HTTP {exc.code} dla OpenRouter: {details}") from exc
    except error.URLError as exc:
        raise RailwayError(f"Blad polaczenia z OpenRouter: {exc.reason}") from exc

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RailwayError(f"OpenRouter zwrocil niepoprawny JSON: {raw}") from exc

    if not isinstance(result, dict):
        raise RailwayError("OpenRouter zwrocil niepoprawny typ odpowiedzi.")
    if result.get("error"):
        raise RailwayError(str(result["error"]))
    return result


def call_openrouter(
    *,
    config: AppConfig,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "tools": TOOLS,
        "tool_choice": "auto",
    }
    headers = {
        "Authorization": f"Bearer {config.openrouter_api_key}",
        "Content-Type": "application/json",
    }
    if config.site_url:
        headers["HTTP-Referer"] = config.site_url
    if config.site_name:
        headers["X-Title"] = config.site_name

    return post_json(OPENROUTER_API_URL, payload, headers)


def extract_message(response_json: dict[str, Any]) -> dict[str, Any]:
    choices = response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RailwayError("Brak pola 'choices' w odpowiedzi OpenRouter.")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise RailwayError("Niepoprawny element 'choices' w odpowiedzi OpenRouter.")

    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise RailwayError("Brak pola 'message' w odpowiedzi OpenRouter.")
    return message


def extract_text_content(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n".join(parts)
    return ""


def execute_tool_call(
    tool_call: dict[str, Any],
    handlers: dict[str, Any],
    *,
    show_tool_results: bool,
) -> dict[str, Any]:
    function_block = tool_call.get("function")
    if not isinstance(function_block, dict):
        raise RailwayError(f"Niepoprawny format tool call: {tool_call}")

    name = function_block.get("name")
    if not isinstance(name, str) or name not in handlers:
        raise RailwayError(f"Model wywolal nieznane narzedzie: {name!r}")

    arguments_raw = function_block.get("arguments") or "{}"
    if not isinstance(arguments_raw, str):
        raise RailwayError(f"Niepoprawny format argumentow narzedzia {name}.")

    try:
        arguments = json.loads(arguments_raw)
    except json.JSONDecodeError as exc:
        raise RailwayError(f"Nie udalo sie sparsowac argumentow narzedzia {name}.") from exc

    if not isinstance(arguments, dict):
        raise RailwayError(f"Argumenty narzedzia {name} musza byc obiektem JSON.")

    result = handlers[name](arguments)
    if show_tool_results:
        logger.debug(
            "Tool {} wywolany z argumentami: {}",
            name,
            json.dumps(arguments, ensure_ascii=False),
        )
        logger.debug(
            "Wynik toola {}: {}",
            name,
            json.dumps(result, ensure_ascii=False, indent=2),
        )

    tool_call_id = tool_call.get("id")
    if not isinstance(tool_call_id, str) or not tool_call_id:
        raise RailwayError(f"Brak identyfikatora wywolania narzedzia dla {name}.")

    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": name,
        "content": json.dumps(result, ensure_ascii=False),
    }


def run_agent(
    *,
    config: AppConfig,
    prompt: str,
    client: RailwayApiClient,
    show_tool_results: bool,
) -> str:
    handlers = build_tool_handlers(client)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    for _ in range(config.max_steps):
        response = call_openrouter(config=config, messages=messages)
        message = extract_message(response)
        tool_calls = message.get("tool_calls")

        assistant_message: dict[str, Any] = {"role": "assistant"}
        text_content = extract_text_content(message)
        if text_content:
            assistant_message["content"] = text_content
        else:
            assistant_message["content"] = ""

        if isinstance(tool_calls, list) and tool_calls:
            assistant_message["tool_calls"] = tool_calls
            messages.append(assistant_message)
            for tool_call in tool_calls:
                tool_message = execute_tool_call(
                    tool_call,
                    handlers,
                    show_tool_results=show_tool_results,
                )
                messages.append(tool_message)
            continue

        if text_content:
            return text_content

        raise RailwayError("Model nie zwrocil ani odpowiedzi tekstowej, ani tool calli.")

    raise RailwayError(
        f"Tool calling nie zakonczyl sie w limicie {config.max_steps} iteracji."
    )


def main() -> int:
    if not OPENROUTER_API_URL:
        logger.error("Brakuje OPENROUTER_BASE_URL w .env.")
        return 1
    args = parse_args()
    configure_logging(args.show_tool_results)
    config = build_config(args)
    prompt = " ".join(args.prompt).strip()
    client = RailwayApiClient(
        api_url=config.railway_api_url,
        api_key=config.railway_api_key,
    )

    try:
        answer = run_agent(
            config=config,
            prompt=prompt,
            client=client,
            show_tool_results=args.show_tool_results,
        )
    except RailwayError as exc:
        logger.error("Blad: {}", exc)
        return 1

    logger.info("Odpowiedz agenta: {}", answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
