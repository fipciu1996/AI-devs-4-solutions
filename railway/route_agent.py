"""Control railway routes through OpenRouter tool calling and shared env config."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT_HINT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_HINT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_HINT))

from devs_utilities.ag3nts import AG3NTS_RAILWAY_URL
from devs_utilities.bootstrap import bootstrap_repo
from devs_utilities.http import HttpRequestError, post_json
from devs_utilities.logging import configure_logging, logger as shared_logger
from devs_utilities.openrouter import (
    build_task_openrouter_client,
    build_task_site_name,
    OpenRouterClient,
    OpenRouterError,
    ToolCall,
)
from devs_utilities.prompts import load_prompt_text
from devs_utilities.repo_env import (
    get_course_api_key,
    get_env,
    get_int_env,
    get_llm_api_key,
    get_llm_base_url,
    get_llm_model,
    get_optional_env,
)


REPO_ROOT = bootstrap_repo(__file__)
logger = shared_logger.bind(component="railway")


OPENROUTER_API_URL = get_llm_base_url()
DEFAULT_MODEL = get_llm_model("RAILWAY_MODEL")
DEFAULT_MAX_STEPS = get_int_env("RAILWAY_MAX_STEPS", 8) or 8
DEFAULT_RAILWAY_RETRY_ATTEMPTS = get_int_env("RAILWAY_RETRY_ATTEMPTS", 3) or 3
DEFAULT_503_RETRY_DELAY_SECONDS = get_int_env("RAILWAY_RETRY_DELAY_503_SECONDS", 30) or 30
RAILWAY_API_TIMEOUT_SECONDS = get_int_env("RAILWAY_API_TIMEOUT_SECONDS", 60) or 60
OPENROUTER_TIMEOUT_SECONDS = get_int_env("OPENROUTER_TIMEOUT_SECONDS", 120) or 120
RAILWAY_TASK_NAME = "railway"
RETRYABLE_STATUS_CODES = {429, 503}
ROUTE_PATTERN = re.compile(r"^[a-z]-[0-9]{1,2}$", re.IGNORECASE)
STATUS_VALUES = {"RTOPEN", "RTCLOSE"}
SYSTEM_PROMPT = load_prompt_text(__file__, "system_prompt.txt")


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
        help=f"Endpoint API tras. Domyslnie: {AG3NTS_RAILWAY_URL}",
    )
    parser.add_argument(
        "--railway-api-key",
        help="Klucz API tras. Domyslnie: COURSE_API_KEY z lokalnej konfiguracji repo.",
    )
    parser.add_argument(
        "--openrouter-api-key",
        help="Klucz bramy LLM. Mozna tez ustawic LLM_API_KEY.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model OpenRouter. Domyslnie: model skonfigurowany w repozytoryjnym .env.",
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
    def read_value(cli_value: str | None, fallback: str) -> str:
        return (cli_value or fallback).strip()

    railway_api_url = (args.railway_api_url or AG3NTS_RAILWAY_URL).strip()
    railway_api_key = read_value(args.railway_api_key, get_course_api_key())
    openrouter_api_key = read_value(args.openrouter_api_key, get_llm_api_key())
    model = (args.model or DEFAULT_MODEL).strip()
    site_url = (args.site_url or get_optional_env("OPENROUTER_SITE_URL") or "").strip() or None
    site_name = build_task_site_name(__file__, task_name=RAILWAY_TASK_NAME)

    missing: list[str] = []
    if not railway_api_key:
        missing.append("COURSE_API_KEY")
    if not openrouter_api_key:
        missing.append("LLM_API_KEY")

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


def extract_route_argument(arguments: dict[str, Any]) -> str:
    for key in ("route", "line", "track", "target", "name", "id"):
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return value
    for value in arguments.values():
        if isinstance(value, str) and ROUTE_PATTERN.fullmatch(value.strip()):
            return value
    raise RailwayError("Brak parametru trasy w wywolaniu narzedzia.")


def extract_status_argument(arguments: dict[str, Any]) -> str:
    for key in ("value", "status", "state"):
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise RailwayError("Brak parametru statusu w wywolaniu narzedzia.")


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

        retries_left = self.retry_attempts

        while True:
            try:
                result = post_json(
                    self.api_url,
                    payload,
                    headers={"Content-Type": "application/json"},
                    timeout_seconds=RAILWAY_API_TIMEOUT_SECONDS,
                )
                break
            except HttpRequestError as exc:
                if exc.status_code in RETRYABLE_STATUS_CODES and retries_left > 0:
                    retry_delay = self._extract_retry_delay(exc)
                    logger.warning(
                        "Railway API zwrocilo {} dla akcji {}. Czekam {} s i ponawiam "
                        "probe (pozostalo prob: {}).",
                        exc.status_code,
                        action,
                        retry_delay,
                        retries_left,
                    )
                    time.sleep(retry_delay)
                    retries_left -= 1
                    continue
                raise RailwayError(str(exc)) from exc

        if not isinstance(result, dict):
            raise RailwayError(
                f"Railway API zwrocilo niepoprawny typ danych: {type(result)!r}"
            )

        if result.get("ok") is False:
            message = (
                result.get("error")
                or result.get("message")
                or json.dumps(result, ensure_ascii=False)
            )
            raise RailwayError(f"Railway API zwrocilo blad: {message}")

        return result

    def _extract_retry_delay(self, request_error: HttpRequestError) -> int:
        retry_after_header = (request_error.headers or {}).get("Retry-After")
        if retry_after_header:
            try:
                return max(1, int(retry_after_header))
            except ValueError:
                logger.debug(
                    "Nie udalo sie sparsowac naglowka Retry-After: {}",
                    retry_after_header,
                )

        try:
            payload = json.loads(request_error.body or "")
        except json.JSONDecodeError:
            payload = None

        if isinstance(payload, dict):
            retry_after = payload.get("retry_after")
            if isinstance(retry_after, int) and retry_after > 0:
                return retry_after
            penalty_seconds = payload.get("penalty_seconds")
            if isinstance(penalty_seconds, int) and penalty_seconds > 0:
                return penalty_seconds

        if request_error.status_code == 503:
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
        "reconfigure": lambda args: client.call(action="reconfigure", route=extract_route_argument(args)),
        "getstatus": lambda args: client.call(action="getstatus", route=extract_route_argument(args)),
        "setstatus": lambda args: client.call(
            action="setstatus",
            route=extract_route_argument(args),
            value=extract_status_argument(args),
        ),
        "save": lambda args: client.call(action="save", route=extract_route_argument(args)),
    }


def execute_tool_call(
    tool_call: ToolCall,
    handlers: dict[str, Any],
    *,
    show_tool_results: bool,
) -> dict[str, Any]:
    if tool_call.name not in handlers:
        raise RailwayError(f"Model wywolal nieznane narzedzie: {tool_call.name!r}")

    result = handlers[tool_call.name](tool_call.arguments)
    if show_tool_results:
        logger.debug(
            "Tool {} wywolany z argumentami: {}",
            tool_call.name,
            json.dumps(tool_call.arguments, ensure_ascii=False),
        )
        logger.debug(
            "Wynik toola {}: {}",
            tool_call.name,
            json.dumps(result, ensure_ascii=False, indent=2),
        )

    return {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "name": tool_call.name,
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
    openrouter_client = build_task_openrouter_client(
        __file__,
        api_key=config.openrouter_api_key,
        base_url=OPENROUTER_API_URL,
        model=config.model,
        task_name=RAILWAY_TASK_NAME,
        timeout_seconds=OPENROUTER_TIMEOUT_SECONDS,
        site_url=config.site_url,
        site_name=config.site_name,
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    for _ in range(config.max_steps):
        try:
            completion = openrouter_client.create_completion(messages, tools=TOOLS)
        except OpenRouterError as exc:
            raise RailwayError(str(exc)) from exc

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
                tool_message = execute_tool_call(
                    tool_call,
                    handlers,
                    show_tool_results=show_tool_results,
                )
                messages.append(tool_message)
            continue

        if completion.content:
            return completion.content

        raise RailwayError("Model nie zwrocil ani odpowiedzi tekstowej, ani tool calli.")

    raise RailwayError(
        f"Tool calling nie zakonczyl sie w limicie {config.max_steps} iteracji."
    )


def main() -> int:
    args = parse_args()
    configure_logging(name="railway", verbose=args.show_tool_results)
    if not OPENROUTER_API_URL:
        logger.error("Brakuje LLM_BASE_URL w lokalnej konfiguracji repo.")
        return 1
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
