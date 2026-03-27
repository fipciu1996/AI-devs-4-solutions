"""Analyze local attachments with OpenRouter and save structured reports."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

REPO_ROOT_HINT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_HINT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_HINT))

from devs_utilities.bootstrap import bootstrap_repo
from devs_utilities.files import read_text_with_fallback, resolve_path, write_json
from devs_utilities.logging import configure_logging, logger as shared_logger
from devs_utilities.openrouter import (
    OpenRouterClient,
    OpenRouterConfig,
    OpenRouterError,
    ToolCall,
    extract_completion_result,
)
from repo_env import get_env, get_int_env, get_optional_env


REPO_ROOT = bootstrap_repo(__file__)
logger = shared_logger.bind(component="sendit.analyze")


OPENROUTER_URL = get_env("OPENROUTER_BASE_URL")
DEFAULT_MODEL = get_env("OPENROUTER_MODEL", "openai/gpt-4.1-mini") or "openai/gpt-4.1-mini"
DEFAULT_MAX_TEXT_CHARS = get_int_env("SENDIT_ANALYZE_MAX_TEXT_CHARS", 24_000) or 24_000
OPENROUTER_TIMEOUT_SECONDS = get_int_env("OPENROUTER_TIMEOUT_SECONDS", 120) or 120
MODEL_MAX_STEPS = 3
DEFAULT_SITE_NAME = (
    get_optional_env("OPENROUTER_SITE_NAME")
    or get_optional_env("OPENROUTER_APP_TITLE")
    or "sendit-local-analyzer"
)
TEXT_EXTENSIONS = {".md", ".txt", ".csv", ".json", ".yaml", ".yml"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass(slots=True)
class AnalysisTarget:
    path: Path
    kind: Literal["text", "image"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analizuje pliki lokalne przy pomocy OpenRouter API i zapisuje raporty "
            "tekstowe lub JSON."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("attachments"),
        help="Katalog z plikami do analizy. Domyslnie: attachments",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis"),
        help="Katalog docelowy na wyniki analizy. Domyslnie: analysis",
    )
    parser.add_argument(
        "--question",
        default=(
            "Przeanalizuj ten plik i wypisz tylko sprawdzalne fakty, istotne reguly, "
            "formaty dokumentow, wymagane pola, ograniczenia oraz wszelkie dane "
            "operacyjne, ktore wynikaja bezposrednio z tresci pliku."
        ),
        help="Instrukcja przekazywana modelowi dla kazdego pliku.",
    )
    parser.add_argument(
        "--glob",
        default="*",
        help="Wzorzec plikow w katalogu wejsciowym. Domyslnie: *",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model OpenRouter. Domyslnie: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Format zapisu wynikow. Domyslnie: text",
    )
    parser.add_argument(
        "--max-text-chars",
        type=int,
        default=DEFAULT_MAX_TEXT_CHARS,
        help="Maksymalna liczba znakow czytanych z jednego pliku tekstowego.",
    )
    parser.add_argument(
        "--site-url",
        default=get_optional_env("OPENROUTER_SITE_URL"),
        help="Opcjonalny naglowek HTTP-Referer dla OpenRouter.",
    )
    parser.add_argument(
        "--site-name",
        default=DEFAULT_SITE_NAME,
        help="Opcjonalny naglowek X-Title dla OpenRouter.",
    )
    return parser.parse_args()


def collect_targets(input_dir: Path, pattern: str) -> list[AnalysisTarget]:
    targets: list[AnalysisTarget] = []
    for path in sorted(input_dir.glob(pattern)):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in TEXT_EXTENSIONS:
            targets.append(AnalysisTarget(path=path, kind="text"))
        elif suffix in IMAGE_EXTENSIONS:
            targets.append(AnalysisTarget(path=path, kind="image"))
        else:
            logger.info("Pomijam nieobslugiwany typ pliku: {}", path.name)
    return targets


def load_text(path: Path, max_chars: int) -> str:
    return read_text_with_fallback(path)[:max_chars]


def encode_image_as_data_url(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    effective_mime_type = mime_type or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{effective_mime_type};base64,{encoded}"


def build_messages(
    target: AnalysisTarget,
    question: str,
    max_text_chars: int,
) -> list[dict[str, object]]:
    system_prompt = (
        "Analizujesz pojedynczy plik dokumentacyjny. "
        "Odpowiadaj po polsku. "
        "Wyciagaj tylko informacje wynikajace bezposrednio z pliku. "
        "Jesli czegos nie ma w materiale, napisz to wprost. "
        "Nie proponuj obchodzenia zabezpieczen, falszowania dokumentow ani "
        "innych dzialan niezgodnych z prawem lub regulaminem."
    )

    if target.kind == "text":
        content = load_text(target.path, max_text_chars)
        user_content: list[dict[str, object]] = [
            {
                "type": "text",
                "text": (
                    f"Plik: {target.path.name}\n"
                    f"Instrukcja: {question}\n\n"
                    "Tresci pliku:\n"
                    f"{content}"
                ),
            }
        ]
    else:
        user_content = [
            {
                "type": "text",
                "text": (
                    f"Plik: {target.path.name}\n"
                    f"Instrukcja: {question}\n\n"
                    "Zinterpretuj obraz i wypisz tylko to, co faktycznie widac."
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": encode_image_as_data_url(target.path)},
            },
        ]

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def write_text_report(output_path: Path, response_text: str) -> None:
    output_path.write_text(response_text, encoding="utf-8")


def write_json_report(output_path: Path, response_json: dict[str, object]) -> None:
    write_json(output_path, response_json, trailing_newline=False)


def output_path_for(target: AnalysisTarget, output_dir: Path, output_format: str) -> Path:
    extension = ".json" if output_format == "json" else ".analysis.md"
    return output_dir / f"{target.path.stem}{extension}"


SENDIT_ANALYZE_TOOLS: list[dict[str, object]] = [
    {
        "type": "function",
        "function": {
            "name": "get_analysis_target_context",
            "description": "Return the full message payload for the current attachment analysis target.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    }
]


def build_sendit_analyze_handlers(
    target: AnalysisTarget,
    question: str,
    max_text_chars: int,
) -> dict[str, object]:
    messages = build_messages(target, question, max_text_chars)

    def get_analysis_target_context(_: dict[str, object]) -> dict[str, object]:
        return {"messages": messages, "kind": target.kind, "filename": target.path.name}

    return {"get_analysis_target_context": get_analysis_target_context}


def execute_sendit_analyze_tool_call(
    tool_call: ToolCall,
    handlers: dict[str, object],
) -> dict[str, object]:
    if tool_call.name not in handlers:
        raise OpenRouterError(f"Unknown sendit analyze tool call: {tool_call.name!r}")
    result = handlers[tool_call.name](tool_call.arguments)
    return {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "name": tool_call.name,
        "content": json.dumps(result, ensure_ascii=False),
    }


def analyze_target_with_tool_calling(
    openrouter_client: OpenRouterClient,
    target: AnalysisTarget,
    question: str,
    max_text_chars: int,
) -> dict[str, object]:
    handlers = build_sendit_analyze_handlers(target, question, max_text_chars)
    messages: list[dict[str, object]] = [
        {
            "role": "system",
            "content": (
                "Analizujesz pojedynczy plik dokumentacyjny. "
                "Użyj tool callingu przed finalną odpowiedzią."
            ),
        },
        {
            "role": "user",
            "content": f"Przeanalizuj plik {target.path.name}.",
        },
    ]
    for _ in range(MODEL_MAX_STEPS):
        response_json = openrouter_client.create_completion(
            messages,
            tools=SENDIT_ANALYZE_TOOLS,
        )
        assistant_message: dict[str, object] = {
            "role": "assistant",
            "content": response_json.content or "",
        }
        if response_json.tool_calls:
            assistant_message["tool_calls"] = [
                tool_call.to_message_dict() for tool_call in response_json.tool_calls
            ]
            messages.append(assistant_message)
            for tool_call in response_json.tool_calls:
                messages.append(execute_sendit_analyze_tool_call(tool_call, handlers))
            continue
        return {
            "content": response_json.content or "",
            "raw": {
                "content": response_json.content,
                "tool_calls": [tool_call.to_message_dict() for tool_call in response_json.tool_calls],
            },
        }
    raise OpenRouterError("OpenRouter tool calling did not finish for sendit analysis.")


def main() -> int:
    configure_logging(name="sendit.analyze")
    if not OPENROUTER_URL:
        logger.error("Brak OPENROUTER_BASE_URL w .env.")
        return 1
    args = parse_args()
    base_dir = Path.cwd()
    input_dir = resolve_path(args.input_dir, base_dir)
    output_dir = resolve_path(args.output_dir, base_dir)
    project_root = REPO_ROOT

    if not input_dir.exists():
        logger.error("Brak katalogu wejsciowego: {}", input_dir)
        return 1

    api_key = get_env("OPENROUTER_API_KEY") or None

    if not api_key:
        logger.error(
            "Brak klucza OpenRouter. Ustaw OPENROUTER_API_KEY w {} "
            "albo zmienna srodowiskowa OPENROUTER_API_KEY.",
            project_root / ".env",
        )
        return 1

    openrouter_client = OpenRouterClient(
        OpenRouterConfig(
            api_key=api_key,
            base_url=OPENROUTER_URL,
            model=args.model,
            timeout_seconds=OPENROUTER_TIMEOUT_SECONDS,
            site_url=args.site_url,
            site_name=args.site_name,
        )
    )

    targets = collect_targets(input_dir, args.glob)
    if not targets:
        logger.warning("Nie znaleziono plikow do analizy w {}.", input_dir)
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Znaleziono {} plikow do analizy.", len(targets))

    failures = 0
    for target in targets:
        logger.info("Analizuje: {}", target.path.name)
        try:
            response_json = analyze_target_with_tool_calling(
                openrouter_client,
                target,
                args.question,
                args.max_text_chars,
            )
            destination = output_path_for(target, output_dir, args.format)
            if args.format == "json":
                write_json_report(destination, response_json)
            else:
                content = str(response_json.get("content") or "").strip()
                if not content:
                    raise ValueError("Nie udalo sie odczytac tresci odpowiedzi modelu.")
                write_text_report(destination, content)
            logger.success("Zapisano wynik do {}", destination)
        except (RuntimeError, ValueError, OSError) as error:
            failures += 1
            logger.error("Nie udalo sie przeanalizowac {}: {}", target.path.name, error)

    if failures:
        logger.warning("Zakonczono z {} bledami.", failures)
        return 1

    logger.success("Analiza zakonczona powodzeniem.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
