"""Analyze local attachments with OpenRouter and save structured reports."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from repo_env import get_env, get_optional_env, load_repo_env


load_repo_env(__file__)


try:
    from loguru import logger
except ImportError as error:
    raise SystemExit(
        "Brak zaleznosci 'loguru'. Zainstaluj ja poleceniem: pip install loguru"
    ) from error


OPENROUTER_URL = get_env("OPENROUTER_BASE_URL")
DEFAULT_MODEL = "openai/gpt-4.1-mini"
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
        default=24_000,
        help="Maksymalna liczba znakow czytanych z jednego pliku tekstowego.",
    )
    parser.add_argument(
        "--site-url",
        default=get_optional_env("OPENROUTER_SITE_URL"),
        help="Opcjonalny naglowek HTTP-Referer dla OpenRouter.",
    )
    parser.add_argument(
        "--site-name",
        default=get_env("OPENROUTER_SITE_NAME", "sendit-local-analyzer"),
        help="Opcjonalny naglowek X-Title dla OpenRouter.",
    )
    return parser.parse_args()


def resolve_dir(path: Path, base_dir: Path) -> Path:
    return path if path.is_absolute() else (base_dir / path)


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
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = path.read_text(encoding="utf-8-sig")
    return content[:max_chars]


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


def call_openrouter(
    *,
    api_key: str,
    model: str,
    messages: list[dict[str, object]],
    site_url: str,
    site_name: str,
) -> dict[str, object]:
    payload = {
        "model": model,
        "messages": messages,
    }
    request = Request(
        OPENROUTER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": site_url,
            "X-Title": site_name,
        },
        method="POST",
    )

    with urlopen(request, timeout=120) as response:
        response_body = response.read().decode("utf-8")
    return json.loads(response_body)


def extract_text_response(response_json: dict[str, object]) -> str:
    choices = response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Brak pola 'choices' w odpowiedzi OpenRouter.")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise ValueError("Niepoprawny format elementu 'choices'.")

    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("Brak pola 'message' w odpowiedzi OpenRouter.")

    content = message.get("content")
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        return "\n".join(part for part in text_parts if part)

    raise ValueError("Nie udalo sie odczytac tresci odpowiedzi modelu.")


def write_text_report(output_path: Path, response_text: str) -> None:
    output_path.write_text(response_text, encoding="utf-8")


def write_json_report(output_path: Path, response_json: dict[str, object]) -> None:
    output_path.write_text(
        json.dumps(response_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def output_path_for(target: AnalysisTarget, output_dir: Path, output_format: str) -> Path:
    extension = ".json" if output_format == "json" else ".analysis.md"
    return output_dir / f"{target.path.stem}{extension}"


def main() -> int:
    if not OPENROUTER_URL:
        logger.error("Brak OPENROUTER_BASE_URL w .env.")
        return 1
    args = parse_args()
    base_dir = Path.cwd()
    input_dir = resolve_dir(args.input_dir, base_dir)
    output_dir = resolve_dir(args.output_dir, base_dir)
    project_root = base_dir.parent

    if not input_dir.exists():
        logger.error("Brak katalogu wejsciowego: {}", input_dir)
        return 1

    api_key = (os.environ.get("OPENROUTER_API_KEY") or "").strip() or None

    if not api_key:
        logger.error(
            "Brak klucza OpenRouter. Ustaw OPENROUTER_API_KEY w {} "
            "albo zmienna srodowiskowa OPENROUTER_API_KEY.",
            project_root / ".env",
        )
        return 1

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
            messages = build_messages(target, args.question, args.max_text_chars)
            response_json = call_openrouter(
                api_key=api_key,
                model=args.model,
                messages=messages,
                site_url=args.site_url,
                site_name=args.site_name,
            )
            destination = output_path_for(target, output_dir, args.format)
            if args.format == "json":
                write_json_report(destination, response_json)
            else:
                write_text_report(destination, extract_text_response(response_json))
            logger.success("Zapisano wynik do {}", destination)
        except (HTTPError, URLError, TimeoutError, ValueError, OSError) as error:
            failures += 1
            logger.error("Nie udalo sie przeanalizowac {}: {}", target.path.name, error)

    if failures:
        logger.warning("Zakonczono z {} bledami.", failures)
        return 1

    logger.success("Analiza zakonczona powodzeniem.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
