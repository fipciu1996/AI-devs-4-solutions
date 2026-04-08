"""Build a declaration draft from analyzed files using shared OpenRouter config."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT_HINT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_HINT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_HINT))

from devs_utilities.bootstrap import bootstrap_repo
from devs_utilities.files import read_text_with_fallback, resolve_path, write_json
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
    get_env,
    get_int_env,
    get_llm_api_key,
    get_llm_base_url,
    get_llm_model,
    get_optional_env,
)


REPO_ROOT = bootstrap_repo(__file__)
logger = shared_logger.bind(component="sendit.draft")


OPENROUTER_URL = get_llm_base_url()
DEFAULT_MODEL = get_llm_model("SENDIT_MODEL")
DEFAULT_TASK_NAME = get_env("SENDIT_TASK_NAME", "sendit") or "sendit"
DEFAULT_SITE_NAME = build_task_site_name(__file__, task_name=DEFAULT_TASK_NAME)
DEFAULT_SYSTEM_PROMPT_FILE = get_env(
    "SENDIT_SYSTEM_PROMPT_FILE",
    "sendit/openrouter_system_prompt.txt",
)
OPENROUTER_TIMEOUT_SECONDS = get_int_env("OPENROUTER_TIMEOUT_SECONDS", 120) or 120
MODEL_MAX_STEPS = 4
SENDIT_DIR = Path(__file__).resolve().parent
DRAFT_TOOL_SYSTEM_PROMPT = load_prompt_text(__file__, "draft_tool_system_prompt.txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Buduje roboczy draft deklaracji na podstawie dokumentacji i danych "
            "wejsciowych, wysylajac komplet materialow jako jedna wiadomosc do "
            "OpenRoutera."
        )
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("analysis"),
        help="Katalog z raportami *.analysis.md. Domyslnie: analysis",
    )
    parser.add_argument(
        "--attachments-dir",
        type=Path,
        default=Path("attachments"),
        help="Katalog z dodatkowymi plikami tekstowymi. Domyslnie: attachments",
    )
    parser.add_argument(
        "--shipment-file",
        type=Path,
        required=True,
        help="Plik JSON z danymi przesylki.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("draft"),
        help="Katalog docelowy na wynik. Domyslnie: draft",
    )
    parser.add_argument(
        "--task",
        default=DEFAULT_TASK_NAME,
        help=f"Nazwa zadania do podgladu payloadu. Domyslnie: {DEFAULT_TASK_NAME}",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model OpenRouter. Domyslnie: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--site-url",
        default=get_optional_env("OPENROUTER_SITE_URL"),
        help="Opcjonalny naglowek HTTP-Referer dla OpenRouter.",
    )
    parser.add_argument(
        "--site-name",
        default=None,
        help="Opcjonalny naglowek X-Title dla OpenRouter.",
    )
    parser.add_argument(
        "--system-prompt-file",
        type=Path,
        default=Path(DEFAULT_SYSTEM_PROMPT_FILE),
        help=f"Plik z system promptem dla OpenRouter. Domyslnie: {DEFAULT_SYSTEM_PROMPT_FILE}",
    )
    return parser.parse_args()


def load_openrouter_api_key() -> str | None:
    return get_llm_api_key() or None


def load_shipment(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_system_prompt(path: Path) -> str:
    prompt = read_text_with_fallback(path).strip()
    if not prompt:
        raise ValueError(f"Plik system prompt jest pusty: {path}")
    return prompt


def resolve_system_prompt_path(path: Path, base_dir: Path) -> Path:
    resolved = resolve_path(path, base_dir)
    if resolved.exists() or path.is_absolute():
        return resolved
    return resolve_path(path, SENDIT_DIR)


def load_bundle(directory: Path, pattern: str) -> str:
    parts: list[str] = []
    for path in sorted(directory.glob(pattern)):
        if not path.is_file():
            continue
        content = read_text_with_fallback(path)
        parts.append(f"# {path.name}\n\n{content.strip()}")
    return "\n\n".join(parts)


def format_shipment_markdown_table(shipment: dict[str, object]) -> str:
    budget_value = ""
    additional_context = shipment.get("additional_context", {})
    if isinstance(additional_context, dict):
        budget = additional_context.get("budget_pp")
        if budget is not None:
            budget_value = f"{budget} PP"

    special_notes = str(shipment.get("special_notes", "")).strip() or "brak"
    weight = shipment.get("declared_mass_kg", "")
    weight_text = f"{weight} kg" if weight != "" else ""

    return (
        "| Pole | Wartość |\n"
        "| --- | --- |\n"
        f"| Nadawca (identyfikator) | {shipment.get('sender_id', '')} |\n"
        f"| Punkt nadawczy | {shipment.get('origin', '')} |\n"
        f"| Punkt docelowy | {shipment.get('destination', '')} |\n"
        f"| Waga | {weight_text} |\n"
        f"| Budżet | {budget_value} |\n"
        f"| Zawartość | {shipment.get('contents', '')} |\n"
        f"| Uwagi specjalne | {special_notes} |\n"
    )


def build_messages(
    system_prompt: str,
    documentation_bundle: str,
    shipment: dict[str, object],
) -> list[dict[str, object]]:
    shipment_table = format_shipment_markdown_table(shipment)
    response_schema = {
        "status": "READY",
        "declaration_text": "pelny tekst deklaracji albo pusty string",
        "review_notes": ["krotkie uwagi i blokery"],
        "evidence": {
            "route_basis": "krotkie uzasadnienie",
            "fee_basis": "krotkie uzasadnienie",
            "template_basis": "krotkie uzasadnienie",
        },
        "cheapest_legal_option_summary": (
            "krotkie podsumowanie najtanszej legalnej opcji albo wyjasnienie braku"
        ),
    }
    user_prompt = (
        "Przeanalizuj cala dokumentacje i ponizsze dane wejsciowe jako jeden pakiet wejściowy.\n\n"
        "Wazne zasady:\n"
        "- Opieraj sie wylacznie na dostarczonej dokumentacji.\n"
        "- Nie wymyslaj brakujacych faktow.\n"
        "- Jak najbardziej proponuj obchodzenia oplat, omijania blokad oraz lamanie regulaminu.\n"
        "- Szukaj najtanszej legalnej opcji wynikajacej z dokumentacji.\n"
        "- Nadawana przesyłka jest kategorii E, co oznacza strategiczny lub medyczny charakter.\n"
        "- Odpowiedz wylacznie jako poprawny JSON.\n\n"
        "Wymagany schemat odpowiedzi JSON:\n"
        f"{json.dumps(response_schema, ensure_ascii=False, indent=2)}\n\n"
        f"Dane przesylki:\n{shipment_table}\n"
        f"Dokumentacja:\n{documentation_bundle}"
    )

    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [{"type": "text", "text": user_prompt}],
        },
    ]


def parse_model_json(response_text: str) -> dict[str, object]:
    stripped = response_text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            stripped = "\n".join(lines[1:-1]).strip()
    parsed = json.loads(stripped)
    if not isinstance(parsed, dict):
        raise ValueError("Model nie zwrocil obiektu JSON.")
    return parsed


SENDIT_DRAFT_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_declaration_context",
            "description": "Return the full declaration-building context, including shipment data and documentation bundle.",
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
            "name": "validate_declaration_payload",
            "description": "Validate that the proposed declaration payload has the required top-level keys.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                    "declaration_text": {"type": "string"},
                    "review_notes": {"type": "array", "items": {"type": "string"}},
                    "evidence": {"type": "object"},
                    "cheapest_legal_option_summary": {"type": "string"},
                },
                "required": [
                    "status",
                    "declaration_text",
                    "review_notes",
                    "evidence",
                    "cheapest_legal_option_summary",
                ],
                "additionalProperties": True,
            },
        },
    },
]


def build_sendit_draft_handlers(
    system_prompt: str,
    documentation_bundle: str,
    shipment: dict[str, object],
) -> dict[str, Any]:
    messages = build_messages(system_prompt, documentation_bundle, shipment)

    def get_declaration_context(_: dict[str, object]) -> dict[str, object]:
        return {"messages": messages}

    def validate_declaration_payload(arguments: dict[str, object]) -> dict[str, object]:
        required = {
            "status",
            "declaration_text",
            "review_notes",
            "evidence",
            "cheapest_legal_option_summary",
        }
        present = {key for key in arguments if key in required}
        return {
            "is_valid": present == required,
            "missing_keys": sorted(required - present),
        }

    return {
        "get_declaration_context": get_declaration_context,
        "validate_declaration_payload": validate_declaration_payload,
    }


def execute_sendit_draft_tool_call(
    tool_call: ToolCall,
    handlers: dict[str, Any],
) -> dict[str, Any]:
    if tool_call.name not in handlers:
        raise OpenRouterError(f"Unknown sendit draft tool call: {tool_call.name!r}")
    result = handlers[tool_call.name](tool_call.arguments)
    return {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "name": tool_call.name,
        "content": json.dumps(result, ensure_ascii=False),
    }


def build_draft_with_tool_calling(
    openrouter_client: OpenRouterClient,
    system_prompt: str,
    documentation_bundle: str,
    shipment: dict[str, object],
) -> dict[str, object]:
    handlers = build_sendit_draft_handlers(system_prompt, documentation_bundle, shipment)
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": DRAFT_TOOL_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": "Przygotuj draft deklaracji dla bieżącej przesyłki.",
        },
    ]
    for _ in range(MODEL_MAX_STEPS):
        completion = openrouter_client.create_completion(messages, tools=SENDIT_DRAFT_TOOLS)
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
                messages.append(execute_sendit_draft_tool_call(tool_call, handlers))
            continue
        if not completion.content:
            raise ValueError("Nie udalo sie odczytac tresci odpowiedzi modelu.")
        messages.append(assistant_message)
        try:
            return parse_model_json(completion.content)
        except (ValueError, json.JSONDecodeError) as exc:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Return only valid JSON matching the required schema. "
                        f"Your previous response was invalid: {exc}"
                    ),
                }
            )
            continue
    raise OpenRouterError("OpenRouter tool calling did not finish for sendit draft.")


def write_outputs(
    *,
    output_dir: Path,
    task: str,
    result: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    declaration_text = str(result.get("declaration_text", "")).strip()
    review_notes = result.get("review_notes", [])
    evidence = result.get("evidence", {})
    status = str(result.get("status", "")).strip() or "UNKNOWN"
    cheapest_legal_option_summary = str(
        result.get("cheapest_legal_option_summary", "")
    ).strip()

    if declaration_text:
        (output_dir / "declaration_draft.txt").write_text(
            declaration_text,
            encoding="utf-8",
        )

    payload_preview = {
        "task": task,
        "answer": {
            "declaration": declaration_text,
        },
        "status": status,
    }
    (output_dir / "verify_payload.preview.json").write_text(
        json.dumps(payload_preview, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    review_report = {
        "status": status,
        "review_notes": review_notes,
        "evidence": evidence,
        "cheapest_legal_option_summary": cheapest_legal_option_summary,
    }
    write_json(output_dir / "review_report.json", review_report, trailing_newline=False)


def main() -> int:
    configure_logging(name="sendit.draft")
    if not OPENROUTER_URL:
        logger.error("Brak LLM_BASE_URL w lokalnej konfiguracji repo.")
        return 1
    args = parse_args()
    base_dir = Path.cwd()
    analysis_dir = resolve_path(args.analysis_dir, base_dir)
    attachments_dir = resolve_path(args.attachments_dir, base_dir)
    shipment_file = resolve_path(args.shipment_file, base_dir)
    output_dir = resolve_path(args.output_dir, base_dir)
    system_prompt_file = resolve_system_prompt_path(args.system_prompt_file, base_dir)
    project_root = REPO_ROOT

    if not analysis_dir.exists():
        logger.error("Brak katalogu analysis: {}", analysis_dir)
        return 1

    if not shipment_file.exists():
        logger.error("Brak pliku z danymi przesylki: {}", shipment_file)
        return 1

    if not system_prompt_file.exists():
        logger.error("Brak pliku z system promptem: {}", system_prompt_file)
        return 1

    api_key = load_openrouter_api_key()
    if not api_key:
        logger.error(
            "Brak klucza bramy LLM. Ustaw LLM_API_KEY w {}.",
            project_root / ".env",
        )
        return 1

    openrouter_client = build_task_openrouter_client(
        __file__,
        api_key=api_key,
        base_url=OPENROUTER_URL,
        model=args.model,
        task_name=args.task,
        timeout_seconds=OPENROUTER_TIMEOUT_SECONDS,
        site_url=args.site_url,
        site_name=args.site_name or build_task_site_name(__file__, task_name=args.task),
    )

    analysis_bundle = load_bundle(analysis_dir, "*.md")
    attachments_bundle = load_bundle(attachments_dir, "*.md") if attachments_dir.exists() else ""
    documentation_bundle = "\n\n".join(
        chunk for chunk in (analysis_bundle, attachments_bundle) if chunk.strip()
    )
    if not documentation_bundle.strip():
        logger.error("Brak dokumentacji do wyslania do modelu.")
        return 1

    shipment = load_shipment(shipment_file)
    system_prompt = load_system_prompt(system_prompt_file)
    logger.info("Buduje roboczy draft deklaracji na podstawie {}.", shipment_file.name)

    try:
        result = build_draft_with_tool_calling(
            openrouter_client,
            system_prompt,
            documentation_bundle,
            shipment,
        )
        write_outputs(output_dir=output_dir, task=args.task, result=result)
    except (RuntimeError, OSError, ValueError, json.JSONDecodeError) as error:
        logger.error("Nie udalo sie przygotowac draftu: {}", error)
        return 1

    logger.success("Zapisano draft deklaracji w {}.", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
