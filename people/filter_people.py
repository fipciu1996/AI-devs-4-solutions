"""Filter people records, classify jobs, and build a verify payload."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
REPO_ROOT_HINT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_HINT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_HINT))

from devs_utilities.ag3nts import AG3NTS_VERIFY_URL, submit_task_answer
from devs_utilities.bootstrap import bootstrap_repo
from devs_utilities.files import resolve_path, write_json
from devs_utilities.logging import configure_logging, logger as shared_logger
from devs_utilities.openrouter import (
    build_task_openrouter_client,
    build_task_site_name,
    OpenRouterClient,
    OpenRouterError,
    ToolCall,
    strip_code_fences,
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
logger = shared_logger.bind(component="people.filter")


OPENROUTER_API_URL = get_llm_base_url()
TASK_NAME = "people"
OPENROUTER_TASK_NAME = "people-filter"
REFERENCE_DATE = date.fromisoformat(get_env("PEOPLE_REFERENCE_DATE", "2026-03-10"))
DEFAULT_OPENROUTER_MODEL = get_llm_model("PEOPLE_MODEL")
DEFAULT_BATCH_SIZE = get_int_env("PEOPLE_BATCH_SIZE", 25) or 25
API_TIMEOUT_SECONDS = get_int_env("PEOPLE_TIMEOUT_SECONDS", 120) or 120
OPENROUTER_TIMEOUT_SECONDS = get_int_env("OPENROUTER_TIMEOUT_SECONDS", 120) or 120
ALLOWED_TAGS = [
    "IT",
    "transport",
    "edukacja",
    "medycyna",
    "praca z ludźmi",
    "praca z pojazdami",
    "praca fizyczna",
]
TAG_DESCRIPTIONS = {
    "IT": "Prace związane z oprogramowaniem, danymi, sieciami, systemami lub sprzętem IT.",
    "transport": "Prace związane z przewozem ludzi lub towarów, logistyką, spedycją, ruchem i dostawami.",
    "edukacja": "Prace związane z nauczaniem, szkoleniami, wychowaniem i przekazywaniem wiedzy.",
    "medycyna": "Prace związane z diagnozą, leczeniem, opieką zdrowotną i rehabilitacją.",
    "praca z ludźmi": "Prace wymagające intensywnej obsługi, pomocy, koordynacji lub bezpośredniej współpracy z ludźmi.",
    "praca z pojazdami": "Prace obejmujące prowadzenie, serwisowanie, diagnostykę lub obsługę pojazdów.",
    "praca fizyczna": "Prace manualne, terenowe, warsztatowe lub wymagające wysiłku fizycznego.",
}
CSV_COLUMNS = {
    "name",
    "surname",
    "gender",
    "birthDate",
    "birthPlace",
    "birthCountry",
    "job",
}
MODEL_MAX_STEPS = 4
FILTER_SYSTEM_PROMPT = load_prompt_text(__file__, "filter_system_prompt.txt")
TRANSPORT_JOB_KEYWORDS = (
    "transport",
    "kierow",
    "kurier",
    "logist",
    "spedy",
    "dyspozytor",
    "przewoz",
    "dostaw",
    "motornicz",
    "maszynist",
    "konduktor",
)


@dataclass(slots=True)
class Person:
    row_id: int
    name: str
    surname: str
    gender: str
    birth_date: date
    birth_place: str
    birth_country: str
    job: str

    @property
    def birth_year(self) -> int:
        return self.birth_date.year

    def age_on(self, current_date: date) -> int:
        years = current_date.year - self.birth_date.year
        birthday_passed = (current_date.month, current_date.day) >= (
            self.birth_date.month,
            self.birth_date.day,
        )
        return years if birthday_passed else years - 1


@dataclass(slots=True)
class AppConfig:
    course_api_key: str
    llm_api_key: str
    openrouter_model: str
    site_url: str | None
    site_name: str | None
    batch_size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rozwiązanie zadania people: filtruje kandydatów z people.csv, "
            "taguje zawody przez LLM i przygotowuje payload do /verify."
        )
    )
    parser.add_argument("--csv", default="people.csv", help="Ścieżka do people.csv.")
    parser.add_argument(
        "--config",
        default="people_config.json",
        help="Ścieżka do pliku JSON z kluczami i ustawieniami.",
    )
    parser.add_argument(
        "--output",
        default="people_result.json",
        help="Plik wynikowy z payloadem do wysłania.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Nadpisuje batch_size z configu.",
    )
    parser.add_argument(
        "--model",
        help="Nadpisuje model z repozytoryjnego .env.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Wyślij wynik na wbudowany endpoint /verify.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Nie wołaj LLM ani verify. Pokaż tylko przefiltrowanych kandydatów.",
    )
    return parser.parse_args()


def load_config(path: Path, args: argparse.Namespace) -> AppConfig:
    payload: dict[str, Any] = {}
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))

    hub_api_key = str(
        payload.get("course_api_key") or get_course_api_key() or ""
    ).strip()
    openrouter_api_key = str(
        payload.get("llm_api_key")
        or get_llm_api_key()
        or ""
    ).strip()
    openrouter_model = DEFAULT_OPENROUTER_MODEL.strip()
    site_url = str(payload.get("site_url") or get_optional_env("OPENROUTER_SITE_URL") or "").strip() or None
    site_name = build_task_site_name(__file__, task_name=OPENROUTER_TASK_NAME)
    batch_size = int(payload.get("batch_size") or DEFAULT_BATCH_SIZE)

    if args.model:
        openrouter_model = args.model
    if args.batch_size:
        batch_size = args.batch_size
    if not openrouter_model:
        raise ValueError("Brakuje PEOPLE_MODEL lub OPENROUTER_MODEL w repozytoryjnym .env.")

    return AppConfig(
        course_api_key=hub_api_key,
        llm_api_key=openrouter_api_key,
        openrouter_model=openrouter_model,
        site_url=site_url,
        site_name=site_name,
        batch_size=batch_size,
    )


def read_people(csv_path: Path) -> list[Person]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = CSV_COLUMNS - set(reader.fieldnames or [])
        if missing:
            missing_names = ", ".join(sorted(missing))
            raise ValueError(f"Brakuje kolumn w CSV: {missing_names}")

        people: list[Person] = []
        for row_id, row in enumerate(reader, start=1):
            people.append(
                Person(
                    row_id=row_id,
                    name=row["name"].strip(),
                    surname=row["surname"].strip(),
                    gender=row["gender"].strip(),
                    birth_date=datetime.strptime(
                        row["birthDate"].strip(), "%Y-%m-%d"
                    ).date(),
                    birth_place=row["birthPlace"].strip(),
                    birth_country=row["birthCountry"].strip(),
                    job=row["job"].strip(),
                )
            )
        return people


def filter_candidates(people: list[Person]) -> list[Person]:
    filtered: list[Person] = []
    for person in people:
        age = person.age_on(REFERENCE_DATE)
        if person.gender != "M":
            continue
        if not 20 <= age <= 40:
            continue
        if person.birth_place.casefold() != "grudziądz".casefold():
            continue
        filtered.append(person)
    return filtered


def chunked(items: list[Person], size: int) -> list[list[Person]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def build_llm_schema() -> dict[str, Any]:
    return {
        "name": "people_tags",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "row_id": {"type": "integer"},
                            "tags": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": ALLOWED_TAGS,
                                },
                            },
                        },
                        "required": ["row_id", "tags"],
                    },
                }
            },
            "required": ["results"],
        },
    }


def build_llm_prompt(batch: list[Person]) -> str:
    descriptions = "\n".join(
        f"- {tag}: {description}" for tag, description in TAG_DESCRIPTIONS.items()
    )
    jobs = "\n".join(
        f"{person.row_id}. {person.name} {person.surname}: {person.job}" for person in batch
    )
    return (
        "Przypisz tagi do opisów stanowisk pracy.\n"
        "Zasady:\n"
        "- Używaj wyłącznie tagów z dozwolonej listy.\n"
        "- Możesz przypisać zero, jeden lub wiele tagów.\n"
        "- Oceniaj wyłącznie na podstawie opisu stanowiska.\n"
        "- Nie zgaduj danych osobowych, płci ani wieku.\n"
        "- Zwróć wynik dla każdego row_id z listy.\n\n"
        "Dozwolone tagi:\n"
        f"{descriptions}\n\n"
        "Rekordy do sklasyfikowania:\n"
        f"{jobs}"
    )


def build_plain_json_prompt(
    batch: list[Person],
    *,
    previous_error: str | None = None,
) -> str:
    prompt = (
        f"{build_llm_prompt(batch)}\n\n"
        'Zwróć wyłącznie poprawny JSON w formacie '
        '{"results":[{"row_id":123,"tags":["transport"]}]}. '
        "Uwzględnij każdy row_id dokładnie raz."
    )
    if previous_error:
        prompt = f"{prompt}\nPoprzednia odpowiedź była niepoprawna: {previous_error}"
    return prompt


PEOPLE_FILTER_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_job_batch_context",
            "description": "Return the current batch of job descriptions and allowed tags.",
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
            "name": "validate_tag_results",
            "description": "Validate that the proposed tags contain all expected row_id values and only allowed tags.",
            "parameters": {
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "row_id": {"type": "integer"},
                                "tags": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["row_id", "tags"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["results"],
                "additionalProperties": False,
            },
        },
    },
]


def _extract_classification_items(parsed: Any) -> list[dict[str, Any]]:
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    if isinstance(parsed, dict):
        for key in ("results", "items", "classifications", "answers"):
            value = parsed.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        if "row_id" in parsed and "tags" in parsed:
            return [parsed]
    raise OpenRouterError("Model classification payload is missing a results list.")


def parse_classification_result(raw_content: str) -> dict[int, list[str]]:
    normalized = strip_code_fences(raw_content).strip()
    try:
        parsed = json.loads(normalized)
    except json.JSONDecodeError as exc:
        raise OpenRouterError("Model classification payload is not valid JSON.") from exc

    items = _extract_classification_items(parsed)
    tags_by_row_id: dict[int, list[str]] = {}
    for item in items:
        row_id = int(item["row_id"])
        tags = [str(tag) for tag in item["tags"]]
        tags_by_row_id[row_id] = tags
    return tags_by_row_id


def build_people_filter_handlers(batch: list[Person]) -> dict[str, Any]:
    expected_ids = {person.row_id for person in batch}
    allowed_tags = set(ALLOWED_TAGS)

    def get_job_batch_context(_: dict[str, Any]) -> dict[str, Any]:
        return {
            "prompt": build_llm_prompt(batch),
            "allowed_tags": ALLOWED_TAGS,
            "row_ids": sorted(expected_ids),
        }

    def validate_tag_results(arguments: dict[str, Any]) -> dict[str, Any]:
        results = arguments.get("results")
        if not isinstance(results, list):
            return {"is_valid": False, "message": "results must be a list"}
        received_ids: set[int] = set()
        invalid_tags: list[str] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            if isinstance(item.get("row_id"), int):
                received_ids.add(int(item["row_id"]))
            tags = item.get("tags")
            if isinstance(tags, list):
                for tag in tags:
                    if isinstance(tag, str) and tag not in allowed_tags:
                        invalid_tags.append(tag)
        return {
            "is_valid": received_ids == expected_ids and not invalid_tags,
            "expected_row_ids": sorted(expected_ids),
            "received_row_ids": sorted(received_ids),
            "invalid_tags": sorted(set(invalid_tags)),
        }

    return {
        "get_job_batch_context": get_job_batch_context,
        "validate_tag_results": validate_tag_results,
    }


def execute_people_filter_tool_call(
    tool_call: ToolCall,
    handlers: dict[str, Any],
) -> dict[str, Any]:
    if tool_call.name not in handlers:
        raise OpenRouterError(f"Unknown people.filter tool call: {tool_call.name!r}")
    result = handlers[tool_call.name](tool_call.arguments)
    return {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "name": tool_call.name,
        "content": json.dumps(result, ensure_ascii=False),
    }


def classify_jobs(
    batch: list[Person],
    config: AppConfig,
    openrouter_client: OpenRouterClient,
) -> dict[int, list[str]]:
    if not config.llm_api_key:
        raise ValueError("Brakuje LLM_API_KEY lub llm_api_key w configu.")
    handlers = build_people_filter_handlers(batch)
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": FILTER_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": "Sklasyfikuj bieżącą partię zawodów.",
        },
    ]
    for _ in range(MODEL_MAX_STEPS):
        completion = openrouter_client.create_completion(messages, tools=PEOPLE_FILTER_TOOLS)
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
                messages.append(execute_people_filter_tool_call(tool_call, handlers))
            continue
        raw_text = str(completion.content or "").strip()
        if not raw_text:
            raise RuntimeError("Brak treści w odpowiedzi modelu.")
        messages.append(assistant_message)
        try:
            return parse_classification_result(raw_text)
        except (OpenRouterError, KeyError, TypeError, ValueError) as exc:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Return only valid JSON with this exact shape: "
                        '{"results":[{"row_id":123,"tags":["transport"]}]}. '
                        f"Your previous response was invalid: {exc}"
                    ),
                }
            )
            continue
    logger.warning(
        "OpenRouter tool calling did not finish for people.filter. Falling back to plain JSON mode."
    )
    return classify_jobs_without_tools(batch, openrouter_client)


def classify_jobs_without_tools(
    batch: list[Person],
    openrouter_client: OpenRouterClient,
) -> dict[int, list[str]]:
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": FILTER_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": build_plain_json_prompt(batch),
        },
    ]
    last_error: str | None = None
    for _ in range(2):
        completion = openrouter_client.create_completion(messages)
        raw_text = str(completion.content or "").strip()
        if not raw_text:
            last_error = "Model classification payload was empty."
        else:
            try:
                return parse_classification_result(raw_text)
            except (OpenRouterError, KeyError, TypeError, ValueError) as exc:
                last_error = str(exc)
        messages.append({"role": "assistant", "content": raw_text})
        messages.append(
            {
                "role": "user",
                "content": build_plain_json_prompt(batch, previous_error=last_error),
            }
        )
    raise OpenRouterError(
        "OpenRouter plain JSON fallback did not produce a valid people.filter response."
    )


def infer_tags_from_job_title(job: str) -> list[str]:
    normalized = unicodedata.normalize("NFKD", job).casefold()
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    if any(keyword in normalized for keyword in TRANSPORT_JOB_KEYWORDS):
        return ["transport"]
    return []


def classify_batch_with_retries(
    batch: list[Person],
    config: AppConfig,
    openrouter_client: OpenRouterClient,
) -> dict[int, list[str]]:
    try:
        tags_by_row_id = classify_jobs(batch, config, openrouter_client)
    except OpenRouterError as exc:
        if len(batch) == 1:
            person = batch[0]
            inferred_tags = infer_tags_from_job_title(person.job)
            logger.warning(
                "Model nie sklasyfikował row_id {} po fallbackach. Używam lokalnego fallbacku na podstawie zawodu: {} -> {} ({})",
                person.row_id,
                person.job,
                inferred_tags,
                exc,
            )
            return {person.row_id: inferred_tags}

        logger.warning(
            "Model nie sklasyfikował partii row_id {}. Dzielę partię na mniejsze po błędzie: {}",
            ", ".join(str(person.row_id) for person in batch),
            exc,
        )
        tags_by_row_id = {}
        next_batch_size = max(1, len(batch) // 2)
        for sub_batch in chunked(batch, next_batch_size):
            tags_by_row_id.update(classify_batch_with_retries(sub_batch, config, openrouter_client))
        return tags_by_row_id

    missing_ids = {person.row_id for person in batch} - set(tags_by_row_id)
    if not missing_ids:
        return tags_by_row_id

    missing_display = ", ".join(str(item) for item in sorted(missing_ids))
    if len(batch) == 1:
        person = batch[0]
        inferred_tags = infer_tags_from_job_title(person.job)
        logger.warning(
            "Model pominął row_id {}. Używam lokalnego fallbacku na podstawie zawodu: {} -> {}",
            person.row_id,
            person.job,
            inferred_tags,
        )
        tags_by_row_id[person.row_id] = inferred_tags
        return tags_by_row_id

    logger.warning(
        "Model nie zwrócił tagów dla row_id: {}. Ponawiam brakujące rekordy w mniejszych partiach.",
        missing_display,
    )
    missing_people = [person for person in batch if person.row_id in missing_ids]
    next_batch_size = max(1, len(missing_people) // 2)
    for sub_batch in chunked(missing_people, next_batch_size):
        tags_by_row_id.update(classify_batch_with_retries(sub_batch, config, openrouter_client))
    return tags_by_row_id


def classify_all(
    candidates: list[Person],
    config: AppConfig,
    openrouter_client: OpenRouterClient,
) -> dict[int, list[str]]:
    tags_by_row_id: dict[int, list[str]] = {}
    for batch in chunked(candidates, config.batch_size):
        tags_by_row_id.update(classify_batch_with_retries(batch, config, openrouter_client))
    missing_ids = {person.row_id for person in candidates} - set(tags_by_row_id)
    if missing_ids:
        missing_display = ", ".join(str(item) for item in sorted(missing_ids))
        raise RuntimeError(f"Model nie zwrócił tagów dla row_id: {missing_display}")
    return tags_by_row_id


def build_answer(candidates: list[Person], tags_by_row_id: dict[int, list[str]]) -> list[dict[str, Any]]:
    answer: list[dict[str, Any]] = []
    for person in candidates:
        tags = tags_by_row_id[person.row_id]
        if "transport" not in tags:
            continue
        answer_tags = ["transport"]
        answer.append(
            {
                "name": person.name,
                "surname": person.surname,
                "gender": person.gender,
                "born": person.birth_year,
                "city": person.birth_place,
                "tags": answer_tags,
            }
        )
    return answer


def build_verify_payload(answer: list[dict[str, Any]], hub_api_key: str) -> dict[str, Any]:
    if not hub_api_key:
        raise ValueError("Brakuje COURSE_API_KEY lub course_api_key w configu.")
    return {
        "apikey": hub_api_key,
        "task": TASK_NAME,
        "answer": answer,
    }


def main() -> None:
    configure_logging(name="people.filter")
    if not OPENROUTER_API_URL:
        raise ValueError("Missing LLM_BASE_URL in the local repository config.")
    args = parse_args()
    config_path = resolve_path(args.config, REPO_ROOT / "people")
    csv_path = resolve_path(args.csv, REPO_ROOT / "people")
    config = load_config(config_path, args)
    people = read_people(csv_path)
    candidates = filter_candidates(people)
    openrouter_client = build_task_openrouter_client(
        __file__,
        api_key=config.llm_api_key,
        base_url=OPENROUTER_API_URL,
        model=config.openrouter_model,
        task_name=TASK_NAME,
        timeout_seconds=OPENROUTER_TIMEOUT_SECONDS,
        site_url=config.site_url,
        site_name=config.site_name,
    )

    if args.dry_run:
        preview = {
            "today": REFERENCE_DATE.isoformat(),
            "candidate_count": len(candidates),
            "candidates": [
                {
                    "row_id": person.row_id,
                    "name": person.name,
                    "surname": person.surname,
                    "age": person.age_on(REFERENCE_DATE),
                    "born": person.birth_year,
                    "city": person.birth_place,
                    "job": person.job,
                }
                for person in candidates
            ],
        }
        logger.info("Preview:\n{}", json.dumps(preview, ensure_ascii=False, indent=2))
        return

    tags_by_row_id = classify_all(candidates, config, openrouter_client)
    answer = build_answer(candidates, tags_by_row_id)
    payload = build_verify_payload(answer, config.course_api_key)
    output_path = resolve_path(args.output, REPO_ROOT / "people")
    write_json(output_path, payload)
    logger.info("Payload:\n{}", json.dumps(payload, ensure_ascii=False, indent=2))

    if args.verify:
        verify_response = submit_task_answer(
            AG3NTS_VERIFY_URL,
            api_key=config.course_api_key,
            task=TASK_NAME,
            answer=answer,
            timeout_seconds=API_TIMEOUT_SECONDS,
        )
        logger.info(
            "Verify response:\n{}",
            json.dumps(verify_response, ensure_ascii=False, indent=2),
        )


if __name__ == "__main__":
    main()
