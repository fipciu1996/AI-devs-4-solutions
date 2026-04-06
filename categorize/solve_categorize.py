"""Send categorize prompts in a controlled order using shared repo config."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

REPO_ROOT_HINT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_HINT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_HINT))

from devs_utilities.ag3nts import (
    AG3NTS_VERIFY_URL,
    build_ag3nts_task_data_url,
    submit_task_answer,
)
from devs_utilities.bootstrap import bootstrap_repo
from devs_utilities.http import HttpRequestError, get_text
from devs_utilities.logging import configure_logging, logger as shared_logger
from devs_utilities.openrouter import (
    build_task_openrouter_client,
    OpenRouterClient,
    OpenRouterError,
    parse_json_object_content,
)
from repo_env import get_course_api_key, get_env, get_int_env, get_optional_env


REPO_ROOT = bootstrap_repo(__file__)
logger = shared_logger.bind(component="categorize")


TASK_NAME = "categorize"
OUTPUT_DIR = Path(__file__).resolve().parent
PROMPT_PLAN_PATH = OUTPUT_DIR / "last_prompt_plan.json"
DEFAULT_SEND_ORDER = get_env("CATEGORIZE_SEND_ORDER", "J-D-I-B-A-C-G-E-H-F")
DEFAULT_ORDER_MODE = get_env("CATEGORIZE_ORDER_MODE", "csv_position") or "csv_position"
REQUEST_TIMEOUT_SECONDS = get_int_env("AG3NTS_TIMEOUT_SECONDS", 30) or 30
DEFAULT_MODEL = (
    get_optional_env("OPENROUTER_MODEL")
    or get_optional_env("LLM_MODEL")
    or "openai/gpt-4.1-mini"
)
DEFAULT_OPENROUTER_TIMEOUT_SECONDS = get_int_env("OPENROUTER_TIMEOUT_SECONDS", 60) or 60
DEFAULT_PROMPT_PREFIX = (
    "Reply DNG or NEU. Reactor/fuel/cassette/core/rod => NEU. DNG only for weapon, "
    "firearm part, ammo, explosive, poison, corrosive, radioactive, biohazard, "
    "drug. Other industrial/electronic parts => NEU."
)
MODEL_SYSTEM_PROMPT = """You optimize prompts for a tiny binary safety classifier.

Task:
- Produce one short reusable prompt prefix for classifying package descriptions.
- The final prompt will later append:
  ID:<item_id>
  <description>
- Keep the prefix concise so the combined prompt stays under 100 tokens.
- Mark reactor and fuel-system items as NEU, not DNG.
- DNG is only for clearly dangerous goods such as weapons, ammo, explosives,
  poisons, corrosives, radioactive, biohazard, or illicit drugs.
- Ordinary industrial, electronic, mechanical, and reactor-related parts are NEU.

Return JSON only:
{"prompt_prefix":"...","reason":"short explanation"}
"""


@dataclass(slots=True)
class Item:
    item_id: str
    description: str


def get_api_key() -> str:
    return get_course_api_key()


def build_data_url(api_key: str) -> str:
    return build_ag3nts_task_data_url(api_key, f"{TASK_NAME}.csv")


def parse_flag(argv: list[str], flag_name: str) -> bool:
    return flag_name in argv


def parse_optional_value(argv: list[str], flag_name: str) -> str | None:
    for index, arg in enumerate(argv):
        if arg == flag_name:
            try:
                return argv[index + 1].strip()
            except IndexError as exc:
                raise ValueError(f"Missing value after {flag_name}.") from exc
        if arg.startswith(f"{flag_name}="):
            return arg.split("=", 1)[1].strip()
    return None


def http_get_text(url: str) -> str:
    return get_text(
        url,
        headers={"Accept": "text/csv,text/plain;q=0.9,*/*;q=0.8"},
        timeout_seconds=REQUEST_TIMEOUT_SECONDS,
        errors="strict",
    )


def submit_json(url: str, payload: dict[str, Any]) -> Any:
    try:
        return submit_task_answer(
            url,
            api_key=str(payload["apikey"]),
            task=str(payload["task"]),
            answer=dict(payload["answer"]),
            timeout_seconds=REQUEST_TIMEOUT_SECONDS,
        )
    except HttpRequestError as exc:
        raise RuntimeError(json.dumps(exc.to_response_dict(), ensure_ascii=False)) from exc


def parse_items(csv_text: str) -> list[Item]:
    sample = csv_text[:2048]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;|\t")
    except csv.Error:
        dialect = csv.get_dialect("excel")

    reader = csv.DictReader(StringIO(csv_text), dialect=dialect)
    if reader.fieldnames is None:
        raise ValueError("CSV has no header row.")

    normalized = {name.strip().lower(): name for name in reader.fieldnames}
    id_key = next(
        (
            normalized[key]
            for key in ("id", "item_id", "uuid", "code")
            if key in normalized
        ),
        None,
    )
    description_key = next(
        (
            normalized[key]
            for key in ("description", "desc", "name", "item", "product", "towar")
            if key in normalized
        ),
        None,
    )
    if id_key is None or description_key is None:
        raise ValueError(f"Unexpected CSV headers: {reader.fieldnames}")

    items: list[Item] = []
    for row in reader:
        item_id = (row.get(id_key) or "").strip()
        description = (row.get(description_key) or "").strip()
        if item_id and description:
            items.append(Item(item_id=item_id, description=description))

    if not items:
        raise ValueError("No items parsed from CSV.")
    return items


def build_prompt(item: Item, prompt_prefix: str = DEFAULT_PROMPT_PREFIX) -> str:
    return f"{prompt_prefix}\nID:{item.item_id}\n{item.description}"


def parse_send_order(raw_order: str) -> list[str]:
    tokens = [
        token.strip().upper()
        for token in raw_order.replace("/", "-").replace(",", "-").split("-")
        if token.strip()
    ]
    if not tokens:
        raise ValueError("Send order is empty.")

    duplicates = sorted({token for token in tokens if tokens.count(token) > 1})
    if duplicates:
        duplicate_text = ", ".join(duplicates)
        raise ValueError(f"Duplicated item IDs in send order: {duplicate_text}")

    return tokens


def parse_order_mode(argv: list[str]) -> str:
    mode = DEFAULT_ORDER_MODE.strip().lower()
    for index, arg in enumerate(argv):
        if arg == "--order-mode":
            try:
                mode = argv[index + 1].strip().lower()
            except IndexError as exc:
                raise ValueError("Missing value after --order-mode.") from exc
            break

        if arg.startswith("--order-mode="):
            mode = arg.split("=", 1)[1].strip().lower()
            break

    allowed_modes = {"csv_position", "item_id"}
    if mode not in allowed_modes:
        allowed_text = ", ".join(sorted(allowed_modes))
        raise ValueError(
            f"Unsupported order mode: {mode}. Allowed values: {allowed_text}"
        )
    return mode


def order_token_to_index(token: str) -> int:
    if len(token) != 1 or not ("A" <= token <= "Z"):
        raise ValueError(
            "Send order tokens must be single letters from A to Z "
            f"(got: {token})."
        )
    return ord(token) - ord("A")


def resolve_send_order_by_csv_position(
    items: list[Item],
    requested_tokens: list[str],
) -> list[Item]:
    if len(requested_tokens) != len(items):
        raise ValueError(
            "Send order length does not match fetched item count. "
            f"Order has {len(requested_tokens)} entries, CSV has {len(items)} items."
        )

    ordered_items: list[Item] = []
    for token in requested_tokens:
        item_index = order_token_to_index(token)
        if item_index >= len(items):
            raise ValueError(
                f"Send order token {token} points outside CSV range "
                f"(max token for this file is {chr(ord('A') + len(items) - 1)})."
            )
        ordered_items.append(items[item_index])

    return ordered_items


def resolve_send_order_by_item_id(
    items: list[Item],
    requested_tokens: list[str],
) -> list[Item]:
    items_by_id = {item.item_id.upper(): item for item in items}
    available_ids = set(items_by_id)
    requested_set = set(requested_tokens)

    missing_ids = [
        item_id for item_id in requested_tokens if item_id not in available_ids
    ]
    if missing_ids:
        missing_text = ", ".join(missing_ids)
        raise ValueError(f"Send order contains unknown item IDs: {missing_text}")

    extra_ids = sorted(available_ids - requested_set)
    if extra_ids:
        extra_text = ", ".join(extra_ids)
        raise ValueError(
            "Send order does not cover all fetched items. "
            f"Missing in order: {extra_text}"
        )

    return [items_by_id[item_id] for item_id in requested_tokens]


def resolve_send_order(items: list[Item], argv: list[str]) -> tuple[list[Item], str]:
    raw_order = DEFAULT_SEND_ORDER
    for index, arg in enumerate(argv):
        if arg == "--order":
            try:
                raw_order = argv[index + 1]
            except IndexError as exc:
                raise ValueError("Missing value after --order.") from exc
            break

        if arg.startswith("--order="):
            raw_order = arg.split("=", 1)[1]
            break

    requested_tokens = parse_send_order(raw_order)
    order_mode = parse_order_mode(argv)
    if order_mode == "item_id":
        return resolve_send_order_by_item_id(items, requested_tokens), order_mode
    return resolve_send_order_by_csv_position(items, requested_tokens), order_mode


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def build_optional_openrouter_client(argv: list[str]) -> OpenRouterClient | None:
    if parse_flag(argv, "--skip-model"):
        return None

    api_key = (
        get_optional_env("OPENROUTER_API_KEY")
        or get_optional_env("LLM_API_KEY")
        or ""
    ).strip()
    base_url = (
        get_optional_env("OPENROUTER_BASE_URL")
        or get_optional_env("LLM_BASE_URL")
        or ""
    ).strip()
    model = (parse_optional_value(argv, "--model") or DEFAULT_MODEL).strip()
    if not api_key or not base_url or not model:
        return None

    return build_task_openrouter_client(
        __file__,
        api_key=api_key,
        base_url=base_url,
        model=model,
        task_name=TASK_NAME,
        timeout_seconds=float(max(30, DEFAULT_OPENROUTER_TIMEOUT_SECONDS)),
    )


def build_prompt_plan_payload(
    *,
    prompt_prefix: str,
    source: str,
    reason: str,
    items: list[Item],
) -> dict[str, Any]:
    return {
        "source": source,
        "reason": reason,
        "prompt_prefix": prompt_prefix,
        "prefix_estimated_tokens": estimate_tokens(prompt_prefix),
        "item_prompts": [
            {
                "id": item.item_id,
                "estimated_tokens": estimate_tokens(build_prompt(item, prompt_prefix)),
            }
            for item in items
        ],
    }


def validate_model_prompt_prefix(prompt_prefix: str, items: list[Item]) -> str:
    normalized = " ".join(prompt_prefix.split())
    if not normalized:
        raise OpenRouterError("Prompt optimizer returned an empty prefix.")
    if estimate_tokens(normalized) > 70:
        raise OpenRouterError("Prompt optimizer returned a prefix that is too long.")
    over_budget = [
        item.item_id
        for item in items
        if estimate_tokens(build_prompt(item, normalized)) > 100
    ]
    if over_budget:
        raise OpenRouterError(
            "Prompt optimizer exceeded the 100-token budget for items: "
            + ", ".join(over_budget)
        )
    return normalized


def choose_prompt_prefix(
    items: list[Item],
    client: OpenRouterClient | None,
) -> tuple[str, str, str]:
    if client is None:
        return DEFAULT_PROMPT_PREFIX, "deterministic", "OpenRouter unavailable."

    sample_items = "\n".join(f"- {item.item_id}: {item.description}" for item in items)
    messages = [
        {"role": "system", "content": MODEL_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Optimize a prompt prefix for these current items:\n"
                f"{sample_items}"
            ),
        },
    ]
    completion = client.create_completion(messages)
    if not completion.content:
        raise OpenRouterError("Prompt optimizer returned no content.")
    payload = parse_json_object_content(completion.content)
    prompt_prefix = validate_model_prompt_prefix(
        str(payload.get("prompt_prefix", "")),
        items,
    )
    reason = str(payload.get("reason", "")).strip() or "OpenRouter optimized the shared prefix."
    return prompt_prefix, "openrouter", reason


def reset_budget(api_key: str) -> Any:
    return submit_json(
        AG3NTS_VERIFY_URL,
        {
            "apikey": api_key,
            "task": TASK_NAME,
            "answer": {"prompt": "reset"},
        },
    )


def submit_prompt(api_key: str, prompt: str) -> Any:
    return submit_json(
        AG3NTS_VERIFY_URL,
        {
            "apikey": api_key,
            "task": TASK_NAME,
            "answer": {"prompt": prompt},
        },
    )


def print_items(items: list[Item]) -> None:
    logger.info("Fetched items:")
    for item in items:
        prompt = build_prompt(item)
        logger.info(
            f"- {item.item_id}: {item.description} "
            f"(prompt~{estimate_tokens(prompt)} toks)"
        )


def main() -> int:
    configure_logging(name="categorize")
    argv = sys.argv[1:]
    api_key = get_api_key()
    if not api_key:
        logger.error("Missing COURSE_API_KEY in the local repository config.")
        return 1
    data_url = build_data_url(api_key)

    try:
        csv_text = http_get_text(data_url)
        items = parse_items(csv_text)
        items, order_mode = resolve_send_order(items, argv)
    except (HttpRequestError, RuntimeError, ValueError) as exc:
        logger.error("Failed to fetch/parse CSV: {}", exc)
        return 1

    prompt_prefix = DEFAULT_PROMPT_PREFIX
    prompt_source = "deterministic"
    prompt_reason = "Fallback prompt prefix."
    try:
        prompt_prefix, prompt_source, prompt_reason = choose_prompt_prefix(
            items,
            build_optional_openrouter_client(argv),
        )
    except (OpenRouterError, ValueError) as exc:
        logger.warning("Prompt optimization failed, using deterministic prefix: {}", exc)

    PROMPT_PLAN_PATH.write_text(
        json.dumps(
            build_prompt_plan_payload(
                prompt_prefix=prompt_prefix,
                source=prompt_source,
                reason=prompt_reason,
                items=items,
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print_items(items)
    logger.info("Order mode: {}", order_mode)
    logger.info("Send order: {}", " -> ".join(item.item_id for item in items))
    logger.info("Prompt source: {} ({})", prompt_source, prompt_reason)
    logger.info("Using verify endpoint: {}", AG3NTS_VERIFY_URL)

    if "--reset" in argv:
        try:
            response = reset_budget(api_key)
        except RuntimeError as exc:
            logger.error("Reset failed: {}", exc)
            return 1
        logger.info("Reset response:\n{}", json.dumps(response, ensure_ascii=False, indent=2))

    for index, item in enumerate(items, start=1):
        prompt = build_prompt(item, prompt_prefix)
        logger.info("[{}/{}] Sending item {}", index, len(items), item.item_id)
        logger.info("Prompt ({} toks est.): {}", estimate_tokens(prompt), prompt)
        try:
            response = submit_prompt(api_key, prompt)
        except RuntimeError as exc:
            logger.error("Submit failed for {}: {}", item.item_id, exc)
            return 1

        logger.info("Response:\n{}", json.dumps(response, ensure_ascii=False, indent=2))
        serialized = json.dumps(response, ensure_ascii=False)
        if "FLG:" in serialized:
            logger.success("Flag received.")
            return 0

    logger.info("Finished sending all items.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
