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
from repo_env import get_env, get_int_env


REPO_ROOT = bootstrap_repo(__file__)
logger = shared_logger.bind(component="categorize")


TASK_NAME = "categorize"
DEFAULT_SEND_ORDER = get_env("CATEGORIZE_SEND_ORDER", "J-D-I-B-A-C-G-E-H-F")
DEFAULT_ORDER_MODE = get_env("CATEGORIZE_ORDER_MODE", "csv_position") or "csv_position"
REQUEST_TIMEOUT_SECONDS = get_int_env("AG3NTS_TIMEOUT_SECONDS", 30) or 30
PROMPT_TEMPLATE = (
    "Reply DNG or NEU. Reactor/fuel/cassette/core/rod => NEU. DNG only for weapon, "
    "firearm part, ammo, explosive, poison, corrosive, radioactive, biohazard, "
    "drug. Other industrial/electronic parts => NEU.\n"
    "ID:{id}\n{description}"
)


@dataclass(slots=True)
class Item:
    item_id: str
    description: str


def get_api_key() -> str:
    return get_env("AG3NTS_API_KEY")


def build_data_url(api_key: str) -> str:
    return build_ag3nts_task_data_url(api_key, f"{TASK_NAME}.csv")


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


def build_prompt(item: Item) -> str:
    return PROMPT_TEMPLATE.format(id=item.item_id, description=item.description)


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
    api_key = get_api_key()
    if not api_key:
        logger.error("Missing AG3NTS_API_KEY in .env.")
        return 1
    data_url = build_data_url(api_key)

    try:
        csv_text = http_get_text(data_url)
        items = parse_items(csv_text)
        items, order_mode = resolve_send_order(items, sys.argv[1:])
    except (HttpRequestError, RuntimeError, ValueError) as exc:
        logger.error("Failed to fetch/parse CSV: {}", exc)
        return 1

    print_items(items)
    logger.info("Order mode: {}", order_mode)
    logger.info("Send order: {}", " -> ".join(item.item_id for item in items))
    logger.info("Using verify endpoint: {}", AG3NTS_VERIFY_URL)

    if "--reset" in sys.argv:
        try:
            response = reset_budget(api_key)
        except RuntimeError as exc:
            logger.error("Reset failed: {}", exc)
            return 1
        logger.info("Reset response:\n{}", json.dumps(response, ensure_ascii=False, indent=2))

    for index, item in enumerate(items, start=1):
        prompt = build_prompt(item)
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
