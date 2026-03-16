"""Send categorize prompts in a controlled order using shared repo config."""

from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from repo_env import get_env, load_repo_env


load_repo_env(__file__)


VERIFY_URL = get_env("AG3NTS_VERIFY_URL")
DATA_BASE_URL = get_env("AG3NTS_DATA_BASE_URL")
TASK_NAME = "categorize"
DEFAULT_SEND_ORDER = "J-D-I-B-A-C-G-E-H-F"
DEFAULT_ORDER_MODE = "csv_position"
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
    return (
        os.getenv("CATEGORIZE_API_KEY")
        or os.getenv("AG3NTS_API_KEY")
        or os.getenv("PACKAGES_API_KEY")
    )


def build_data_url(api_key: str) -> str:
    if not DATA_BASE_URL:
        raise ValueError("Missing AG3NTS_DATA_BASE_URL in .env.")
    return f"{DATA_BASE_URL.rstrip('/')}/{api_key}/{TASK_NAME}.csv"


def http_get_text(url: str) -> str:
    request = Request(url, headers={"Accept": "text/csv,text/plain;q=0.9,*/*;q=0.8"})
    with urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8")


def http_post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {"raw": raw}
        data["http_status"] = exc.code
        raise RuntimeError(json.dumps(data, ensure_ascii=False)) from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}


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
    mode = os.getenv("CATEGORIZE_ORDER_MODE", DEFAULT_ORDER_MODE).strip().lower()
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


def reset_budget(api_key: str) -> dict[str, Any]:
    payload = {
        "apikey": api_key,
        "task": TASK_NAME,
        "answer": {"prompt": "reset"},
    }
    return http_post_json(VERIFY_URL, payload)


def submit_prompt(api_key: str, prompt: str) -> dict[str, Any]:
    payload = {
        "apikey": api_key,
        "task": TASK_NAME,
        "answer": {"prompt": prompt},
    }
    return http_post_json(VERIFY_URL, payload)


def print_items(items: list[Item]) -> None:
    print("Fetched items:")
    for item in items:
        prompt = build_prompt(item)
        print(
            f"- {item.item_id}: {item.description} "
            f"(prompt~{estimate_tokens(prompt)} toks)"
        )


def main() -> int:
    api_key = get_api_key()
    if not api_key:
        print("Missing CATEGORIZE_API_KEY/AG3NTS_API_KEY/PACKAGES_API_KEY.", file=sys.stderr)
        return 1
    if not VERIFY_URL:
        print("Missing AG3NTS_VERIFY_URL in .env.", file=sys.stderr)
        return 1
    data_url = build_data_url(api_key)

    try:
        csv_text = http_get_text(data_url)
        items = parse_items(csv_text)
        items, order_mode = resolve_send_order(items, sys.argv[1:])
    except (HTTPError, URLError, TimeoutError, RuntimeError, ValueError) as exc:
        print(f"Failed to fetch/parse CSV: {exc}", file=sys.stderr)
        return 1

    print_items(items)
    print(f"\nOrder mode: {order_mode}")
    print(
        "\nSend order: "
        + " -> ".join(item.item_id for item in items)
    )
    print(f"\nUsing verify endpoint: {VERIFY_URL}")

    if "--reset" in sys.argv:
        try:
            response = reset_budget(api_key)
        except (HTTPError, URLError, TimeoutError, RuntimeError) as exc:
            print(f"Reset failed: {exc}", file=sys.stderr)
            return 1
        print("\nReset response:")
        print(json.dumps(response, ensure_ascii=False, indent=2))

    for index, item in enumerate(items, start=1):
        prompt = build_prompt(item)
        print(f"\n[{index}/{len(items)}] Sending item {item.item_id}")
        print(f"Prompt ({estimate_tokens(prompt)} toks est.): {prompt}")
        try:
            response = submit_prompt(api_key, prompt)
        except (HTTPError, URLError, TimeoutError, RuntimeError) as exc:
            print(f"Submit failed for {item.item_id}: {exc}", file=sys.stderr)
            return 1

        print(json.dumps(response, ensure_ascii=False, indent=2))
        serialized = json.dumps(response, ensure_ascii=False)
        if "FLG:" in serialized:
            print("\nFlag received.")
            return 0

    print("\nFinished sending all items.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
