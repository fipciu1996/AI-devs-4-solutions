"""Solve the AG3NTS electricity puzzle from the analyzed board state."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from repo_env import get_env, load_repo_env


load_repo_env(__file__)


TASK_NAME = "electricity"
VERIFY_URL = get_env("AG3NTS_VERIFY_URL")
DATA_BASE_URL = get_env("AG3NTS_DATA_BASE_URL")

# Sequence derived from the board downloaded on 2026-03-17.
ROTATION_SEQUENCE = [
    "1x2",
    "1x3",
    "2x1",
    "2x2",
    "2x2",
    "2x2",
    "3x1",
]


def get_api_key() -> str:
    api_key = get_env("AG3NTS_API_KEY")
    if not api_key:
        raise ValueError("Missing AG3NTS_API_KEY in .env.")
    return api_key


def build_board_url(api_key: str, *, reset: bool = False) -> str:
    if not DATA_BASE_URL:
        raise ValueError("Missing AG3NTS_DATA_BASE_URL in .env.")

    suffix = f"/{api_key}/{TASK_NAME}.png"
    if reset:
        suffix += "?reset=1"
    return f"{DATA_BASE_URL.rstrip('/')}{suffix}"


def http_get(url: str) -> bytes:
    request = Request(url, headers={"Accept": "image/png,*/*;q=0.8"})
    with urlopen(request, timeout=30) as response:
        return response.read()


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


def maybe_extract_flag(payload: Any) -> str | None:
    if isinstance(payload, str):
        return payload if payload.startswith("{FLG:") else None

    if isinstance(payload, dict):
        for value in payload.values():
            flag = maybe_extract_flag(value)
            if flag:
                return flag
        return None

    if isinstance(payload, list):
        for item in payload:
            flag = maybe_extract_flag(item)
            if flag:
                return flag

    return None


def reset_board(api_key: str) -> None:
    http_get(build_board_url(api_key, reset=True))


def rotate_once(api_key: str, position: str) -> dict[str, Any]:
    payload = {
        "apikey": api_key,
        "task": TASK_NAME,
        "answer": {"rotate": position},
    }
    return http_post_json(VERIFY_URL, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset the board to the initial state before sending rotations.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned rotations without sending them to the hub.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = get_api_key()

    print(f"Planned rotations: {' '.join(ROTATION_SEQUENCE)}")

    if args.dry_run:
        return 0

    if not VERIFY_URL:
        raise ValueError("Missing AG3NTS_VERIFY_URL in .env.")

    if args.reset:
        print("Resetting board...")
        reset_board(api_key)

    for index, position in enumerate(ROTATION_SEQUENCE, start=1):
        response = rotate_once(api_key, position)
        print(f"[{index}/{len(ROTATION_SEQUENCE)}] rotate {position} -> {response}")
        flag = maybe_extract_flag(response)
        if flag:
            print(flag)
            return 0

    print("Flag not returned. The board may already be in a different state.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
