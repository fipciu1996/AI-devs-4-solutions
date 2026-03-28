"""Solve the AG3NTS electricity puzzle from the analyzed board state."""

from __future__ import annotations

import argparse
import json
import sys
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
from devs_utilities.flags import extract_flag
from devs_utilities.http import HttpRequestError, get_bytes
from devs_utilities.logging import configure_logging, logger as shared_logger
from repo_env import get_course_api_key, get_env, get_int_env


REPO_ROOT = bootstrap_repo(__file__)
logger = shared_logger.bind(component="electricity")


TASK_NAME = "electricity"
REQUEST_TIMEOUT_SECONDS = get_int_env("AG3NTS_TIMEOUT_SECONDS", 30) or 30

ROTATION_SEQUENCE = [
    token.strip()
    for token in get_env(
        "ELECTRICITY_ROTATION_SEQUENCE",
        "1x2,1x3,2x1,2x2,2x2,2x2,3x1",
    ).split(",")
    if token.strip()
]


def get_api_key() -> str:
    api_key = get_course_api_key()
    if not api_key:
        raise ValueError("Missing COURSE_API_KEY in the local repository config.")
    return api_key


def build_board_url(api_key: str, *, reset: bool = False) -> str:
    suffix = f"{TASK_NAME}.png"
    if reset:
        return f"{build_ag3nts_task_data_url(api_key, suffix)}?reset=1"
    return build_ag3nts_task_data_url(api_key, suffix)


def http_get(url: str) -> bytes:
    return get_bytes(
        url,
        headers={"Accept": "image/png,*/*;q=0.8"},
        timeout_seconds=REQUEST_TIMEOUT_SECONDS,
    )


def http_post_json(url: str, payload: dict[str, object]) -> object:
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


def reset_board(api_key: str) -> None:
    http_get(build_board_url(api_key, reset=True))


def rotate_once(api_key: str, position: str) -> Any:
    return http_post_json(
        AG3NTS_VERIFY_URL,
        {
            "apikey": api_key,
            "task": TASK_NAME,
            "answer": {"rotate": position},
        },
    )


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
    configure_logging(name="electricity")
    args = parse_args()
    api_key = get_api_key()

    logger.info("Planned rotations: {}", " ".join(ROTATION_SEQUENCE))

    if args.dry_run:
        return 0

    if args.reset:
        logger.info("Resetting board...")
        reset_board(api_key)

    for index, position in enumerate(ROTATION_SEQUENCE, start=1):
        response = rotate_once(api_key, position)
        logger.info("[{}/{}] rotate {} -> {}", index, len(ROTATION_SEQUENCE), position, response)
        flag = extract_flag(response)
        if flag:
            logger.success("Flag: {}", flag)
            return 0

    logger.warning("Flag not returned. The board may already be in a different state.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
