"""Solve the AG3NTS electricity puzzle from the analyzed board state."""

from __future__ import annotations

import argparse
import base64
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
from devs_utilities.openrouter import (
    build_task_openrouter_client,
    OpenRouterClient,
    OpenRouterError,
    parse_json_object_content,
)
from devs_utilities.prompts import load_prompt_text
from devs_utilities.repo_env import (
    get_course_api_key,
    get_env,
    get_int_env,
    get_llm_model,
    get_optional_env,
)


REPO_ROOT = bootstrap_repo(__file__)
logger = shared_logger.bind(component="electricity")


TASK_NAME = "electricity"
REQUEST_TIMEOUT_SECONDS = get_int_env("AG3NTS_TIMEOUT_SECONDS", 30) or 30
DEFAULT_MODEL = get_llm_model("ELECTRICITY_MODEL")
DEFAULT_OPENROUTER_TIMEOUT_SECONDS = get_int_env("OPENROUTER_TIMEOUT_SECONDS", 60) or 60
OUTPUT_DIR = Path(__file__).resolve().parent
CURRENT_BOARD_PATH = OUTPUT_DIR / "current.png"
SOLVED_BOARD_PATH = OUTPUT_DIR / "solved.png"
MODEL_PLAN_PATH = OUTPUT_DIR / "last_model_plan.json"

ROTATION_SEQUENCE = [
    token.strip()
    for token in get_env(
        "ELECTRICITY_ROTATION_SEQUENCE",
        "1x2,1x3,2x1,2x2,2x2,2x2,3x1",
    ).split(",")
    if token.strip()
]
VISION_SYSTEM_PROMPT = load_prompt_text(__file__, "vision_system_prompt.txt")


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
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Disable OpenRouter board analysis and use the deterministic sequence only.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"OpenRouter model override. Default: {DEFAULT_MODEL}.",
    )
    return parser.parse_args()


def build_optional_openrouter_client(args: argparse.Namespace) -> OpenRouterClient | None:
    if args.skip_model:
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
    model = (args.model or DEFAULT_MODEL).strip()
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


def normalize_rotation_token(raw_value: str) -> str:
    normalized = raw_value.strip()
    if not normalized:
        raise OpenRouterError("Rotation token cannot be empty.")
    row_text, separator, column_text = normalized.partition("x")
    if separator != "x":
        raise OpenRouterError(f"Rotation token must use row x column format, got: {raw_value!r}")
    row = int(row_text)
    column = int(column_text)
    if not 1 <= row <= 3 or not 1 <= column <= 3:
        raise OpenRouterError(f"Rotation token is outside the 3x3 board: {raw_value!r}")
    return f"{row}x{column}"


def validate_rotation_sequence(raw_rotations: Any) -> list[str]:
    if not isinstance(raw_rotations, list) or not raw_rotations:
        raise OpenRouterError("Rotation plan must be a non-empty list.")
    normalized = [normalize_rotation_token(str(item)) for item in raw_rotations]
    if len(normalized) > 20:
        raise OpenRouterError("Rotation plan is unexpectedly long.")
    return normalized


def load_board_image_bytes(api_key: str) -> bytes:
    try:
        board_bytes = http_get(build_board_url(api_key))
        CURRENT_BOARD_PATH.write_bytes(board_bytes)
        return board_bytes
    except HttpRequestError:
        if CURRENT_BOARD_PATH.exists():
            return CURRENT_BOARD_PATH.read_bytes()
        raise


def prime_remote_board_state(api_key: str) -> None:
    """Ensure the hub has the live map loaded for this session."""

    board_bytes = http_get(build_board_url(api_key))
    CURRENT_BOARD_PATH.write_bytes(board_bytes)


def build_image_content(label: str, image_bytes: bytes) -> list[dict[str, Any]]:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return [
        {"type": "text", "text": label},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
    ]


def choose_rotation_sequence(
    api_key: str,
    client: OpenRouterClient | None,
) -> tuple[list[str], str, str]:
    if client is None:
        return ROTATION_SEQUENCE, "deterministic", "OpenRouter unavailable."

    current_board = load_board_image_bytes(api_key)
    content: list[dict[str, Any]] = []
    content.extend(build_image_content("Current board:", current_board))
    if SOLVED_BOARD_PATH.exists():
        content.extend(build_image_content("Solved reference board:", SOLVED_BOARD_PATH.read_bytes()))

    completion = client.create_completion(
        [
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
    )
    if not completion.content:
        raise OpenRouterError("Board analysis returned no content.")
    payload = parse_json_object_content(completion.content)
    rotations = validate_rotation_sequence(payload.get("rotations"))
    reason = str(payload.get("reason", "")).strip() or "OpenRouter analyzed the board image."
    MODEL_PLAN_PATH.write_text(
        json.dumps(
            {
                "source": "openrouter",
                "reason": reason,
                "rotations": rotations,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return rotations, "openrouter", reason


def execute_rotation_sequence(api_key: str, rotation_sequence: list[str]) -> str | None:
    for index, position in enumerate(rotation_sequence, start=1):
        response = rotate_once(api_key, position)
        logger.info("[{}/{}] rotate {} -> {}", index, len(rotation_sequence), position, response)
        flag = extract_flag(response)
        if flag:
            return flag
    return None


def main() -> int:
    configure_logging(name="electricity")
    args = parse_args()
    api_key = get_api_key()

    if args.reset:
        logger.info("Resetting board...")
        reset_board(api_key)

    rotation_sequence = ROTATION_SEQUENCE
    rotation_source = "deterministic"
    rotation_reason = "Fallback sequence."
    try:
        rotation_sequence, rotation_source, rotation_reason = choose_rotation_sequence(
            api_key,
            build_optional_openrouter_client(args),
        )
    except (HttpRequestError, OpenRouterError, ValueError) as exc:
        logger.warning("Board analysis failed, using deterministic sequence: {}", exc)

    logger.info("Planned rotations: {}", " ".join(rotation_sequence))
    logger.info("Rotation source: {} ({})", rotation_source, rotation_reason)

    if args.dry_run:
        return 0

    try:
        prime_remote_board_state(api_key)
    except HttpRequestError as exc:
        logger.error("Failed to fetch the live board before rotating: {}", exc)
        return 1

    flag = execute_rotation_sequence(api_key, rotation_sequence)
    if flag:
        logger.success("Flag: {}", flag)
        return 0

    if rotation_source != "deterministic" and rotation_sequence != ROTATION_SEQUENCE:
        logger.warning(
            "Flag not returned for the model plan. Resetting the board and trying the deterministic fallback."
        )
        try:
            reset_board(api_key)
            prime_remote_board_state(api_key)
        except HttpRequestError as exc:
            logger.error("Failed to prepare the fallback board state: {}", exc)
            return 1
        flag = execute_rotation_sequence(api_key, ROTATION_SEQUENCE)
        if flag:
            logger.success("Flag: {}", flag)
            return 0

    logger.warning("Flag not returned. The board may already be in a different state.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
