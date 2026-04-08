"""Solve the AG3NTS drone task."""

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

from devs_utilities.ag3nts import AG3NTS_VERIFY_URL, submit_task_answer
from devs_utilities.bootstrap import bootstrap_repo
from devs_utilities.files import write_json
from devs_utilities.flags import extract_flag
from devs_utilities.http import HttpRequestError
from devs_utilities.logging import configure_logging, logger as shared_logger
from devs_utilities.openrouter import (
    build_task_openrouter_client,
    OpenRouterClient,
    OpenRouterError,
    parse_json_object_content,
)
from devs_utilities.repo_env import (
    get_course_api_key,
    get_env,
    get_int_env,
    get_llm_model,
    get_optional_env,
)


REPO_ROOT = bootstrap_repo(__file__)
logger = shared_logger.bind(component="drone")


TASK_NAME = "drone"
OUTPUT_DIR = Path(__file__).resolve().parent
LAST_RESPONSE_PATH = OUTPUT_DIR / "last_verify_response.json"
LAST_PROBE_PATH = OUTPUT_DIR / "grid_probe_results.json"
LAST_MODEL_PLAN_PATH = OUTPUT_DIR / "last_model_target.json"
MAP_IMAGE_PATH = OUTPUT_DIR / "drone.png"
DEFAULT_MODEL = get_llm_model("DRONE_MODEL")
DEFAULT_OPENROUTER_TIMEOUT_SECONDS = get_int_env("OPENROUTER_TIMEOUT_SECONDS", 60) or 60
VISION_SYSTEM_PROMPT = """You analyze a 3x4 tactical grid map for a drone mission.

Task:
- Inspect the map image.
- Find the sector containing the dam, not the power plant.
- Return the target sector as 1-based grid coordinates.
- The grid has 3 columns and 4 rows.

Return JSON only:
{"sector_x":2,"sector_y":4,"reason":"short explanation"}
"""

# The map analysis plus API feedback narrows the dam to column 2, row 4
# on a 3x4 grid. The destination object is the Żarnowiec plant code from
# the earlier tasks data set.
DESTINATION_OBJECT = get_env("DRONE_DESTINATION_OBJECT", "PWR6132PL")
DAM_SECTOR_X = get_int_env("DRONE_DAM_SECTOR_X", 2) or 2
DAM_SECTOR_Y = get_int_env("DRONE_DAM_SECTOR_Y", 4) or 4
GRID_COLUMNS = get_int_env("DRONE_GRID_COLUMNS", 3) or 3
GRID_ROWS = get_int_env("DRONE_GRID_ROWS", 4) or 4
DEFAULT_POWER = get_env("DRONE_DEFAULT_POWER", "100%") or "100%"
DEFAULT_HEIGHT = get_env("DRONE_DEFAULT_HEIGHT", "10m") or "10m"
REQUEST_TIMEOUT_SECONDS = get_int_env("AG3NTS_TIMEOUT_SECONDS", 30) or 30


def escape_argparse_help(value: str) -> str:
    """Escape percent signs so argparse does not treat them as format markers."""

    return value.replace("%", "%%")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe-grid",
        action="store_true",
        help="Probe the 3x4 grid and save hub feedback for every sector.",
    )
    parser.add_argument(
        "--power",
        default=DEFAULT_POWER,
        help=(
            "Engine power to set before take-off. "
            f"Default: {escape_argparse_help(DEFAULT_POWER)}."
        ),
    )
    parser.add_argument(
        "--height",
        default=DEFAULT_HEIGHT,
        help=f"Flight height to set before take-off. Default: {DEFAULT_HEIGHT}.",
    )
    parser.add_argument(
        "--x",
        type=int,
        default=DAM_SECTOR_X,
        help=f"Target sector column. Default: {DAM_SECTOR_X}.",
    )
    parser.add_argument(
        "--y",
        type=int,
        default=DAM_SECTOR_Y,
        help=f"Target sector row. Default: {DAM_SECTOR_Y}.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the instructions without sending them to the hub.",
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Disable OpenRouter map analysis and use the deterministic target sector.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"OpenRouter model override. Default: {DEFAULT_MODEL}.",
    )
    return parser.parse_args()


def get_api_key() -> str:
    api_key = get_course_api_key()
    if not api_key:
        raise SystemExit("Missing COURSE_API_KEY in the local repository config.")
    return api_key


def post_json(url: str, payload: dict[str, Any]) -> Any:
    try:
        return submit_task_answer(
            url,
            api_key=str(payload["apikey"]),
            task=str(payload["task"]),
            answer=dict(payload["answer"]),
            timeout_seconds=REQUEST_TIMEOUT_SECONDS,
        )
    except HttpRequestError as exc:
        return exc.to_response_dict()


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


def build_final_instructions(
    *,
    destination: str,
    sector_x: int,
    sector_y: int,
    power: str,
    height: str,
) -> list[str]:
    return [
        "hardReset",
        f"setDestinationObject({destination})",
        f"set({sector_x},{sector_y})",
        "set(engineON)",
        f"set({power})",
        f"set({height})",
        "set(destroy)",
        "set(return)",
        "flyToLocation",
    ]


def build_probe_instructions(*, destination: str, sector_x: int, sector_y: int) -> list[str]:
    return [
        "hardReset",
        f"setDestinationObject({destination})",
        f"set({sector_x},{sector_y})",
        "set(100%)",
        "set(10m)",
        "set(destroy)",
        "flyToLocation",
    ]


def validate_target_sector(raw_payload: dict[str, Any]) -> tuple[int, int, str]:
    sector_x = int(raw_payload.get("sector_x"))
    sector_y = int(raw_payload.get("sector_y"))
    if not 1 <= sector_x <= GRID_COLUMNS or not 1 <= sector_y <= GRID_ROWS:
        raise OpenRouterError(
            f"Model target sector is outside the {GRID_COLUMNS}x{GRID_ROWS} grid."
        )
    reason = str(raw_payload.get("reason", "")).strip() or "OpenRouter analyzed the tactical map."
    return sector_x, sector_y, reason


def choose_target_sector(
    args: argparse.Namespace,
    client: OpenRouterClient | None,
) -> tuple[int, int, str, str]:
    if args.x != DAM_SECTOR_X or args.y != DAM_SECTOR_Y:
        return args.x, args.y, "explicit", "Command-line sector override."
    if client is None or not MAP_IMAGE_PATH.exists():
        return args.x, args.y, "deterministic", "OpenRouter unavailable."

    encoded = base64.b64encode(MAP_IMAGE_PATH.read_bytes()).decode("ascii")
    completion = client.create_completion(
        [
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Find the dam sector on this drone map."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded}"},
                    },
                ],
            },
        ]
    )
    if not completion.content:
        raise OpenRouterError("Drone map analysis returned no content.")
    payload = parse_json_object_content(completion.content)
    sector_x, sector_y, reason = validate_target_sector(payload)
    write_json(
        LAST_MODEL_PLAN_PATH,
        {
            "source": "openrouter",
            "sector_x": sector_x,
            "sector_y": sector_y,
            "reason": reason,
        },
    )
    return sector_x, sector_y, "openrouter", reason


def submit_instructions(api_key: str, instructions: list[str]) -> Any:
    return post_json(
        AG3NTS_VERIFY_URL,
        {
            "apikey": api_key,
            "task": TASK_NAME,
            "answer": {"instructions": instructions},
        },
    )


def probe_grid(api_key: str) -> int:
    results: list[dict[str, Any]] = []
    for row in range(1, GRID_ROWS + 1):
        for column in range(1, GRID_COLUMNS + 1):
            instructions = build_probe_instructions(
                destination=DESTINATION_OBJECT,
                sector_x=column,
                sector_y=row,
            )
            response = submit_instructions(api_key, instructions)
            message = response.get("message", "") if isinstance(response, dict) else str(response)
            results.append(
                {
                    "column": column,
                    "row": row,
                    "instructions": instructions,
                    "response": response,
                    "message": message,
                }
            )
            logger.info("{},{} -> {}", column, row, message)

    write_json(LAST_PROBE_PATH, results, trailing_newline=False)
    logger.info("Saved grid probe results to {}", LAST_PROBE_PATH)
    return 0


def main() -> int:
    configure_logging(name="drone")
    args = parse_args()
    api_key = get_api_key()
    sector_x = args.x
    sector_y = args.y
    target_source = "deterministic"
    target_reason = "Configured default sector."
    try:
        sector_x, sector_y, target_source, target_reason = choose_target_sector(
            args,
            build_optional_openrouter_client(args),
        )
    except (OpenRouterError, ValueError) as exc:
        logger.warning("Map analysis failed, using deterministic target sector: {}", exc)

    instructions = build_final_instructions(
        destination=DESTINATION_OBJECT,
        sector_x=sector_x,
        sector_y=sector_y,
        power=args.power,
        height=args.height,
    )

    if args.probe_grid:
        return probe_grid(api_key)

    logger.info(
        "Target sector: ({}, {}) via {} ({})",
        sector_x,
        sector_y,
        target_source,
        target_reason,
    )

    if args.dry_run:
        logger.info("Instructions:\n{}", json.dumps(instructions, ensure_ascii=False, indent=2))
        return 0

    response = submit_instructions(api_key, instructions)
    write_json(LAST_RESPONSE_PATH, response)

    logger.info("Verify response:\n{}", json.dumps(response, ensure_ascii=False, indent=2))

    flag = extract_flag(response)
    if flag:
        logger.success("Flag: {}", flag)
        return 0

    logger.warning("Flag not found. Full response saved to {}", LAST_RESPONSE_PATH)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
