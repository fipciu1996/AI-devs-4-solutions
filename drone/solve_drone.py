"""Solve the AG3NTS drone task."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from urllib import error, request


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from repo_env import get_env, load_repo_env


load_repo_env(__file__)


TASK_NAME = "drone"
VERIFY_URL = get_env("AG3NTS_VERIFY_URL")
OUTPUT_DIR = Path(__file__).resolve().parent
LAST_RESPONSE_PATH = OUTPUT_DIR / "last_verify_response.json"
LAST_PROBE_PATH = OUTPUT_DIR / "grid_probe_results.json"

# The map analysis plus API feedback narrows the dam to column 2, row 4
# on a 3x4 grid. The destination object is the Żarnowiec plant code from
# the earlier tasks data set.
DESTINATION_OBJECT = "PWR6132PL"
DAM_SECTOR_X = 2
DAM_SECTOR_Y = 4
GRID_COLUMNS = 3
GRID_ROWS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe-grid",
        action="store_true",
        help="Probe the 3x4 grid and save hub feedback for every sector.",
    )
    parser.add_argument(
        "--power",
        default="100%",
        help="Engine power to set before take-off. Default: 100%%.",
    )
    parser.add_argument(
        "--height",
        default="10m",
        help="Flight height to set before take-off. Default: 10m.",
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
    return parser.parse_args()


def get_api_key() -> str:
    api_key = get_env("AG3NTS_API_KEY")
    if not api_key:
        raise SystemExit("Missing AG3NTS_API_KEY in .env.")
    if not VERIFY_URL:
        raise SystemExit("Missing AG3NTS_VERIFY_URL in .env.")
    return api_key


def post_json(url: str, payload: dict[str, Any]) -> Any:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    http_request = request.Request(
        url=url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(http_request, timeout=30) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"raw": raw}
        payload["http_status"] = exc.code
        return payload
    except error.URLError as exc:
        raise SystemExit(f"Network error for {url}: {exc.reason}") from exc

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


def submit_instructions(api_key: str, instructions: list[str]) -> Any:
    payload = {
        "apikey": api_key,
        "task": TASK_NAME,
        "answer": {"instructions": instructions},
    }
    return post_json(VERIFY_URL, payload)


def save_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
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
            print(f"{column},{row} -> {message}")

    save_json(LAST_PROBE_PATH, results)
    print(f"Saved grid probe results to {LAST_PROBE_PATH}")
    return 0


def main() -> int:
    args = parse_args()
    api_key = get_api_key()

    instructions = build_final_instructions(
        destination=DESTINATION_OBJECT,
        sector_x=args.x,
        sector_y=args.y,
        power=args.power,
        height=args.height,
    )

    if args.probe_grid:
        return probe_grid(api_key)

    if args.dry_run:
        print(json.dumps(instructions, ensure_ascii=False, indent=2))
        return 0

    response = submit_instructions(api_key, instructions)
    save_json(LAST_RESPONSE_PATH, response)

    print(json.dumps(response, ensure_ascii=False, indent=2))

    flag = maybe_extract_flag(response)
    if flag:
        print(flag)
        return 0

    print(f"Flag not found. Full response saved to {LAST_RESPONSE_PATH}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
