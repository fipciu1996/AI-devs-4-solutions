"""Submit or check the negotiations tool registration in AG3NTS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urljoin

from devs_utilities.ag3nts import AG3NTS_VERIFY_URL, submit_task_answer
from devs_utilities.env import load_repo_env
from repo_env import get_course_api_key


TASK_NAME = "negotiations"
TOOL_PATH = "/api/find-cities"


def build_answer(public_base_url: str) -> dict[str, object]:
    """Build the tool declaration payload."""

    normalized_base = public_base_url.rstrip("/") + "/"
    tool_url = urljoin(normalized_base, TOOL_PATH.lstrip("/"))
    description = (
        "Szukaj miast dla jednego przedmiotu opisanego naturalnie w polu "
        "params. Podaj pojedynczy produkt z parametrami, np. "
        "'akumulator AGM 48V 150Ah' albo 'przetwornica 48V 3000W'. "
        "Odpowiedz zawiera dopasowany produkt i liste miast."
    )
    return {
        "tools": [
            {
                "URL": tool_url,
                "description": description,
            }
        ]
    }


def main() -> None:
    """Parse arguments and submit the tool definition."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--public-base-url")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    load_repo_env(repo_root)
    api_key = get_course_api_key()

    if args.check:
        answer = {"action": "check"}
    else:
        if not args.public_base_url:
            parser.error("--public-base-url is required unless --check is used.")
        answer = build_answer(args.public_base_url)

    response = submit_task_answer(
        AG3NTS_VERIFY_URL,
        api_key=api_key,
        task=TASK_NAME,
        answer=answer,
        timeout_seconds=30,
    )
    print(json.dumps(response, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
