from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
PEOPLE_DIR = REPO_ROOT / "people"
for candidate in (str(REPO_ROOT), str(PEOPLE_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from devs_utilities.openrouter import ChatCompletionResult
from filter_people import (
    AppConfig,
    Person,
    classify_jobs,
    load_config,
    parse_classification_result,
)


class StubOpenRouterClient:
    def __init__(self, completions: list[ChatCompletionResult]) -> None:
        self._completions = list(completions)
        self.calls: list[list[dict[str, object]]] = []

    def create_completion(self, messages, **_kwargs):  # type: ignore[no-untyped-def]
        self.calls.append([dict(message) for message in messages])
        return self._completions.pop(0)


class PeopleFilterTests(unittest.TestCase):
    def test_load_config_uses_repo_env_model_instead_of_json_override(self) -> None:
        config_path = Path("people_config.json")
        args = SimpleNamespace(model=None, batch_size=None)

        with (
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "read_text", return_value='{"openrouter_model":"json-model"}'),
            patch("filter_people.DEFAULT_OPENROUTER_MODEL", "env-model"),
            patch("filter_people.get_llm_api_key", return_value="llm-token"),
            patch("filter_people.get_course_api_key", return_value="course-token"),
            patch("filter_people.get_optional_env", return_value=""),
        ):
            config = load_config(config_path, args)

        self.assertEqual(config.openrouter_model, "env-model")

    def test_parse_classification_result_accepts_alternate_items_key(self) -> None:
        parsed = parse_classification_result(
            '{"items":[{"row_id":7,"tags":["transport","praca z pojazdami"]}]}'
        )

        self.assertEqual(parsed, {7: ["transport", "praca z pojazdami"]})

    def test_parse_classification_result_accepts_single_result_object(self) -> None:
        parsed = parse_classification_result(
            '{"row_id":11,"tags":["transport"]}'
        )

        self.assertEqual(parsed, {11: ["transport"]})

    def test_classify_jobs_retries_after_invalid_non_tool_response(self) -> None:
        client = StubOpenRouterClient(
            [
                ChatCompletionResult(content='{"status":"ok"}', tool_calls=[]),
                ChatCompletionResult(
                    content='{"results":[{"row_id":1,"tags":["transport"]}]}',
                    tool_calls=[],
                ),
            ]
        )
        config = AppConfig(
            course_api_key="course",
            llm_api_key="llm",
            openrouter_model="model",
            site_url=None,
            site_name=None,
            batch_size=10,
        )
        batch = [
            Person(
                row_id=1,
                name="Jan",
                surname="Kowalski",
                gender="M",
                birth_date=date(2000, 1, 1),
                birth_place="Grudziądz",
                birth_country="Poland",
                job="Kierowca autobusu",
            )
        ]

        result = classify_jobs(batch, config, client)

        self.assertEqual(result, {1: ["transport"]})
        self.assertEqual(len(client.calls), 2)
        self.assertIn("Return only valid JSON", str(client.calls[1][-1]["content"]))


if __name__ == "__main__":
    unittest.main()
