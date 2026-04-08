from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

import devs_utilities.openrouter as openrouter_module
from devs_utilities.http import HttpRequestError
from devs_utilities.openrouter import (
    OpenRouterError,
    OpenRouterClient,
    OpenRouterConfig,
    build_task_site_name,
    build_task_usage_output_path,
    extract_usage_snapshot,
    extract_completion_result,
    extract_tool_calls,
)


class OpenRouterParsingTests(unittest.TestCase):
    def test_extract_completion_result_reads_string_content(self) -> None:
        result = extract_completion_result(
            {
                "choices": [
                    {
                        "message": {
                            "content": "  hello world  ",
                        }
                    }
                ]
            }
        )

        self.assertEqual(result.content, "hello world")
        self.assertEqual(result.tool_calls, [])

    def test_extract_completion_result_reads_text_blocks(self) -> None:
        result = extract_completion_result(
            {
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "text", "text": "first line"},
                                {"type": "image", "image_url": "ignored"},
                                {"type": "text", "text": "second line"},
                            ]
                        }
                    }
                ]
            }
        )

        self.assertEqual(result.content, "first line\nsecond line")

    def test_extract_tool_calls_parses_valid_arguments(self) -> None:
        result = extract_tool_calls(
            [
                {
                    "id": "call_1",
                    "function": {
                        "name": "search_messages",
                        "arguments": '{"query":"from:proton.me"}',
                    },
                }
            ]
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "search_messages")
        self.assertEqual(result[0].arguments, {"query": "from:proton.me"})

    def test_extract_tool_calls_rejects_invalid_json_arguments(self) -> None:
        with self.assertRaises(OpenRouterError):
            extract_tool_calls(
                [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "search_messages",
                            "arguments": "{not json}",
                        },
                    }
                ]
            )

    def test_extract_tool_calls_wraps_top_level_lists_for_batch_get_records(self) -> None:
        result = extract_tool_calls(
            [
                {
                    "id": "call_1",
                    "function": {
                        "name": "batch_get_records",
                        "arguments": '[{"page":"incydenty","id":"abc"}]',
                    },
                }
            ]
        )

        self.assertEqual(result[0].arguments, {"requests": [{"page": "incydenty", "id": "abc"}]})

    def test_extract_tool_calls_repairs_python_style_arguments(self) -> None:
        result = extract_tool_calls(
            [
                {
                    "id": "call_1",
                    "function": {
                        "name": "batch_get_records",
                        "arguments": "requests=[{'page':'incydenty','id':'abc'}]",
                    },
                }
            ]
        )

        self.assertEqual(result[0].arguments, {"requests": [{"page": "incydenty", "id": "abc"}]})

    def test_extract_tool_calls_balances_missing_closing_braces(self) -> None:
        result = extract_tool_calls(
            [
                {
                    "id": "call_1",
                    "function": {
                        "name": "update_record",
                        "arguments": '{"page":"incydenty","id":"abc"',
                    },
                }
            ]
        )

        self.assertEqual(result[0].arguments, {"page": "incydenty", "id": "abc"})

    def test_extract_tool_calls_sanitizes_tool_name_noise(self) -> None:
        result = extract_tool_calls(
            [
                {
                    "id": "call_1",
                    "function": {
                        "name": "run_shell_command<|channel|>json",
                        "arguments": '{"command":"pwd"}',
                    },
                }
            ]
        )

        self.assertEqual(result[0].name, "run_shell_command")

    def test_extract_usage_snapshot_reads_cached_and_reasoning_tokens(self) -> None:
        usage = extract_usage_snapshot(
            {
                "usage": {
                    "prompt_tokens": 194,
                    "completion_tokens": 2,
                    "total_tokens": 196,
                    "prompt_tokens_details": {
                        "cached_tokens": 32,
                        "cache_write_tokens": 100,
                    },
                    "completion_tokens_details": {
                        "reasoning_tokens": 7,
                    },
                }
            }
        )

        self.assertEqual(usage.input_tokens, 194)
        self.assertEqual(usage.output_tokens, 2)
        self.assertEqual(usage.cached_tokens, 32)
        self.assertEqual(usage.reasoning_tokens, 7)
        self.assertEqual(usage.cache_write_tokens, 100)
        self.assertEqual(usage.total_tokens, 196)

    def test_build_task_site_name_uses_explicit_task_name(self) -> None:
        site_name = build_task_site_name("C:/repo/findhim/solve_findhim.py", task_name="findhim")

        self.assertEqual(site_name, "AI Devs 4 - findhim")

    def test_build_task_usage_output_path_uses_task_specific_name_when_needed(self) -> None:
        output_path = build_task_usage_output_path(
            "C:/repo/sensors/solve_sensors.py",
            task_name="evaluation",
        )

        self.assertEqual(output_path.as_posix(), "C:/repo/sensors/openrouter_usage_evaluation.json")

    def test_build_task_usage_output_path_respects_explicit_env_output_path(self) -> None:
        with patch(
            "devs_utilities.openrouter.get_optional_env",
            side_effect=lambda name: (
                "C:/repo/fire_in_the_hole_costs/people.json"
                if name == "OPENROUTER_USAGE_OUTPUT_PATH"
                else None
            ),
        ):
            output_path = build_task_usage_output_path(
                "C:/repo/people/filter_people.py",
                task_name="people",
            )

        self.assertEqual(output_path.as_posix(), "C:/repo/fire_in_the_hole_costs/people.json")

    def test_usage_tracker_loads_existing_report_between_client_instances(self) -> None:
        usage_path = Path(__file__).with_name("_openrouter_usage_tracker.json")
        usage_path.unlink(missing_ok=True)
        openrouter_module.OpenRouterClient._trackers.clear()

        first_response = {
            "model": "model-a",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
            },
            "choices": [{"message": {"content": "first"}}],
        }
        second_response = {
            "model": "model-a",
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 3,
                "total_tokens": 8,
            },
            "choices": [{"message": {"content": "second"}}],
        }

        try:
            with patch("devs_utilities.openrouter.post_json", return_value=first_response):
                first_client = OpenRouterClient(
                    OpenRouterConfig(
                        api_key="token",
                        base_url="https://openrouter.example/api/v1/chat/completions",
                        model="model-a",
                        usage_output_path=usage_path,
                        usage_task_name="sendit",
                    )
                )
                first_client.create_raw_completion([{"role": "user", "content": "one"}])

            openrouter_module.OpenRouterClient._trackers.clear()

            with patch("devs_utilities.openrouter.post_json", return_value=second_response):
                second_client = OpenRouterClient(
                    OpenRouterConfig(
                        api_key="token",
                        base_url="https://openrouter.example/api/v1/chat/completions",
                        model="model-a",
                        usage_output_path=usage_path,
                        usage_task_name="sendit",
                    )
                )
                second_client.create_raw_completion([{"role": "user", "content": "two"}])

            payload = json.loads(usage_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["request_count"], 2)
            self.assertEqual(payload["totals"]["input_tokens"], 15)
            self.assertEqual(payload["totals"]["output_tokens"], 5)
            self.assertEqual(payload["totals"]["total_tokens"], 20)
        finally:
            openrouter_module.OpenRouterClient._trackers.clear()
            usage_path.unlink(missing_ok=True)

    def test_create_completion_retries_after_empty_response(self) -> None:
        client = OpenRouterClient(
            OpenRouterConfig(
                api_key="token",
                base_url="https://openrouter.example/api/v1/chat/completions",
                model="openai/gpt-4.1-mini",
            )
        )

        with patch.object(
            client,
            "create_raw_completion",
            side_effect=[
                {"choices": [{"message": {"content": None}}]},
                {"choices": [{"message": {"content": "working reply"}}]},
            ],
        ) as mock_create_raw_completion:
            result = client.create_completion([{"role": "user", "content": "hello"}])

        self.assertEqual(result.content, "working reply")
        self.assertEqual(mock_create_raw_completion.call_count, 2)

    def test_create_raw_completion_retries_after_429(self) -> None:
        client = OpenRouterClient(
            OpenRouterConfig(
                api_key="token",
                base_url="https://openrouter.example/api/v1/chat/completions",
                model="openrouter/free",
                retry_attempts=3,
                retry_base_delay_seconds=2.0,
                retry_max_delay_seconds=10.0,
            )
        )
        response_payload = {
            "choices": [{"message": {"content": "working reply"}}],
        }

        with patch(
            "devs_utilities.openrouter.post_json",
            side_effect=[
                HttpRequestError(
                    url="https://openrouter.example/api/v1/chat/completions",
                    message="HTTP 429",
                    status_code=429,
                ),
                response_payload,
            ],
        ) as mock_post_json, patch("devs_utilities.openrouter.time.sleep") as mock_sleep:
            parsed = client.create_raw_completion([{"role": "user", "content": "hello"}])

        self.assertEqual(parsed, response_payload)
        self.assertEqual(mock_post_json.call_count, 2)
        mock_sleep.assert_called_once_with(2.0)

    def test_create_raw_completion_uses_rate_limit_reset_from_error_payload(self) -> None:
        client = OpenRouterClient(
            OpenRouterConfig(
                api_key="token",
                base_url="https://openrouter.example/api/v1/chat/completions",
                model="openrouter/free",
                retry_attempts=2,
                retry_base_delay_seconds=2.0,
                retry_max_delay_seconds=90.0,
            )
        )
        reset_timestamp = 1_700_000_030
        error = HttpRequestError(
            url="https://openrouter.example/api/v1/chat/completions",
            message="HTTP 429",
            status_code=429,
            body=json.dumps(
                {
                    "error": {
                        "metadata": {
                            "headers": {
                                "X-RateLimit-Reset": str(reset_timestamp),
                            }
                        }
                    }
                }
            ),
        )
        response_payload = {
            "choices": [{"message": {"content": "working reply"}}],
        }

        with patch(
            "devs_utilities.openrouter.post_json",
            side_effect=[error, response_payload],
        ), patch("devs_utilities.openrouter.time.sleep") as mock_sleep, patch(
            "devs_utilities.openrouter.time.time",
            return_value=1_700_000_000,
        ):
            client.create_raw_completion([{"role": "user", "content": "hello"}])

        mock_sleep.assert_called_once_with(30.0)

    def test_create_raw_completion_does_not_retry_non_retryable_http_error(self) -> None:
        client = OpenRouterClient(
            OpenRouterConfig(
                api_key="token",
                base_url="https://openrouter.example/api/v1/chat/completions",
                model="openrouter/free",
                retry_attempts=4,
            )
        )
        error = HttpRequestError(
            url="https://openrouter.example/api/v1/chat/completions",
            message="HTTP 400",
            status_code=400,
        )

        with patch(
            "devs_utilities.openrouter.post_json",
            side_effect=error,
        ) as mock_post_json, patch("devs_utilities.openrouter.time.sleep") as mock_sleep:
            with self.assertRaises(OpenRouterError):
                client.create_raw_completion([{"role": "user", "content": "hello"}])

        self.assertEqual(mock_post_json.call_count, 1)
        mock_sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
