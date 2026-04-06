from __future__ import annotations

import unittest

from devs_utilities.openrouter import (
    OpenRouterError,
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
        site_name = build_task_site_name("C:/repo/people/find-agent.py", task_name="findhim")

        self.assertEqual(site_name, "AI Devs 4 - findhim")

    def test_build_task_usage_output_path_uses_task_specific_name_when_needed(self) -> None:
        output_path = build_task_usage_output_path(
            "C:/repo/sensors/solve_sensors.py",
            task_name="evaluation",
        )

        self.assertEqual(output_path.as_posix(), "C:/repo/sensors/openrouter_usage_evaluation.json")


if __name__ == "__main__":
    unittest.main()
