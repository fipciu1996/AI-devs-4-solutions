from __future__ import annotations

import unittest

from devs_utilities.openrouter import (
    OpenRouterError,
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


if __name__ == "__main__":
    unittest.main()
