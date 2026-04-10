from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
MAILBOX_DIR = REPO_ROOT / "mailbox"
for candidate in (str(REPO_ROOT), str(MAILBOX_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from devs_utilities.http import HttpRequestError
from solve_mailbox import AppConfig, ToolCall, ZmailClient, execute_tool_call, normalize_message_ids


class NormalizeMessageIdsTests(unittest.TestCase):
    def test_keeps_plain_message_id_string(self) -> None:
        self.assertEqual(normalize_message_ids("6624add090a5cb06f5c192653b5a243c"), "6624add090a5cb06f5c192653b5a243c")

    def test_parses_stringified_json_array(self) -> None:
        self.assertEqual(
            normalize_message_ids('["6624add090a5cb06f5c192653b5a243c"]'),
            ["6624add090a5cb06f5c192653b5a243c"],
        )

    def test_converts_numeric_strings_to_ints(self) -> None:
        self.assertEqual(normalize_message_ids(["15", "27"]), [15, 27])

    def test_zmail_client_returns_structured_http_error_payload(self) -> None:
        client = ZmailClient(
            AppConfig(
                ag3nts_api_key="apikey",
                verify_url="https://example.com/verify",
                zmail_url="https://example.com/api/zmail",
                openrouter_api_key="llm",
                openrouter_url="https://example.com/openrouter",
                model="model",
                api_timeout_seconds=30,
                openrouter_timeout_seconds=30,
                site_url=None,
                site_name=None,
                max_steps=4,
                show_tool_results=False,
            )
        )
        error = HttpRequestError(
            url="https://example.com/api/zmail",
            message="HTTP 404 for https://example.com/api/zmail: not found",
            status_code=404,
            body='{"ok":false,"error":"No messages found for provided ids."}',
        )

        with patch("solve_mailbox.post_json", side_effect=error):
            result = client.get_messages(ids=["missing-id"])

        self.assertFalse(result["ok"])
        self.assertEqual(result["http_status"], 404)
        self.assertIn("stale", result["agent_note"])

    def test_execute_tool_call_returns_error_payload_when_arguments_are_missing(self) -> None:
        tool_message = execute_tool_call(
            ToolCall(id="call_1", name="get_thread", arguments={}),
            {"get_thread": lambda args: int(args["thread_id"])},
            show_tool_results=False,
        )

        self.assertEqual(tool_message["role"], "tool")
        self.assertIn("thread_id", tool_message["content"])

    def test_get_thread_messages_refreshes_thread_and_fetches_current_messages(self) -> None:
        client = ZmailClient(
            AppConfig(
                ag3nts_api_key="apikey",
                verify_url="https://example.com/verify",
                zmail_url="https://example.com/api/zmail",
                openrouter_api_key="llm",
                openrouter_url="https://example.com/openrouter",
                model="model",
                api_timeout_seconds=30,
                openrouter_timeout_seconds=30,
                site_url=None,
                site_name=None,
                max_steps=4,
                show_tool_results=False,
            )
        )

        with patch.object(client, "get_thread", return_value={"items": [{"messageID": "abc"}, {"messageID": "def"}]}):
            with patch.object(client, "get_messages", return_value={"ok": True, "items": [{"messageID": "abc"}]}) as get_messages_mock:
                result = client.get_thread_messages(thread_id=123)

        get_messages_mock.assert_called_once_with(ids=["abc", "def"])
        self.assertTrue(result["ok"])
        self.assertEqual(result["message_ids"], ["abc", "def"])


if __name__ == "__main__":
    unittest.main()
