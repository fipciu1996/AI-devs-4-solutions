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
from solve_mailbox import AppConfig, ZmailClient, normalize_message_ids


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


if __name__ == "__main__":
    unittest.main()
