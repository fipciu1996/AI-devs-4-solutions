from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SENDIT_DIR = REPO_ROOT / "sendit"
for candidate in (str(REPO_ROOT), str(SENDIT_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from devs_utilities.openrouter import ToolCall
from analyze_attachments_openrouter import DEFAULT_SITE_NAME, execute_sendit_analyze_tool_call


class SenditAnalyzeToolTests(unittest.TestCase):
    def test_default_site_name_uses_specific_solver_name(self) -> None:
        self.assertEqual(DEFAULT_SITE_NAME, "AI Devs 4 - sendit-analyze")

    def test_unknown_tool_is_reported_back_to_model(self) -> None:
        tool_message = execute_sendit_analyze_tool_call(
            ToolCall(id="call_1", name="get_attachment_payload_version_current", arguments={}),
            handlers={"get_analysis_target_context": lambda _args: {"ok": True}},
        )

        payload = json.loads(str(tool_message["content"]))
        self.assertFalse(payload["ok"])
        self.assertIn("Unknown sendit analyze tool call", payload["error"])


if __name__ == "__main__":
    unittest.main()
