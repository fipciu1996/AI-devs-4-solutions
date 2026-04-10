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
from analyze_attachments_openrouter import (
    DEFAULT_SITE_NAME,
    collect_targets,
    execute_sendit_analyze_tool_call,
    is_retryable_analysis_error,
    output_path_for,
)


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

    def test_retryable_analysis_error_detects_gateway_abort(self) -> None:
        self.assertTrue(is_retryable_analysis_error(RuntimeError("{'message': 'The operation was aborted', 'code': 504}")))
        self.assertFalse(is_retryable_analysis_error(RuntimeError("invalid prompt")))

    def test_output_path_can_be_reused_as_cache_signal(self) -> None:
        root = REPO_ROOT / "tests" / "_tmp_sendit_analyze_cache"
        root.mkdir(parents=True, exist_ok=True)
        attachment = root / "note.md"
        output = root / "note.analysis.md"
        try:
            attachment.write_text("hello", encoding="utf-8")
            target = collect_targets(root, "*")[0]
            output = output_path_for(target, root, "text")
            output.write_text("cached", encoding="utf-8")

            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)
        finally:
            for path in sorted(root.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink(missing_ok=True)
                else:
                    path.rmdir()
            root.rmdir()


if __name__ == "__main__":
    unittest.main()
