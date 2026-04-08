from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OKOEDITOR_DIR = REPO_ROOT / "okoeditor"
for candidate in (str(REPO_ROOT), str(OKOEDITOR_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from solve_okoeditor import AgentState, recover_from_non_tool_response


class StubVerifyClient:
    def __init__(self) -> None:
        self.done_calls = 0

    def done(self) -> dict[str, str]:
        self.done_calls += 1
        return {"flag": "{{FLG:OKO}}"}


class RecoverFromNonToolResponseTests(unittest.TestCase):
    def test_recover_auto_submits_when_state_is_ready(self) -> None:
        messages: list[dict[str, object]] = []
        state = AgentState()
        verify_client = StubVerifyClient()

        class ReadyWebClient:
            pass

        def fake_execute_tool_call(*args, **kwargs):  # type: ignore[no-untyped-def]
            tool_call = args[0]
            if tool_call.name == "evaluate_state":
                return {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.name,
                    "content": json.dumps({"ready_for_done": True}),
                }
            state.final_flag = "{{FLG:OKO}}"
            verify_client.done_calls += 1
            return {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.name,
                "content": json.dumps({"flag": "{{FLG:OKO}}"}),
            }

        from unittest.mock import patch

        with patch("solve_okoeditor.execute_tool_call", side_effect=fake_execute_tool_call):
            finished = recover_from_non_tool_response(
                step=3,
                messages=messages,
                response_text="Task completed.",
                web_client=ReadyWebClient(),
                verify_client=verify_client,
                state=state,
                show_tool_results=False,
            )

        self.assertTrue(finished)
        self.assertEqual(verify_client.done_calls, 1)
        self.assertEqual(len(messages), 2)

    def test_recover_adds_tool_only_reminder_when_state_is_not_ready(self) -> None:
        messages: list[dict[str, object]] = []
        state = AgentState()

        class IdleVerifyClient:
            def done(self) -> dict[str, str]:
                raise AssertionError("done should not be called when state is not ready")

        class ReadyWebClient:
            pass

        def fake_execute_tool_call(*args, **kwargs):  # type: ignore[no-untyped-def]
            tool_call = args[0]
            return {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.name,
                "content": json.dumps({"ready_for_done": False}),
            }

        from unittest.mock import patch

        with patch("solve_okoeditor.execute_tool_call", side_effect=fake_execute_tool_call):
            finished = recover_from_non_tool_response(
                step=4,
                messages=messages,
                response_text="All set.",
                web_client=ReadyWebClient(),
                verify_client=IdleVerifyClient(),
                state=state,
                show_tool_results=False,
            )

        self.assertFalse(finished)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[-1]["role"], "user")
        self.assertIn("Use tools only.", str(messages[-1]["content"]))


if __name__ == "__main__":
    unittest.main()
