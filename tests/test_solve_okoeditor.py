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

from solve_okoeditor import (
    AgentState,
    evaluate_and_maybe_submit_done,
    execute_tool_call,
    find_targets,
    is_retryable_openrouter_error,
    recover_plaintext_tool_calls,
    recover_from_non_tool_response,
    submit_done_from_ready_state,
)
from devs_utilities.openrouter import ToolCall


class StubVerifyClient:
    def __init__(self) -> None:
        self.done_calls = 0

    def done(self) -> dict[str, str]:
        self.done_calls += 1
        return {"flag": "{{FLG:OKO}}"}


class RecoverFromNonToolResponseTests(unittest.TestCase):
    def test_retryable_openrouter_error_detects_transient_provider_failure(self) -> None:
        self.assertTrue(is_retryable_openrouter_error(RuntimeError("{'message': 'Internal Server Error', 'code': 500}")))
        self.assertFalse(is_retryable_openrouter_error(RuntimeError("validation failed")))

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

    def test_recover_plaintext_tool_calls_parses_tool_dump(self) -> None:
        recovered = recover_plaintext_tool_calls(
            'TOOLCALL>[{"name":"submit_done","arguments":{}}]',
            step=2,
        )

        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0].name, "submit_done")

    def test_recover_requests_retry_for_unparseable_plaintext_tool_call(self) -> None:
        messages: list[dict[str, object]] = []
        state = AgentState()

        finished = recover_from_non_tool_response(
            step=2,
            messages=messages,
            response_text='CALL>[{"name":"evaluate_state","arguments":{}</TOOLCALL>',
            web_client=object(),
            verify_client=object(),
            state=state,
            show_tool_results=False,
        )

        self.assertFalse(finished)
        self.assertEqual(messages[-1]["role"], "user")
        self.assertIn("malformed tool call", str(messages[-1]["content"]))

    def test_recover_executes_parseable_plaintext_submit_done(self) -> None:
        messages: list[dict[str, object]] = []
        state = AgentState()

        def fake_execute_tool_call(*args, **kwargs):  # type: ignore[no-untyped-def]
            tool_call = args[0]
            state.completed = True
            return {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.name,
                "content": json.dumps({"code": 0, "message": "done"}),
            }

        from unittest.mock import patch

        with patch("solve_okoeditor.execute_tool_call", side_effect=fake_execute_tool_call):
            finished = recover_from_non_tool_response(
                step=8,
                messages=messages,
                response_text='OLCALL>[{"name":"submit_done","arguments":{}}]',
                web_client=object(),
                verify_client=object(),
                state=state,
                show_tool_results=False,
            )

        self.assertTrue(finished)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["name"], "submit_done")

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

    def test_recover_finishes_when_submit_done_succeeds_without_flag(self) -> None:
        messages: list[dict[str, object]] = []
        state = AgentState()

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
            state.completed = True
            state.final_response = {"code": 0, "message": "done"}
            return {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.name,
                "content": json.dumps({"code": 0, "message": "done"}),
            }

        from unittest.mock import patch

        with patch("solve_okoeditor.execute_tool_call", side_effect=fake_execute_tool_call):
            finished = recover_from_non_tool_response(
                step=5,
                messages=messages,
                response_text="Task completed.",
                web_client=ReadyWebClient(),
                verify_client=object(),
                state=state,
                show_tool_results=False,
            )

        self.assertTrue(finished)
        self.assertTrue(state.completed)

    def test_evaluate_and_maybe_submit_done_returns_false_when_not_ready(self) -> None:
        messages: list[dict[str, object]] = []
        state = AgentState()

        def fake_execute_tool_call(*args, **kwargs):  # type: ignore[no-untyped-def]
            tool_call = args[0]
            return {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.name,
                "content": json.dumps({"ready_for_done": False if tool_call.name == "evaluate_state" else {}}),
            }

        from unittest.mock import patch

        with patch("solve_okoeditor.execute_tool_call", side_effect=fake_execute_tool_call):
            finished = evaluate_and_maybe_submit_done(
                step=6,
                suffix="update_record",
                messages=messages,
                web_client=object(),
                verify_client=object(),
                state=state,
                show_tool_results=False,
            )

        self.assertFalse(finished)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["name"], "evaluate_state")

    def test_submit_done_from_ready_state_appends_done_tool_message(self) -> None:
        messages: list[dict[str, object]] = []
        state = AgentState()

        def fake_execute_tool_call(*args, **kwargs):  # type: ignore[no-untyped-def]
            tool_call = args[0]
            state.completed = True
            return {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.name,
                "content": json.dumps({"code": 0, "message": "done"}),
            }

        from unittest.mock import patch

        with patch("solve_okoeditor.execute_tool_call", side_effect=fake_execute_tool_call):
            finished = submit_done_from_ready_state(
                step=7,
                suffix="evaluate_state",
                messages=messages,
                web_client=object(),
                verify_client=object(),
                state=state,
                show_tool_results=False,
            )

        self.assertTrue(finished)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["name"], "submit_done")

    def test_execute_tool_call_skips_sync_for_invalid_update_identifier(self) -> None:
        state = AgentState()

        class VerifyClient:
            def update(self, **_kwargs):  # type: ignore[no-untyped-def]
                raise RuntimeError("Record id must be a 32-character hex string.")

        from unittest.mock import patch

        with (
            patch("solve_okoeditor.wait_for_visible_updates") as wait_mock,
            patch("solve_okoeditor.write_json"),
        ):
            execute_tool_call(
                ToolCall(
                    id="call-1",
                    name="update_record",
                    arguments={"page": "incydenty", "id": "bad-id", "title": "MOVE01 test"},
                ),
                web_client=object(),
                verify_client=VerifyClient(),
                state=state,
                show_tool_results=False,
            )

        assert isinstance(state.last_update_response, dict)
        self.assertFalse(state.last_update_response["ok"])
        self.assertIn("32-character hex", state.last_update_response["error"])
        wait_mock.assert_not_called()

    def test_find_targets_falls_back_to_incident_id_for_skolwin_task(self) -> None:
        class FallbackTaskWebClient:
            def list_records(self, page: str) -> dict[str, object]:
                if page == "incydenty":
                    return {
                        "entries": [
                            {
                                "id": "abc123abc123abc123abc123abc123ab",
                                "title": "MOVE03 Problem w rejonie Skolwin",
                                "summary": "Ruch przy Skolwin.",
                            }
                        ]
                    }
                if page == "zadania":
                    return {"entries": []}
                return {
                    "entries": [
                        {
                            "id": "note123note123note123note123note12",
                            "title": "Metody kodowania incydentów",
                            "summary": "",
                        }
                    ]
                }

            def get_record(self, page: str, record_id: str) -> dict[str, object]:
                self.last_get_record = (page, record_id)
                return {
                    "id": record_id,
                    "page": page,
                    "title": "Zbadanie nagrań z okolic Skolwina",
                    "content": "Widziano tam zwierzęta, na przykład bobry.",
                    "status": "pending",
                }

        client = FallbackTaskWebClient()

        targets = find_targets(client)

        self.assertEqual(targets["skolwin_task_id"], "abc123abc123abc123abc123abc123ab")
        self.assertEqual(client.last_get_record, ("zadania", "abc123abc123abc123abc123abc123ab"))


if __name__ == "__main__":
    unittest.main()
