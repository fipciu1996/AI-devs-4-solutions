"""Unit tests for the shellaccess OpenRouter agent."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from devs_utilities.http import HttpRequestError

from shellaccess.solve_shellaccess import (
    AgentState,
    AppConfig,
    ShellAccessError,
    ToolCall,
    VerifyClient,
    build_answer_command,
    execute_tool_call,
    build_tool_handlers,
    is_retryable_completion_error,
    split_shell_segments,
    validate_remote_command,
)


def make_config() -> AppConfig:
    return AppConfig(
        api_key="test-key",
        verify_url="https://example.com/verify",
        openrouter_api_key="llm-key",
        openrouter_url="https://example.com/chat",
        model="openai/gpt-4.1-mini",
        api_timeout_seconds=30,
        openrouter_timeout_seconds=120,
        site_url=None,
        site_name="AI Devs 4 - shellaccess",
        max_steps=8,
        show_tool_results=False,
    )


class ShellAccessHelpersTests(unittest.TestCase):
    def test_retryable_completion_error_detects_malformed_tool_call(self) -> None:
        self.assertTrue(
            is_retryable_completion_error(
                RuntimeError("OpenRouter tool call arguments for search_time_logs are not valid JSON.")
            )
        )
        self.assertFalse(is_retryable_completion_error(RuntimeError("invalid answer")))

    def test_build_answer_command_preserves_unicode_and_format(self) -> None:
        command = build_answer_command(
            {
                "date": "2024-11-12",
                "city": "Grudziądz",
                "longitude": 18.968774,
                "latitude": 53.432303,
            }
        )

        self.assertEqual(
            command,
            "echo '{\"date\":\"2024-11-12\",\"city\":\"Grudziądz\","
            "\"longitude\":18.968774,\"latitude\":53.432303}'",
        )

    def test_validate_remote_command_allows_read_only_pipeline(self) -> None:
        command = "grep -n 'Rafa' /data/time_logs.csv | tail -n 40"
        self.assertEqual(validate_remote_command(command), command)

    def test_validate_remote_command_rejects_disallowed_binary(self) -> None:
        with self.assertRaises(ShellAccessError):
            validate_remote_command("rm -rf /data")

    def test_validate_remote_command_rejects_redirection(self) -> None:
        with self.assertRaises(ShellAccessError):
            validate_remote_command("echo test > /tmp/out.txt")

    def test_validate_remote_command_accepts_cut_delimiter_and_jq_filter_pipes(self) -> None:
        cut_command = "tail -n 1 /data/time_logs.csv | cut -d ';' -f 1"
        jq_command = "jq -r '.[] | select(.location_id==219) | .name' /data/locations.json"

        self.assertEqual(validate_remote_command(cut_command), cut_command)
        self.assertEqual(validate_remote_command(jq_command), jq_command)

    def test_split_shell_segments_ignores_operators_inside_quotes(self) -> None:
        command = "jq -r '.[] | select(.location_id==219) | .name' /data/locations.json"
        self.assertEqual(split_shell_segments(command), [command])


class VerifyClientTests(unittest.TestCase):
    @patch("shellaccess.solve_shellaccess.submit_task_answer")
    def test_execute_command_returns_parsed_http_error_payload(self, submit_mock) -> None:
        submit_mock.side_effect = HttpRequestError(
            url="https://example.com/verify",
            message="HTTP 400",
            status_code=400,
            body='{"code":-860,"message":"Output too large."}',
        )
        client = VerifyClient(make_config())

        result = client.execute_command("grep -n 'Rafa' /data/time_logs.csv | head -n 100")

        self.assertEqual(result["code"], -860)
        self.assertEqual(result["message"], "Output too large.")
        self.assertEqual(result["http_status"], 400)
        self.assertIn("command", result)


class ToolHandlerTests(unittest.TestCase):
    @patch("shellaccess.solve_shellaccess.write_json")
    def test_submit_answer_updates_state_and_extracts_flag(self, write_json_mock) -> None:
        del write_json_mock

        class FakeVerifyClient:
            def submit_answer(self, **_kwargs):
                return {"code": 0, "message": "{FLG:TEST}"}

            def execute_command(self, _command: str):
                return {"code": 100, "message": "Command executed.", "output": "/data"}

        state = AgentState()
        handlers = build_tool_handlers(verify_client=FakeVerifyClient(), state=state)

        result = handlers["submit_answer"](
            {
                "date": "2024-11-12",
                "city": "Grudziądz",
                "longitude": 18.968774,
                "latitude": 53.432303,
            }
        )

        self.assertEqual(result["message"], "{FLG:TEST}")
        self.assertEqual(state.final_flag, "{FLG:TEST}")
        self.assertEqual(state.last_answer["city"], "Grudziądz")

    def test_execute_tool_call_returns_error_payload_for_invalid_command(self) -> None:
        tool_call = ToolCall(
            id="call_1",
            name="execute_remote_command",
            arguments={"command": "jq '", "reason": "broken quote"},
        )

        message = execute_tool_call(
            tool_call,
            {"execute_remote_command": lambda args: validate_remote_command(args["command"])},
            show_tool_results=False,
        )

        self.assertEqual(message["role"], "tool")
        self.assertIn("Invalid shell syntax", message["content"])

    def test_execute_tool_call_returns_error_payload_when_required_argument_is_missing(self) -> None:
        state = AgentState()

        class FakeVerifyClient:
            def submit_answer(self, **_kwargs):
                return {"code": 0, "message": "ok"}

            def execute_command(self, _command: str):
                return {"code": 0, "message": "ok"}

        handlers = build_tool_handlers(verify_client=FakeVerifyClient(), state=state)
        tool_call = ToolCall(id="call_2", name="get_location_name", arguments={})

        message = execute_tool_call(tool_call, handlers, show_tool_results=False)

        self.assertEqual(message["role"], "tool")
        self.assertIn("location_id is required", message["content"])


if __name__ == "__main__":
    unittest.main()
