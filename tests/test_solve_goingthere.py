from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from devs_utilities.ag3nts import AG3NTS_VERIFY_URL, build_ag3nts_api_url
from devs_utilities.http import HttpRequestError
from devs_utilities.openrouter import ChatCompletionResult, ToolCall
from goingthere.solve_goingthere import (
    GameLostError,
    GAME_AGENT_SYSTEM_PROMPT_PATH,
    GameState,
    Position,
    RadarTrap,
    ToolExecutionResult,
    build_game_agent_system_prompt,
    build_game_agent_tools,
    compute_disarm_hash,
    ensure_safe_to_move,
    execute_game_agent_tool,
    extract_trap_data,
    looks_like_clear_scan,
    parse_game_state,
    recover_plaintext_tool_calls,
    run_single_game_with_tool_calling,
    should_apply_hub_spacing,
    state_snapshot,
)


class SolveGoingThereTests(unittest.TestCase):
    def test_looks_like_clear_scan_accepts_noisy_variants(self) -> None:
        self.assertTrue(looks_like_clear_scan('"IT\'S   ClEAr!"'))
        self.assertTrue(looks_like_clear_scan('"Its     cleeeeeeeear"'))

    def test_extract_trap_data_handles_json_like_noise(self) -> None:
        trap = extract_trap_data(
            'broken >>> {"frequency": 145, "detectionCode":"alpha-123_xyz"} <<<',
        )

        self.assertEqual(
            trap,
            RadarTrap(frequency=145, detection_code="alpha-123_xyz"),
        )

    def test_extract_trap_data_handles_corrupted_field_names(self) -> None:
        trap = extract_trap_data(
            '{ "frepuency": 109, "bata": { "weap0nType": "air-to-air missile", '
            '"betecti0nC0be": "ZRpRkw" } }',
        )

        self.assertEqual(trap, RadarTrap(frequency=109, detection_code="ZRpRkw"))

    def test_extract_trap_data_handles_backtick_wrapped_code(self) -> None:
        trap = extract_trap_data(
            '{ "frEpueNCY": 154, "bATA": { "BEtECTI0Nc0BE": `FUXrvB", '
            '"WeaP0NtyPe": "air-to-air missile` } }',
        )

        self.assertEqual(trap, RadarTrap(frequency=154, detection_code="FUXrvB"))

    def test_compute_disarm_hash_matches_sha1_requirement(self) -> None:
        self.assertEqual(
            compute_disarm_hash("abc"),
            "718f0de3b87d0ce303c42a99ad075ae02cbdc3ce",
        )

    def test_parse_game_state_reuses_base_when_move_payload_omits_it(self) -> None:
        state = parse_game_state(
            {
                "player": {"row": 3, "col": 2},
                "currentColumn": {"column": 2},
            },
            current_base=Position(row=1, col=12),
        )

        self.assertEqual(state.player, Position(row=3, col=2))
        self.assertEqual(state.base, Position(row=1, col=12))

    def test_state_snapshot_reports_valid_commands_and_remaining_moves(self) -> None:
        snapshot = state_snapshot(
            GameState(
                player=Position(row=1, col=4),
                base=Position(row=3, col=12),
                current_stone_row=2,
            )
        )

        self.assertEqual(snapshot["valid_commands"], ["go", "right"])
        self.assertEqual(snapshot["remaining_move_commands"], 8)
        self.assertEqual(snapshot["current_stone_row"], 2)

    def test_build_game_agent_tools_exposes_expected_tool_names(self) -> None:
        tool_names = [tool["function"]["name"] for tool in build_game_agent_tools()]

        self.assertEqual(
            tool_names,
            [
                "start_game",
                "scan_frequency",
                "disarm_trap",
                "get_radio_hint",
                "move_rocket",
            ],
        )

    def test_system_prompt_requires_per_game_reasoning(self) -> None:
        prompt = build_game_agent_system_prompt()
        file_prompt = GAME_AGENT_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()

        self.assertEqual(prompt, file_prompt)
        self.assertEqual(GAME_AGENT_SYSTEM_PROMPT_PATH.name, "game_agent_system_prompt.txt")
        self.assertEqual(GAME_AGENT_SYSTEM_PROMPT_PATH.parent.name, "goingthere")
        self.assertIn("Every game is a brand-new random board.", prompt)
        self.assertIn("Never rely on previous games, previous traces", prompt)
        self.assertIn("Use only the current game's tool outputs.", prompt)
        self.assertIn(
            "Treat each radio hint as a separate natural-language statement",
            prompt,
        )
        self.assertIn("get_radio_hint exactly once for that turn", prompt)
        self.assertIn("If two directions are safe, prefer the move", prompt)

    def test_should_apply_hub_spacing_only_for_hub_host(self) -> None:
        self.assertTrue(should_apply_hub_spacing(build_ag3nts_api_url("getmessage")))
        self.assertTrue(should_apply_hub_spacing(AG3NTS_VERIFY_URL))
        self.assertFalse(
            should_apply_hub_spacing("https://openrouter.ai/api/v1/chat/completions")
        )
        self.assertFalse(should_apply_hub_spacing(None))

    def test_recover_plaintext_tool_calls_parses_olcall_dump(self) -> None:
        recovered = recover_plaintext_tool_calls(
            'OLCALL>[{"name":"get_radio_hint","arguments":{}}]>',
            step=4,
        )

        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0].name, "get_radio_hint")
        self.assertEqual(recovered[0].arguments, {})

    def test_execute_game_agent_tool_rejects_move_before_start(self) -> None:
        result = execute_game_agent_tool(
            name="move_rocket",
            arguments={"command": "go"},
            api_key="test-key",
            trace=[],
            game_index=1,
            current_state=None,
        )

        self.assertEqual(result.message["status"], "error")
        self.assertIn("start_game", result.message["error"])

    def test_execute_game_agent_tool_starts_game_and_returns_state(self) -> None:
        with patch(
            "goingthere.solve_goingthere.submit_command",
            return_value={
                "code": 110,
                "message": "New game started.",
                "player": {"row": 2, "col": 1},
                "base": {"row": 3, "col": 12},
                "currentColumn": {
                    "column": 1,
                    "yourRow": 2,
                    "stoneRow": 1,
                    "freeRows": [2, 3],
                },
            },
        ), patch("goingthere.solve_goingthere.append_trace_event"):
            result = execute_game_agent_tool(
                name="start_game",
                arguments={},
                api_key="test-key",
                trace=[],
                game_index=1,
                current_state=None,
            )

        self.assertEqual(result.message["status"], "started")
        self.assertIsNotNone(result.state)
        self.assertEqual(result.state.player, Position(row=2, col=1))
        self.assertEqual(
            result.message["state"]["valid_commands"],
            ["left", "go", "right"],
        )

    def test_execute_game_agent_tool_rejects_invalid_edge_move(self) -> None:
        state = GameState(
            player=Position(row=1, col=2),
            base=Position(row=3, col=12),
        )

        with patch("goingthere.solve_goingthere.submit_command") as mocked_submit:
            result = execute_game_agent_tool(
                name="move_rocket",
                arguments={"command": "left"},
                api_key="test-key",
                trace=[],
                game_index=1,
                current_state=state,
            )

        self.assertEqual(result.message["status"], "error")
        self.assertEqual(result.message["valid_commands"], ["go", "right"])
        mocked_submit.assert_not_called()

    def test_execute_game_agent_tool_returns_hint_payload(self) -> None:
        with patch(
            "goingthere.solve_goingthere.request_hint",
            return_value="The obstacle is centered on your flight line.",
        ), patch("goingthere.solve_goingthere.append_trace_event"):
            result = execute_game_agent_tool(
                name="get_radio_hint",
                arguments={},
                api_key="test-key",
                trace=[],
                game_index=1,
                current_state=GameState(
                    player=Position(row=2, col=3),
                    base=Position(row=1, col=12),
                ),
            )

        self.assertEqual(result.message["status"], "ok")
        self.assertIn("flight line", result.message["hint"])

    def test_run_single_game_with_tool_calling_recovers_plaintext_tool_call(self) -> None:
        started_state = GameState(
            player=Position(row=2, col=1),
            base=Position(row=1, col=12),
        )

        class FakeClient:
            def __init__(self) -> None:
                self.completions = [
                    ChatCompletionResult(
                        content='OLCALL>[{"name":"start_game","arguments":{}}]>',
                        tool_calls=[],
                    ),
                    ChatCompletionResult(
                        content=None,
                        tool_calls=[
                            ToolCall(
                                id="tool_2",
                                name="move_rocket",
                                arguments={"command": "go"},
                            )
                        ],
                    ),
                ]

            def create_completion(self, *_args, **_kwargs) -> ChatCompletionResult:
                return self.completions.pop(0)

        with patch(
            "goingthere.solve_goingthere.build_game_agent_client",
            return_value=FakeClient(),
        ), patch(
            "goingthere.solve_goingthere.execute_game_agent_tool",
            side_effect=[
                ToolExecutionResult(
                    message={"status": "started"},
                    state=started_state,
                ),
                ToolExecutionResult(
                    message={"status": "flag"},
                    final_response={"code": 0, "message": "{FLG:TEST}"},
                ),
            ],
        ) as mocked_execute, patch("goingthere.solve_goingthere.append_trace_event"):
            final_response, _trace = run_single_game_with_tool_calling(
                "test-key",
                game_index=1,
            )

        self.assertEqual(final_response["message"], "{FLG:TEST}")
        self.assertEqual(
            [call.kwargs["name"] for call in mocked_execute.call_args_list],
            ["start_game", "move_rocket"],
        )

    def test_ensure_safe_to_move_turns_trap_crash_into_game_loss(self) -> None:
        trap = RadarTrap(frequency=411, detection_code="abc")
        crash_error = HttpRequestError(
            url=build_ag3nts_api_url("frequencyScanner"),
            message="boom",
            status_code=400,
            body=json.dumps(
                {
                    "code": -950,
                    "message": "The rocket was tracked and hit by a missile.",
                    "crashed": True,
                    "crashReason": "trap",
                    "crashMessage": "The rocket was tracked and hit by a missile.",
                }
            ),
        )

        with patch("goingthere.solve_goingthere.read_scanner", return_value=trap), patch(
            "goingthere.solve_goingthere.disarm_trap",
            side_effect=crash_error,
        ), patch("goingthere.solve_goingthere.append_trace_event"):
            with self.assertRaises(GameLostError) as context:
                ensure_safe_to_move("test-key", [], game_index=1)

        self.assertIn("tracked and hit by a missile", str(context.exception))


if __name__ == "__main__":
    unittest.main()
