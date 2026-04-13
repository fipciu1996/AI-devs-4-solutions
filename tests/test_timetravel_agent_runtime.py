from __future__ import annotations

import importlib
import os
import unittest
from unittest.mock import patch

import timetravel.openrouter_backend as openrouter_backend_module
import timetravel.openrouter_frontend as openrouter_frontend_module
from timetravel.backend_agent import _resolve_agent_runtime as resolve_backend_runtime
from timetravel.frontend_agent import _resolve_agent_runtime as resolve_frontend_runtime
from timetravel.openrouter_backend import OPENROUTER_TOOLS as BACKEND_TOOLS
from timetravel.openrouter_frontend import OPENROUTER_TOOLS as FRONTEND_TOOLS


class TimetravelAgentRuntimeTests(unittest.TestCase):
    def test_backend_runtime_auto_prefers_openrouter_when_llm_config_exists(self) -> None:
        with (
            patch("timetravel.backend_agent.get_llm_api_key", return_value="llm-key"),
            patch("timetravel.backend_agent.get_llm_base_url", return_value="https://openrouter.example/api/v1/chat/completions"),
        ):
            self.assertEqual(resolve_backend_runtime("auto"), "openrouter")

    def test_frontend_runtime_auto_falls_back_without_llm_config(self) -> None:
        with (
            patch("timetravel.frontend_agent.get_llm_api_key", return_value=""),
            patch("timetravel.frontend_agent.get_llm_base_url", return_value=""),
        ):
            self.assertEqual(resolve_frontend_runtime("auto"), "deterministic")

    def test_backend_openrouter_tools_cover_phase_lifecycle(self) -> None:
        tool_names = [tool["function"]["name"] for tool in BACKEND_TOOLS]

        self.assertEqual(
            tool_names,
            [
                "inspect_state",
                "prepare_current_phase",
                "configure_current_phase",
                "wait_for_required_internal_mode",
                "wait_for_ui_standby_alignment",
                "arm_current_phase",
                "wait_for_ui_ready",
                "publish_jump_request",
                "wait_for_phase_completion",
                "advance_phase",
            ],
        )

    def test_frontend_openrouter_tools_cover_ui_lifecycle(self) -> None:
        tool_names = [tool["function"]["name"] for tool in FRONTEND_TOOLS]

        self.assertEqual(
            tool_names,
            [
                "inspect_state",
                "ensure_preview_ready",
                "read_ui_state",
                "apply_desired_ui_step",
                "consume_jump_if_ready",
                "persist_session",
            ],
        )

    def test_backend_default_model_supports_legacy_timetavel_alias(self) -> None:
        with patch.dict(os.environ, {"TIMETAVEL_MODEL": "legacy-typo-model"}, clear=True):
            reloaded = importlib.reload(openrouter_backend_module)
            self.assertEqual(reloaded.DEFAULT_MODEL, "legacy-typo-model")
        importlib.reload(openrouter_backend_module)

    def test_frontend_default_model_supports_legacy_timetavel_alias(self) -> None:
        with patch.dict(os.environ, {"TIMETAVEL_MODEL": "legacy-typo-model"}, clear=True):
            reloaded = importlib.reload(openrouter_frontend_module)
            self.assertEqual(reloaded.DEFAULT_MODEL, "legacy-typo-model")
        importlib.reload(openrouter_frontend_module)


if __name__ == "__main__":
    unittest.main()
