from __future__ import annotations

from pathlib import Path
import unittest

from timetravel.frontend_agent import merge_observed_ui, should_click_orb
from timetravel.frontend_ui import normalize_ui_state, required_ui_selectors
from timetravel.models import DesiredUiState, MissionStatus, ObservedUiState


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "timetravel_preview_fixture.html"
FRONTEND_AGENT_PATH = Path(__file__).resolve().parents[1] / "timetravel" / "frontend_agent.py"


class TimetravelFrontendTests(unittest.TestCase):
    def test_preview_fixture_contains_required_selectors(self) -> None:
        html = FIXTURE_PATH.read_text(encoding="utf-8")

        for selector in required_ui_selectors():
            self.assertIn(selector.replace("#", 'id="'), html)

    def test_normalize_ui_state_maps_browser_snapshot(self) -> None:
        raw_state = {
            "PTA": True,
            "PTB": False,
            "PWR": 28,
            "mode": "standby",
            "batteryStatus": "3/3",
            "fluxDensity": 100,
            "condition": "stable",
            "orbClickable": False,
            "flagText": "{FLG:test}",
            "lastConsumedJumpId": 7,
        }

        normalized = normalize_ui_state(raw_state)

        self.assertEqual(normalized.PWR, 28)
        self.assertEqual(normalized.battery_level, 3)
        self.assertEqual(normalized.flag, "{FLG:test}")
        self.assertEqual(normalized.last_consumed_jump_id, 7)

    def test_frontend_agent_does_not_use_persistent_user_profile(self) -> None:
        source = FRONTEND_AGENT_PATH.read_text(encoding="utf-8")

        self.assertNotIn("launch_persistent_context", source)
        self.assertNotIn("user_data_dir", source)
        self.assertNotIn("browser-profile-copy", source)
        self.assertIn("storage_state", source)
        self.assertIn("frontend_storage_state.json", source)

    def test_merge_observed_ui_preserves_last_consumed_jump_id(self) -> None:
        snapshot = ObservedUiState(
            mode="active",
            PTA=True,
            PTB=False,
            PWR=28,
            battery_status="2/3",
            flux_density=100,
            condition="stable",
            orb_clickable=True,
            last_consumed_jump_id=0,
        )

        merged = merge_observed_ui(snapshot, last_consumed_jump_id=7)

        self.assertEqual(merged.last_consumed_jump_id, 7)
        self.assertEqual(merged.mode, "active")

    def test_should_click_orb_requires_backend_ready_status(self) -> None:
        desired = DesiredUiState(PTA=True, PTB=False, PWR=28, mode="active", jump_request_id=2)
        observed = ObservedUiState(
            mode="active",
            PTA=True,
            PTB=False,
            PWR=28,
            battery_status="3/3",
            flux_density=100,
            condition="stable",
            orb_clickable=True,
            last_consumed_jump_id=1,
        )

        self.assertFalse(
            should_click_orb(
                desired,
                observed,
                mission_status=MissionStatus.WAITING_UI.value,
            )
        )
        self.assertTrue(
            should_click_orb(
                desired,
                observed,
                mission_status=MissionStatus.READY_FOR_JUMP.value,
            )
        )


if __name__ == "__main__":
    unittest.main()
