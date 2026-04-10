from __future__ import annotations

from pathlib import Path
import unittest
from uuid import uuid4

from timetravel.models import DesiredUiState, MissionPhase, MissionStatus, ObservedUiState
from timetravel.store import SharedStateStore


class TimetravelStoreTests(unittest.TestCase):
    def test_store_persists_mission_and_ui_state(self) -> None:
        db_path = Path(__file__).resolve().parent / f"timetravel-store-{uuid4().hex}.sqlite3"
        try:
            store = SharedStateStore(db_path)

            store.set_mission_state(
                phase=MissionPhase.RETURN_PRESENT,
                status=MissionStatus.WAITING_UI,
                present_date="2026-04-10",
            )
            store.set_desired_ui(
                DesiredUiState(
                    PTA=True,
                    PTB=False,
                    PWR=28,
                    mode="standby",
                    jump_request_id=2,
                )
            )
            store.set_observed_ui(
                ObservedUiState(
                    PTA=True,
                    PTB=False,
                    PWR=28,
                    mode="standby",
                    battery_status="3/3",
                )
            )
            store.set_backend_snapshot({"mode": "standby"})
            store.record_event("test", "INFO", "hello")

            mission = store.get_mission_state()
            desired = store.get_desired_ui()
            observed = store.get_observed_ui()
            snapshot = store.get_backend_snapshot()

            self.assertEqual(mission["phase"], MissionPhase.RETURN_PRESENT.value)
            self.assertEqual(mission["status"], MissionStatus.WAITING_UI.value)
            self.assertEqual(desired.jump_request_id, 2)
            self.assertEqual(observed.battery_level, 3)
            self.assertEqual(snapshot["mode"], "standby")
        finally:
            if db_path.exists():
                db_path.unlink()

    def test_bump_jump_request_increments_counter(self) -> None:
        db_path = Path(__file__).resolve().parent / f"timetravel-store-{uuid4().hex}.sqlite3"
        try:
            store = SharedStateStore(db_path)

            self.assertEqual(store.bump_jump_request(), 1)
            self.assertEqual(store.bump_jump_request(), 2)
        finally:
            if db_path.exists():
                db_path.unlink()


if __name__ == "__main__":
    unittest.main()
