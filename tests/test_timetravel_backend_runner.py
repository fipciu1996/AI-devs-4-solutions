from __future__ import annotations

from datetime import date
from pathlib import Path
import unittest

from timetravel.backend_agent import BackendMissionRunner
from timetravel.models import DeviceConfig, build_target_profiles, load_pwr_lookup
from timetravel.store import SharedStateStore


class _FakeApiClient:
    def __init__(self) -> None:
        self.configure_calls: list[tuple[str, object]] = []

    def configure(self, param: str, value: object) -> dict[str, object]:
        self.configure_calls.append((param, value))
        guidance = {
            "year": (
                "Dla tego rodzaju skoku podręczniki operatora sugerują zwykle "
                "dziewięćset jednostek. Mimo to zalecane jest obniżenie poziomu "
                "o siedemset jedenaście."
            ),
            "month": (
                "Punktem wyjścia powinna być nastawa rzędu sześćset pięćdziesiąt "
                "jednostek. Ponieważ odnotowano niestabilność warstwy jonosferycznej, "
                "trzeba dodać jeszcze dwadzieścia cztery."
            ),
            "day": (
                "Najbardziej typowy poziom dla podobnej podróży to wartość bazowa "
                "wynosząca siedemset. Można od tej wartości odjąć sto dwanaście jednostek."
            ),
        }
        payload: dict[str, object] = {"code": 11}
        if param in guidance:
            payload["config"] = {"stabilization": 0}
            payload["needConfig"] = guidance[param]
        return payload

    def get_config(self) -> DeviceConfig:
        return DeviceConfig(
            current_date="2026-04-10",
            day=10,
            month=4,
            year=2026,
            sync_ratio=0.69,
            stabilization=588,
            condition="unstable",
            flux_density=80,
            battery_status="1/3",
            PTA=True,
            PTB=False,
            PWR=28,
            mode="standby",
            internal_mode=2,
        )


class TimetravelBackendRunnerTests(unittest.TestCase):
    def test_configure_target_uses_guidance_from_final_date_response(self) -> None:
        runtime_root = Path("timetravel") / "runtime"
        runtime_root.mkdir(parents=True, exist_ok=True)
        state_path = runtime_root / "test_backend_runner.sqlite3"
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(f"{state_path}{suffix}")
            if candidate.exists():
                candidate.unlink()
        store = SharedStateStore(state_path)
        api_client = _FakeApiClient()
        runner = BackendMissionRunner(
            store=store,
            api_client=api_client,
            poll_interval_seconds=0.01,
            should_reset=False,
        )
        profile = build_target_profiles(date(2026, 4, 10), load_pwr_lookup())[1]

        stabilization = runner._configure_target(profile)

        for suffix in ("", "-wal", "-shm"):
            candidate = Path(f"{state_path}{suffix}")
            if candidate.exists():
                candidate.unlink()

        self.assertEqual(stabilization, 588)
        self.assertEqual(
            api_client.configure_calls[-2:],
            [("syncRatio", profile.sync_ratio), ("stabilization", 588)],
        )


if __name__ == "__main__":
    unittest.main()
