from __future__ import annotations

import sys
import unittest
from pathlib import Path

WINDPOWER_DIR = Path(__file__).resolve().parent
REPO_ROOT = WINDPOWER_DIR.parent
for candidate in (str(REPO_ROOT), str(WINDPOWER_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from solve_windpower import (
    ConfigPoint,
    TurbineDocumentation,
    WeatherPoint,
    build_config_plan,
    parse_required_power_kw,
)


def build_documentation() -> TurbineDocumentation:
    return TurbineDocumentation(
        rated_power_kw=14.0,
        min_operational_wind_ms=4.0,
        cutoff_wind_ms=14.0,
        wind_yield_points=((4.0, 12.5), (6.0, 35.0), (8.0, 65.0), (10.0, 95.0), (12.0, 100.0)),
        pitch_yield_factors={0: 1.0, 45: 0.65, 90: 0.0},
    )


class WindpowerPlanningTests(unittest.TestCase):
    def test_parse_required_power_kw_uses_upper_bound_of_range(self) -> None:
        self.assertEqual(parse_required_power_kw("2-3"), 3.0)

    def test_estimate_power_kw_interpolates_between_documented_points(self) -> None:
        documentation = build_documentation()

        estimated_power = documentation.estimate_power_kw(4.9, pitch_angle=0)

        self.assertGreaterEqual(estimated_power, 3.0)

    def test_build_config_plan_adds_storm_protection_and_earliest_viable_slot(self) -> None:
        documentation = build_documentation()
        weather_points = [
            WeatherPoint("2026-04-01 18:00:00", 25.0, 0.0, 30.6),
            WeatherPoint("2026-04-01 20:00:00", 4.9, 0.0, 30.0),
            WeatherPoint("2026-04-03 20:00:00", 4.9, 0.0, 29.6),
            WeatherPoint("2026-04-04 18:00:00", 22.0, 0.0, 29.4),
            WeatherPoint("2026-04-05 18:00:00", 28.0, 0.2, 31.5),
        ]

        plan = build_config_plan(weather_points, documentation, required_power_kw=3.0)

        self.assertEqual(
            plan,
            [
                ConfigPoint("2026-04-01 18:00:00", 25.0, 90, "idle"),
                ConfigPoint("2026-04-01 20:00:00", 4.9, 0, "production"),
                ConfigPoint("2026-04-04 18:00:00", 22.0, 90, "idle"),
                ConfigPoint("2026-04-05 18:00:00", 28.0, 90, "idle"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
