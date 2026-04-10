from __future__ import annotations

from datetime import date
import unittest

from timetravel.models import (
    DeviceConfig,
    MissionPhase,
    ObservedUiState,
    build_target_profiles,
    calculate_sync_ratio,
    calculate_temporal_index,
    is_ui_ready_for_jump,
    load_pwr_lookup,
    phase_success_reached,
    required_internal_mode,
)


class TimetravelModelTests(unittest.TestCase):
    def test_calculate_temporal_index_uses_documented_weights(self) -> None:
        self.assertEqual(calculate_temporal_index(5, 11, 2238), 82)
        self.assertEqual(calculate_temporal_index(10, 4, 2026), 69)
        self.assertEqual(calculate_temporal_index(12, 11, 2024), 54)

    def test_calculate_sync_ratio_matches_known_dates(self) -> None:
        self.assertEqual(calculate_sync_ratio(5, 11, 2238), 0.82)
        self.assertEqual(calculate_sync_ratio(10, 4, 2026), 0.69)
        self.assertEqual(calculate_sync_ratio(12, 11, 2024), 0.54)

    def test_required_internal_mode_respects_documented_ranges(self) -> None:
        self.assertEqual(required_internal_mode(1999), 1)
        self.assertEqual(required_internal_mode(2000), 2)
        self.assertEqual(required_internal_mode(2150), 2)
        self.assertEqual(required_internal_mode(2151), 3)
        self.assertEqual(required_internal_mode(2300), 3)
        self.assertEqual(required_internal_mode(2301), 4)

    def test_pwr_lookup_contains_required_mission_years(self) -> None:
        lookup = load_pwr_lookup()

        self.assertEqual(lookup[2024], 19)
        self.assertEqual(lookup[2026], 28)
        self.assertEqual(lookup[2238], 91)

    def test_build_target_profiles_keeps_expected_phase_sequence(self) -> None:
        profiles = build_target_profiles(date(2026, 4, 10), load_pwr_lookup())

        self.assertEqual(
            [profile.phase for profile in profiles],
            [
                MissionPhase.ACQUIRE_BATTERIES,
                MissionPhase.RETURN_PRESENT,
                MissionPhase.OPEN_TUNNEL_2024,
            ],
        )
        self.assertEqual(profiles[0].sync_ratio, 0.82)
        self.assertEqual(profiles[1].PWR, 28)
        self.assertTrue(profiles[2].tunnel_mode)

    def test_is_ui_ready_for_jump_requires_fully_matching_state(self) -> None:
        profile = build_target_profiles(date(2026, 4, 10), load_pwr_lookup())[2]
        config = DeviceConfig(
            current_date="2026-04-10",
            day=12,
            month=11,
            year=2024,
            sync_ratio=0.54,
            stabilization=321,
            condition="stable",
            flux_density=100,
            battery_status="2/3",
            PTA=True,
            PTB=True,
            PWR=19,
            mode="active",
            internal_mode=2,
        )
        observed = ObservedUiState(
            PTA=True,
            PTB=True,
            PWR=19,
            mode="active",
            battery_status="2/3",
            flux_density=100,
            condition="stable",
            orb_clickable=True,
        )

        self.assertTrue(is_ui_ready_for_jump(profile, config, observed))

    def test_phase_success_reached_requires_flag_for_tunnel(self) -> None:
        profile = build_target_profiles(date(2026, 4, 10), load_pwr_lookup())[2]
        config = DeviceConfig(
            current_date="2026-04-10",
            day=12,
            month=11,
            year=2024,
            sync_ratio=0.54,
            stabilization=123,
            condition="stable",
            flux_density=100,
            battery_status="2/3",
            PTA=True,
            PTB=True,
            PWR=19,
            mode="active",
            internal_mode=2,
        )

        self.assertFalse(phase_success_reached(profile, config, ObservedUiState()))
        self.assertTrue(
            phase_success_reached(
                profile,
                config,
                ObservedUiState(flag="{FLG:example}"),
            )
        )


if __name__ == "__main__":
    unittest.main()

