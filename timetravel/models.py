"""Typed models and mission helpers for the timetravel task."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path
import json


class MissionPhase(StrEnum):
    """Known mission phases for the automated run."""

    ACQUIRE_BATTERIES = "acquire_batteries"
    RETURN_PRESENT = "return_present"
    OPEN_TUNNEL_2024 = "open_tunnel_2024"
    COMPLETED = "completed"
    FAILED = "failed"


class MissionStatus(StrEnum):
    """Allowed mission statuses shared between agents."""

    CONFIGURING = "configuring"
    WAITING_INTERNAL_MODE = "waiting_internal_mode"
    WAITING_UI = "waiting_ui"
    READY_FOR_JUMP = "ready_for_jump"
    JUMP_IN_PROGRESS = "jump_in_progress"
    LANDED = "landed"
    FAILED = "failed"
    COMPLETED = "completed"


@dataclass(frozen=True, slots=True)
class DeviceConfig:
    """Current CHRONOS-P1 configuration returned by the official API."""

    current_date: str | None
    day: int | None
    month: int | None
    year: int | None
    sync_ratio: float
    stabilization: int
    condition: str
    flux_density: int
    battery_status: str
    PTA: bool
    PTB: bool
    PWR: int
    mode: str
    internal_mode: int

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "DeviceConfig":
        """Normalize a raw `/verify` config payload."""

        return cls(
            current_date=_as_optional_str(payload.get("currentDate")),
            day=_as_optional_int(payload.get("day")),
            month=_as_optional_int(payload.get("month")),
            year=_as_optional_int(payload.get("year")),
            sync_ratio=float(payload.get("syncRatio") or 0.0),
            stabilization=int(payload.get("stabilization") or 0),
            condition=str(payload.get("condition") or "unstable"),
            flux_density=int(payload.get("fluxDensity") or 0),
            battery_status=str(payload.get("batteryStatus") or "0/3"),
            PTA=bool(payload.get("PTA")),
            PTB=bool(payload.get("PTB")),
            PWR=int(payload.get("PWR") or 0),
            mode=str(payload.get("mode") or "standby"),
            internal_mode=int(payload.get("internalMode") or 0),
        )

    @property
    def configured_date(self) -> str | None:
        """Return the configured target date as ISO-8601."""

        if self.day is None or self.month is None or self.year is None:
            return None
        return date(self.year, self.month, self.day).isoformat()

    @property
    def battery_level(self) -> int:
        """Return the charged battery cells from the `x/3` status."""

        raw_level = self.battery_status.split("/", 1)[0]
        try:
            return int(raw_level)
        except ValueError:
            return 0


@dataclass(frozen=True, slots=True)
class TargetProfile:
    """Mission parameters for a single travel phase."""

    phase: MissionPhase
    label: str
    year: int
    month: int
    day: int
    sync_ratio: float
    PWR: int
    PTA: bool
    PTB: bool
    required_internal_mode: int
    min_battery_before_jump: int
    min_battery_after_landing: int | None = None
    tunnel_mode: bool = False

    @property
    def target_date(self) -> str:
        """Return the profile date as ISO-8601."""

        return date(self.year, self.month, self.day).isoformat()


@dataclass(frozen=True, slots=True)
class DesiredUiState:
    """Desired UI controls published by the backend agent."""

    PTA: bool
    PTB: bool
    PWR: int
    mode: str
    jump_request_id: int = 0


@dataclass(frozen=True, slots=True)
class ObservedUiState:
    """Observed UI state published by the frontend agent."""

    PTA: bool | None = None
    PTB: bool | None = None
    PWR: int | None = None
    mode: str | None = None
    battery_status: str | None = None
    flux_density: int | None = None
    condition: str | None = None
    orb_clickable: bool = False
    flag: str | None = None
    last_consumed_jump_id: int = 0

    @property
    def battery_level(self) -> int:
        """Return the charged battery cells from the observed UI state."""

        if not self.battery_status:
            return 0
        raw_level = self.battery_status.split("/", 1)[0]
        try:
            return int(raw_level)
        except ValueError:
            return 0


def load_pwr_lookup(base_dir: Path | None = None) -> dict[int, int]:
    """Load the checked-in year -> PWR lookup."""

    root = base_dir or Path(__file__).resolve().parent
    lookup_path = root / "pwr_levels.json"
    raw_payload = json.loads(lookup_path.read_text(encoding="utf-8"))
    return {int(year): int(value) for year, value in raw_payload.items()}


def calculate_temporal_index(day: int, month: int, year: int) -> int:
    """Calculate the documentation-defined temporal index."""

    return ((day * 8) + (month * 12) + (year * 7)) % 101


def calculate_sync_ratio(day: int, month: int, year: int) -> float:
    """Convert the temporal index into the API sync ratio format."""

    temporal_index = calculate_temporal_index(day, month, year)
    if temporal_index == 100:
        return 1.0
    return round(temporal_index / 100, 2)


def required_internal_mode(year: int) -> int:
    """Map a target year to the required `internalMode`."""

    if year < 2000:
        return 1
    if year <= 2150:
        return 2
    if year <= 2300:
        return 3
    return 4


def build_target_profiles(
    present_date: date,
    pwr_lookup: dict[int, int],
) -> tuple[TargetProfile, ...]:
    """Build the fixed three-step mission using the current present date."""

    targets = (
        (
            MissionPhase.ACQUIRE_BATTERIES,
            "Acquire replacement batteries in 2238",
            date(2238, 11, 5),
            False,
            True,
            1,
            3,
            False,
        ),
        (
            MissionPhase.RETURN_PRESENT,
            "Return to present day",
            present_date,
            True,
            False,
            1,
            None,
            False,
        ),
        (
            MissionPhase.OPEN_TUNNEL_2024,
            "Open a tunnel to Rafał's meeting date",
            date(2024, 11, 12),
            True,
            True,
            2,
            None,
            True,
        ),
    )

    profiles: list[TargetProfile] = []
    for phase, label, target_date, pta, ptb, min_before, min_after, tunnel_mode in targets:
        profiles.append(
            TargetProfile(
                phase=phase,
                label=label,
                year=target_date.year,
                month=target_date.month,
                day=target_date.day,
                sync_ratio=calculate_sync_ratio(
                    day=target_date.day,
                    month=target_date.month,
                    year=target_date.year,
                ),
                PWR=get_pwr_for_year(target_date.year, pwr_lookup),
                PTA=pta,
                PTB=ptb,
                required_internal_mode=required_internal_mode(target_date.year),
                min_battery_before_jump=min_before,
                min_battery_after_landing=min_after,
                tunnel_mode=tunnel_mode,
            )
        )
    return tuple(profiles)


def get_pwr_for_year(year: int, pwr_lookup: dict[int, int]) -> int:
    """Return the documented PWR recommendation for a given year."""

    try:
        return pwr_lookup[year]
    except KeyError as exc:
        raise KeyError(f"Missing PWR lookup for year {year}.") from exc


def is_ui_ready_for_jump(
    profile: TargetProfile,
    config: DeviceConfig,
    observed: ObservedUiState | None,
) -> bool:
    """Check whether both backend and UI state permit an orb click."""

    if config.configured_date != profile.target_date:
        return False
    if config.sync_ratio != profile.sync_ratio:
        return False
    if config.internal_mode != profile.required_internal_mode:
        return False
    if config.PTA != profile.PTA or config.PTB != profile.PTB:
        return False
    if config.PWR != profile.PWR:
        return False
    if config.mode != "active":
        return False
    if config.condition != "stable":
        return False
    if config.flux_density != 100:
        return False
    if config.battery_level < profile.min_battery_before_jump:
        return False
    if observed is None:
        return False
    if observed.PTA != profile.PTA or observed.PTB != profile.PTB:
        return False
    if observed.PWR != profile.PWR:
        return False
    if observed.mode != "active":
        return False
    if observed.condition != "stable":
        return False
    if observed.flux_density != 100:
        return False
    if observed.battery_level < profile.min_battery_before_jump:
        return False
    return observed.orb_clickable


def phase_success_reached(
    profile: TargetProfile,
    config: DeviceConfig,
    observed: ObservedUiState | None,
) -> bool:
    """Check whether the current phase has completed successfully."""

    if profile.tunnel_mode:
        return bool(observed and observed.flag)
    if config.current_date != profile.target_date:
        return False
    if profile.min_battery_after_landing is None:
        return True
    return config.battery_level >= profile.min_battery_after_landing


def _as_optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _as_optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
