"""Shared UI constants and pure helpers for the frontend agent."""

from __future__ import annotations

from typing import Any

from devs_utilities.ag3nts import AG3NTS_TIMETRAVEL_PREVIEW_URL

from .backend_api import extract_flag_from_ui_text
from .models import ObservedUiState


PREVIEW_URL = AG3NTS_TIMETRAVEL_PREVIEW_URL

TIMETRAVEL_UI_SELECTORS = {
    "mode_switch": "#mainSwitch",
    "port_a": "#portA",
    "port_b": "#portB",
    "pwr_slider": "#pwrSlider",
    "orb": "#orb",
    "flux_pct": "#fluxPct",
    "device_mode": "#deviceMode",
    "flag_text": "#flagText",
    "condition_label": "#condLabel",
    "battery_indicator": "#batteryIndicator",
}


def required_ui_selectors() -> tuple[str, ...]:
    """Return the selectors the automation requires from the preview page."""

    return tuple(TIMETRAVEL_UI_SELECTORS.values())


def normalize_ui_state(raw_payload: dict[str, Any]) -> ObservedUiState:
    """Convert a browser-collected state payload into the typed store model."""

    return ObservedUiState(
        PTA=bool(raw_payload.get("PTA")) if raw_payload.get("PTA") is not None else None,
        PTB=bool(raw_payload.get("PTB")) if raw_payload.get("PTB") is not None else None,
        PWR=int(raw_payload["PWR"]) if raw_payload.get("PWR") is not None else None,
        mode=str(raw_payload["mode"]) if raw_payload.get("mode") else None,
        battery_status=str(raw_payload["batteryStatus"]) if raw_payload.get("batteryStatus") else None,
        flux_density=int(raw_payload["fluxDensity"]) if raw_payload.get("fluxDensity") is not None else None,
        condition=str(raw_payload["condition"]) if raw_payload.get("condition") else None,
        orb_clickable=bool(raw_payload.get("orbClickable")),
        flag=extract_flag_from_ui_text(raw_payload.get("flagText")),
        last_consumed_jump_id=int(raw_payload.get("lastConsumedJumpId") or 0),
    )
