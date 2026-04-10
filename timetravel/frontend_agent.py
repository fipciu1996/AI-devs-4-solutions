"""Frontend agent for the timetravel task."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

REPO_ROOT_HINT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_HINT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_HINT))

from devs_utilities.bootstrap import bootstrap_repo
from devs_utilities.logging import configure_logging, logger as shared_logger
from devs_utilities.repo_env import get_llm_api_key, get_llm_base_url

from timetravel.frontend_ui import PREVIEW_URL, TIMETRAVEL_UI_SELECTORS, normalize_ui_state
from timetravel.models import DesiredUiState, MissionPhase, MissionStatus, ObservedUiState
from timetravel.store import SharedStateStore


REPO_ROOT = bootstrap_repo(__file__)
logger = shared_logger.bind(component="timetravel-frontend")
DEFAULT_RUNTIME_DIR = Path(__file__).resolve().parent / "runtime"


UI_SNAPSHOT_SCRIPT = """
() => {
  const batteryCells = Array.from(document.querySelectorAll('#batteryIndicator .battery-cell'));
  const chargedCells = batteryCells.filter((cell) => cell.classList.contains('charged')).length;
  const orb = document.querySelector('#orb');
  return {
    PTA: document.querySelector('#portA')?.classList.contains('connected') ?? null,
    PTB: document.querySelector('#portB')?.classList.contains('connected') ?? null,
    PWR: Number(document.querySelector('#pwrSlider')?.value ?? 0),
    mode: document.querySelector('#mainSwitch')?.classList.contains('active') ? 'active' : 'standby',
    batteryStatus: `${chargedCells}/3`,
    fluxDensity: parseInt(document.querySelector('#fluxPct')?.textContent ?? '0', 10) || 0,
    condition: document.querySelector('#condLabel')?.classList.contains('stable') ? 'stable' : 'unstable',
    orbClickable: Boolean(
      orb &&
      orb.classList.contains('powered') &&
      !orb.classList.contains('danger')
    ),
    flagText: document.querySelector('#flagText')?.textContent?.trim() ?? '',
  };
}
"""


@dataclass(frozen=True, slots=True)
class BrowserLaunch:
    """Browser objects returned by the launch strategy."""

    page: Any
    context: Any
    browser: Any
    source: str = "unknown"


class FrontendAgent:
    """Drive the preview UI and publish observed state."""

    def __init__(
        self,
        *,
        store: SharedStateStore,
        manual_login: bool,
        poll_interval_seconds: float,
        storage_state_path: Path,
    ) -> None:
        self.store = store
        self.manual_login = manual_login
        self.poll_interval_seconds = poll_interval_seconds
        self.storage_state_path = storage_state_path
        self._last_received_desired: DesiredUiState | None = None
        self._last_published_observed: ObservedUiState | None = None
        self._last_consumed_jump_id = 0

    def run(self) -> None:
        """Open the preview page and keep applying desired UI state."""

        self._last_consumed_jump_id = self.store.get_observed_ui().last_consumed_jump_id
        sync_api = self._import_playwright_sync_api()
        with sync_api.sync_playwright() as playwright:
            launch = self._launch_page(playwright)
            try:
                self._wait_for_ready_page(launch.page)
                self._persist_storage_state(launch.context, reason="preview became ready")
                while True:
                    mission_state = self.store.get_mission_state()
                    if mission_state.get("status") == MissionStatus.FAILED.value:
                        return
                    if mission_state.get("status") == MissionStatus.COMPLETED.value:
                        return

                    desired = self.store.get_desired_ui()
                    self._log_received_desired_if_changed(desired, mission_state.get("phase"))
                    observed = merge_observed_ui(
                        self._read_observed_ui(launch.page),
                        last_consumed_jump_id=self._last_consumed_jump_id,
                    )
                    self.store.set_observed_ui(observed)
                    self._log_published_observed_if_changed(observed, mission_state.get("phase"))

                    if observed.flag:
                        self._persist_storage_state(launch.context, reason="flag captured")
                        self.store.set_mission_state(
                            phase=MissionPhase.COMPLETED,
                            status=MissionStatus.COMPLETED,
                            flag=observed.flag,
                        )
                        self._event("INFO", f"Captured final flag: {observed.flag}")
                        return

                    self._apply_desired_ui(
                        launch.page,
                        desired,
                        observed,
                        mission_status=mission_state.get("status"),
                    )
                    time.sleep(self.poll_interval_seconds)
            finally:
                self._persist_storage_state(launch.context, reason="frontend shutdown")
                launch.context.close()
                launch.browser.close()

    def _launch_page(self, playwright: Any) -> BrowserLaunch:
        self._event("INFO", "Launching isolated Playwright Chromium.")
        try:
            return self._launch_isolated_browser(playwright)
        except Exception as exc:
            raise RuntimeError(
                "Could not launch bundled Playwright Chromium. "
                "Run `.venv\\Scripts\\python.exe -m playwright install chromium`."
            ) from exc

    def _wait_for_ready_page(self, page: Any) -> None:
        deadline = time.monotonic() + (900.0 if self.manual_login else 300.0)
        wait_message_shown = False
        while time.monotonic() < deadline:
            if self._preview_controls_present(page):
                self._event("INFO", "Timetravel preview controls are available.")
                return

            session_key = self._read_session_key(page)
            if session_key and self._ensure_preview_route(page):
                time.sleep(self.poll_interval_seconds)
                continue

            if not wait_message_shown:
                current_url = self._safe_page_url(page)
                message = (
                    "Preview controls are not available yet. If the opened browser shows a login page, "
                    "finish authentication manually; the frontend agent will continue automatically."
                )
                self._event("WARNING", f"{message} Current page: {current_url}")
                wait_message_shown = True

            time.sleep(self.poll_interval_seconds)

        raise TimeoutError(
            "Timed out waiting for timetravel preview controls to load. "
            f"Last page URL: {self._safe_page_url(page)}"
        )

    def _apply_desired_ui(
        self,
        page: Any,
        desired: DesiredUiState,
        observed: ObservedUiState,
        *,
        mission_status: str | None,
    ) -> None:
        if desired.mode and observed.mode != desired.mode:
            page.locator(TIMETRAVEL_UI_SELECTORS["mode_switch"]).click()
            self._event("INFO", f"Applied backend command: switched mode to {desired.mode}.")
            return

        if desired.PTA is not None and observed.PTA != desired.PTA:
            page.locator(TIMETRAVEL_UI_SELECTORS["port_a"]).click()
            self._event("INFO", f"Applied backend command: adjusted PT-A to {desired.PTA}.")
            return

        if desired.PTB is not None and observed.PTB != desired.PTB:
            if desired.PTA and desired.PTB and observed.battery_level < 2:
                self._event("INFO", "Waiting for battery >= 2/3 before enabling tunnel mode.")
                return
            page.locator(TIMETRAVEL_UI_SELECTORS["port_b"]).click()
            self._event("INFO", f"Applied backend command: adjusted PT-B to {desired.PTB}.")
            return

        if desired.PWR is not None and observed.PWR != desired.PWR:
            slider = page.locator(TIMETRAVEL_UI_SELECTORS["pwr_slider"])
            slider.evaluate(
                """
                (element, value) => {
                  element.value = String(value);
                  element.dispatchEvent(new Event('input', { bubbles: true }));
                  element.dispatchEvent(new Event('change', { bubbles: true }));
                }
                """,
                desired.PWR,
            )
            self._event("INFO", f"Applied backend command: adjusted PWR to {desired.PWR}.")
            return

        if not should_click_orb(desired, observed, mission_status=mission_status):
            return

        page.locator(TIMETRAVEL_UI_SELECTORS["orb"]).click()
        self._last_consumed_jump_id = desired.jump_request_id
        new_state = merge_observed_ui(
            self._read_observed_ui(page),
            last_consumed_jump_id=self._last_consumed_jump_id,
        )
        self.store.set_observed_ui(
            new_state
        )
        self._event("INFO", f"Clicked the orb for jump request {desired.jump_request_id}.")

    def _import_playwright_sync_api(self) -> Any:
        try:
            from playwright import sync_api
        except ImportError as exc:
            raise RuntimeError(
                "Playwright is not installed. Install dependencies before running the frontend agent."
            ) from exc
        return sync_api

    def _launch_isolated_browser(self, playwright: Any) -> BrowserLaunch:
        browser = playwright.chromium.launch(headless=False)
        context_options: dict[str, Any] = {}
        if self.storage_state_path.exists():
            context_options["storage_state"] = str(self.storage_state_path)
            self._event("INFO", f"Loading persisted Playwright session from {self.storage_state_path}.")
        context = browser.new_context(**context_options)
        page = context.new_page()
        page.goto(PREVIEW_URL, wait_until="domcontentloaded")
        self._event("INFO", "Opened preview using isolated Playwright Chromium.")
        return BrowserLaunch(
            page=page,
            context=context,
            browser=browser,
            source="playwright-chromium",
        )

    def _preview_controls_present(self, page: Any) -> bool:
        try:
            return bool(
                page.evaluate(
                    """
                    (selectors) => selectors.every((selector) => Boolean(document.querySelector(selector)))
                    """,
                    tuple(TIMETRAVEL_UI_SELECTORS.values()),
                )
            )
        except Exception:
            return False

    def _read_session_key(self, page: Any) -> str:
        try:
            return str(page.evaluate("() => window._EC_USER_UUID || ''") or "")
        except Exception:
            return ""

    def _ensure_preview_route(self, page: Any) -> bool:
        current_url = self._safe_page_url(page)
        if current_url.rstrip("/") == PREVIEW_URL.rstrip("/"):
            return False
        page.goto(PREVIEW_URL, wait_until="domcontentloaded")
        self._event("INFO", "Detected authenticated session; navigating back to timetravel preview.")
        return True

    def _safe_page_url(self, page: Any) -> str:
        try:
            return str(page.url)
        except Exception:
            return "<unavailable>"

    def _persist_storage_state(self, context: Any, *, reason: str) -> None:
        self.storage_state_path.parent.mkdir(parents=True, exist_ok=True)
        context.storage_state(path=str(self.storage_state_path))
        self._event("INFO", f"Saved Playwright session state to {self.storage_state_path} after {reason}.")

    def _read_observed_ui(self, page: Any) -> ObservedUiState:
        return normalize_ui_state(page.evaluate(UI_SNAPSHOT_SCRIPT))

    def _log_received_desired_if_changed(self, desired: DesiredUiState, phase: str | None) -> None:
        if desired == self._last_received_desired:
            return
        self._last_received_desired = desired
        self._event(
            "INFO",
            "Agent channel backend->frontend: "
            f"phase={phase or '<unknown>'} desired={_format_desired_ui(desired)}",
        )

    def _log_published_observed_if_changed(self, observed: ObservedUiState, phase: str | None) -> None:
        if observed == self._last_published_observed:
            return
        self._last_published_observed = observed
        self._event(
            "INFO",
            "Agent channel frontend->backend: "
            f"phase={phase or '<unknown>'} observed={_format_observed_ui(observed)}",
        )

    def _event(self, level: str, message: str) -> None:
        self.store.record_event("frontend", level, message)
        logger.log(level.upper(), message)


def _format_desired_ui(state: DesiredUiState) -> str:
    return (
        f"mode={state.mode} PT-A={state.PTA} PT-B={state.PTB} "
        f"PWR={state.PWR} jumpRequest={state.jump_request_id}"
    )


def _format_observed_ui(state: ObservedUiState) -> str:
    return (
        f"mode={state.mode} PT-A={state.PTA} PT-B={state.PTB} PWR={state.PWR} "
        f"battery={state.battery_status} flux={state.flux_density} "
        f"condition={state.condition} orbClickable={state.orb_clickable} "
        f"jumpConsumed={state.last_consumed_jump_id} flag={state.flag}"
    )


def merge_observed_ui(
    snapshot: ObservedUiState,
    *,
    last_consumed_jump_id: int,
) -> ObservedUiState:
    return replace(snapshot, last_consumed_jump_id=last_consumed_jump_id)


def should_click_orb(
    desired: DesiredUiState,
    observed: ObservedUiState,
    *,
    mission_status: str | None,
) -> bool:
    if mission_status != MissionStatus.READY_FOR_JUMP.value:
        return False
    if desired.jump_request_id <= observed.last_consumed_jump_id:
        return False
    return observed.orb_clickable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-path",
        type=Path,
        required=True,
        help="Path to the shared SQLite state database.",
    )
    parser.add_argument(
        "--agent-runtime",
        choices=("auto", "deterministic", "openrouter"),
        default="auto",
        help="Execution runtime for the frontend agent. Default: auto",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="OpenRouter model override for the frontend agent.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum OpenRouter tool-calling rounds for the frontend agent.",
    )
    parser.add_argument(
        "--openrouter-timeout-seconds",
        type=int,
        default=None,
        help="OpenRouter timeout for the frontend agent.",
    )
    parser.add_argument(
        "--transcript-path",
        type=Path,
        default=DEFAULT_RUNTIME_DIR / "frontend_agent_transcript.json",
        help="Where to save the frontend OpenRouter transcript.",
    )
    parser.add_argument(
        "--show-tool-results",
        action="store_true",
        help="Print each frontend OpenRouter tool result.",
    )
    parser.add_argument(
        "--storage-state-path",
        type=Path,
        default=None,
        help="Path to the persisted Playwright storage_state JSON file.",
    )
    parser.add_argument(
        "--manual-login",
        action="store_true",
        help="Open the preview and wait until the session becomes authenticated.",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=1.0,
        help="Polling interval for preview synchronization. Default: 1.0",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(name="timetravel-frontend", verbose=args.verbose)
    store = SharedStateStore(args.db_path)
    storage_state_path = args.storage_state_path or (args.db_path.parent / "frontend_storage_state.json")
    deterministic_agent = FrontendAgent(
        store=store,
        manual_login=args.manual_login,
        poll_interval_seconds=args.poll_interval_seconds,
        storage_state_path=storage_state_path,
    )
    agent_runtime = _resolve_agent_runtime(args.agent_runtime)
    try:
        if agent_runtime == "openrouter":
            from timetravel.openrouter_frontend import (
                DEFAULT_MAX_STEPS,
                DEFAULT_MODEL,
                DEFAULT_OPENROUTER_TIMEOUT_SECONDS,
                OpenRouterFrontendAgent,
            )

            agent = OpenRouterFrontendAgent(
                controller=deterministic_agent,
                transcript_path=args.transcript_path.resolve(),
                model=(args.model or DEFAULT_MODEL).strip(),
                timeout_seconds=args.openrouter_timeout_seconds or DEFAULT_OPENROUTER_TIMEOUT_SECONDS,
                max_steps=args.max_steps or DEFAULT_MAX_STEPS,
                show_tool_results=args.show_tool_results,
            )
            agent.run()
        else:
            deterministic_agent.run()
    except Exception as exc:
        message = str(exc)
        store.set_mission_state(
            phase=MissionPhase.FAILED,
            status=MissionStatus.FAILED,
            error=message,
        )
        store.record_event("frontend", "ERROR", message)
        logger.exception("Frontend agent failed.")
        return 1
    return 0


def _resolve_agent_runtime(requested_runtime: str) -> str:
    if requested_runtime != "auto":
        return requested_runtime
    if get_llm_api_key().strip() and get_llm_base_url().strip():
        return "openrouter"
    return "deterministic"


if __name__ == "__main__":
    raise SystemExit(main())
