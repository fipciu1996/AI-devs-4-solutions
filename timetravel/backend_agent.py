"""Backend agent for the timetravel task."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from datetime import date
from pathlib import Path

REPO_ROOT_HINT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_HINT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_HINT))

from devs_utilities.bootstrap import bootstrap_repo
from devs_utilities.logging import configure_logging, logger as shared_logger
from devs_utilities.repo_env import get_course_api_key, get_llm_api_key, get_llm_base_url

from timetravel.backend_api import TimetravelApiClient, extract_stabilization_value
from timetravel.models import (
    DesiredUiState,
    MissionPhase,
    MissionStatus,
    TargetProfile,
    build_target_profiles,
    is_ui_ready_for_jump,
    load_pwr_lookup,
    phase_success_reached,
)
from timetravel.store import SharedStateStore


REPO_ROOT = bootstrap_repo(__file__)
logger = shared_logger.bind(component="timetravel-backend")
DEFAULT_RUNTIME_DIR = Path(__file__).resolve().parent / "runtime"


class BackendMissionRunner:
    """Drive the official `/verify` API and coordinate the UI agent."""

    def __init__(
        self,
        *,
        store: SharedStateStore,
        api_client: TimetravelApiClient,
        poll_interval_seconds: float,
        should_reset: bool,
    ) -> None:
        self.store = store
        self.api_client = api_client
        self.poll_interval_seconds = poll_interval_seconds
        self.should_reset = should_reset

    def run(self) -> str | None:
        """Run the three-phase mission to completion."""

        self._event("INFO", "Starting backend mission runner.")
        self.api_client.help()
        if self.should_reset:
            self._event("INFO", "Reset requested; resetting device before mission start.")
            self.api_client.reset()

        initial_config = self._refresh_config()
        if not initial_config.current_date:
            raise RuntimeError("Missing currentDate in initial getConfig response.")
        present_date = date.fromisoformat(initial_config.current_date)
        self.store.set_mission_state(
            phase=MissionPhase.ACQUIRE_BATTERIES,
            status=MissionStatus.CONFIGURING,
            error=None,
            flag=None,
            present_date=present_date.isoformat(),
        )
        profiles = build_target_profiles(present_date, load_pwr_lookup())
        for profile in profiles:
            self._run_phase(profile)

        final_state = self.store.get_mission_state()
        return final_state.get("flag")

    def _run_phase(self, profile: TargetProfile) -> None:
        self._event("INFO", f"Preparing phase: {profile.phase.value}.")
        self.store.set_mission_state(
            phase=profile.phase,
            status=MissionStatus.CONFIGURING,
            error=None,
        )
        self._publish_desired_ui(
            DesiredUiState(
                PTA=profile.PTA,
                PTB=profile.PTB,
                PWR=profile.PWR,
                mode="standby",
                jump_request_id=self.store.get_desired_ui().jump_request_id,
            ),
            reason=f"prepare {profile.phase.value} in standby",
        )
        self._wait_for_backend_mode("standby")
        stabilization = self._configure_target(profile)
        self._event(
            "INFO",
            (
                f"Configured {profile.target_date} with syncRatio={profile.sync_ratio:.2f} "
                f"and stabilization={stabilization}."
            ),
        )
        self.store.set_mission_state(status=MissionStatus.WAITING_INTERNAL_MODE)
        self._wait_for_internal_mode(profile.required_internal_mode)
        self.store.set_mission_state(status=MissionStatus.WAITING_UI)
        self._wait_for_ui_in_standby(profile)

        desired_state = self.store.get_desired_ui()
        self._publish_desired_ui(
            replace(desired_state, mode="active"),
            reason=f"arm {profile.phase.value} for jump",
        )
        self._wait_for_ui_ready(profile)

        jump_request_id = self.store.bump_jump_request()
        self.store.set_mission_state(status=MissionStatus.READY_FOR_JUMP)
        self._event(
            "INFO",
            "Agent channel backend->frontend: "
            f"phase={profile.phase.value} jumpRequest={jump_request_id}",
        )
        self._wait_for_jump_consumption(jump_request_id)
        self.store.set_mission_state(status=MissionStatus.JUMP_IN_PROGRESS)
        self._wait_for_phase_success(profile)
        if profile.phase == MissionPhase.OPEN_TUNNEL_2024:
            self.store.set_mission_state(
                phase=MissionPhase.COMPLETED,
                status=MissionStatus.COMPLETED,
            )
        else:
            self.store.set_mission_state(status=MissionStatus.LANDED)

    def _configure_target(self, profile: TargetProfile) -> int:
        responses = [
            self.api_client.configure("year", profile.year),
            self.api_client.configure("month", profile.month),
            self.api_client.configure("day", profile.day),
        ]
        stabilization = extract_stabilization_value(responses[-1])
        if stabilization is None:
            for response in reversed(responses[:-1]):
                stabilization = extract_stabilization_value(response)
                if stabilization is not None:
                    break
        if stabilization is None:
            for response in responses:
                self._event("DEBUG", f"Stabilization guidance payload: {response!r}")
        if stabilization is None:
            raise RuntimeError(
                f"Could not extract stabilization guidance for {profile.target_date}."
            )

        self.api_client.configure("syncRatio", profile.sync_ratio)
        self.api_client.configure("stabilization", stabilization)
        confirmed_config = self._refresh_config()
        if confirmed_config.configured_date != profile.target_date:
            raise RuntimeError(
                f"Configured date mismatch after API writes: {confirmed_config.configured_date!r}"
            )
        if confirmed_config.sync_ratio != profile.sync_ratio:
            raise RuntimeError(
                f"Configured syncRatio mismatch: {confirmed_config.sync_ratio!r}"
            )
        if confirmed_config.stabilization != stabilization:
            raise RuntimeError(
                f"Configured stabilization mismatch: {confirmed_config.stabilization!r}"
            )
        return stabilization

    def _wait_for_backend_mode(self, mode: str, timeout_seconds: float = 120.0) -> None:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            config = self._refresh_config()
            if config.mode == mode:
                return
            time.sleep(self.poll_interval_seconds)
        raise TimeoutError(f"Timed out waiting for backend mode={mode}.")

    def _wait_for_internal_mode(self, internal_mode: int, timeout_seconds: float = 180.0) -> None:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            config = self._refresh_config()
            if config.internal_mode == internal_mode:
                self._event("INFO", f"internalMode matched required value {internal_mode}.")
                return
            time.sleep(self.poll_interval_seconds)
        raise TimeoutError(f"Timed out waiting for internalMode={internal_mode}.")

    def _wait_for_ui_in_standby(
        self,
        profile: TargetProfile,
        timeout_seconds: float = 180.0,
    ) -> None:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            config = self._refresh_config()
            observed = self.store.get_observed_ui()
            if (
                config.mode == "standby"
                and config.PTA == profile.PTA
                and config.PTB == profile.PTB
                and config.PWR == profile.PWR
                and observed.mode == "standby"
                and observed.PTA == profile.PTA
                and observed.PTB == profile.PTB
                and observed.PWR == profile.PWR
            ):
                return
            time.sleep(self.poll_interval_seconds)
        raise TimeoutError(f"Timed out waiting for standby UI alignment for {profile.phase.value}.")

    def _wait_for_ui_ready(
        self,
        profile: TargetProfile,
        timeout_seconds: float = 240.0,
    ) -> None:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            config = self._refresh_config()
            observed = self.store.get_observed_ui()
            if is_ui_ready_for_jump(profile, config, observed):
                self._event("INFO", f"UI is ready for {profile.phase.value}.")
                return
            time.sleep(self.poll_interval_seconds)
        raise TimeoutError(f"Timed out waiting for active UI readiness for {profile.phase.value}.")

    def _wait_for_jump_consumption(
        self,
        jump_request_id: int,
        timeout_seconds: float = 60.0,
    ) -> None:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            observed = self.store.get_observed_ui()
            if observed.last_consumed_jump_id >= jump_request_id:
                return
            if observed.flag:
                return
            time.sleep(self.poll_interval_seconds)
        raise TimeoutError(f"Timed out waiting for frontend to consume jump request {jump_request_id}.")

    def _wait_for_phase_success(
        self,
        profile: TargetProfile,
        timeout_seconds: float = 180.0,
    ) -> None:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            config = self._refresh_config()
            observed = self.store.get_observed_ui()
            if observed.flag:
                self.store.set_mission_state(flag=observed.flag)
            if phase_success_reached(profile, config, observed):
                self._event("INFO", f"Phase {profile.phase.value} finished successfully.")
                return
            time.sleep(self.poll_interval_seconds)
        raise TimeoutError(f"Timed out waiting for success of phase {profile.phase.value}.")

    def _refresh_config(self):
        config = self.api_client.get_config()
        self.store.set_backend_snapshot(
            {
                "currentDate": config.current_date,
                "day": config.day,
                "month": config.month,
                "year": config.year,
                "syncRatio": config.sync_ratio,
                "stabilization": config.stabilization,
                "condition": config.condition,
                "fluxDensity": config.flux_density,
                "batteryStatus": config.battery_status,
                "PTA": config.PTA,
                "PTB": config.PTB,
                "PWR": config.PWR,
                "mode": config.mode,
                "internalMode": config.internal_mode,
            }
        )
        return config

    def _event(self, level: str, message: str) -> None:
        self.store.record_event("backend", level, message)
        logger.log(level.upper(), message)

    def _publish_desired_ui(self, state: DesiredUiState, *, reason: str) -> None:
        self.store.set_desired_ui(state)
        self._event(
            "INFO",
            "Agent channel backend->frontend: "
            f"{reason}; desired={_format_desired_ui(state)}",
        )


def _format_desired_ui(state: DesiredUiState) -> str:
    return (
        f"mode={state.mode} PT-A={state.PTA} PT-B={state.PTB} "
        f"PWR={state.PWR} jumpRequest={state.jump_request_id}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-path",
        type=Path,
        required=True,
        help="Path to the shared SQLite state database.",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=1.0,
        help="Polling interval for backend/UI coordination. Default: 1.0",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset the device before starting the mission.",
    )
    parser.add_argument(
        "--agent-runtime",
        choices=("auto", "deterministic", "openrouter"),
        default="auto",
        help="Execution runtime for the backend agent. Default: auto",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="OpenRouter model override for the backend agent.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum OpenRouter tool-calling rounds for the backend agent.",
    )
    parser.add_argument(
        "--openrouter-timeout-seconds",
        type=int,
        default=None,
        help="OpenRouter timeout for the backend agent.",
    )
    parser.add_argument(
        "--transcript-path",
        type=Path,
        default=DEFAULT_RUNTIME_DIR / "backend_agent_transcript.json",
        help="Where to save the backend OpenRouter transcript.",
    )
    parser.add_argument(
        "--show-tool-results",
        action="store_true",
        help="Print each backend OpenRouter tool result.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(name="timetravel-backend", verbose=args.verbose)
    store = SharedStateStore(args.db_path)
    api_client = TimetravelApiClient(api_key=get_course_api_key())
    deterministic_runner = BackendMissionRunner(
        store=store,
        api_client=api_client,
        poll_interval_seconds=args.poll_interval_seconds,
        should_reset=args.reset,
    )
    agent_runtime = _resolve_agent_runtime(args.agent_runtime)
    try:
        if agent_runtime == "openrouter":
            from timetravel.openrouter_backend import (
                DEFAULT_MAX_STEPS,
                DEFAULT_MODEL,
                DEFAULT_OPENROUTER_TIMEOUT_SECONDS,
                OpenRouterBackendAgent,
            )

            runner = OpenRouterBackendAgent(
                runner=deterministic_runner,
                transcript_path=args.transcript_path.resolve(),
                model=(args.model or DEFAULT_MODEL).strip(),
                timeout_seconds=args.openrouter_timeout_seconds or DEFAULT_OPENROUTER_TIMEOUT_SECONDS,
                max_steps=args.max_steps or DEFAULT_MAX_STEPS,
                show_tool_results=args.show_tool_results,
            )
            flag = runner.run()
        else:
            flag = deterministic_runner.run()
    except Exception as exc:
        message = str(exc)
        store.set_mission_state(
            phase=MissionPhase.FAILED,
            status=MissionStatus.FAILED,
            error=message,
        )
        store.record_event("backend", "ERROR", message)
        logger.exception("Backend mission runner failed.")
        return 1

    if flag:
        logger.info("Mission finished with flag: {}", flag)
    return 0


def _resolve_agent_runtime(requested_runtime: str) -> str:
    if requested_runtime != "auto":
        return requested_runtime
    if get_llm_api_key().strip() and get_llm_base_url().strip():
        return "openrouter"
    return "deterministic"


if __name__ == "__main__":
    raise SystemExit(main())
