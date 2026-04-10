"""OpenRouter-backed backend agent for the timetravel task."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

from devs_utilities.openrouter import OpenRouterClient, build_task_openrouter_client
from devs_utilities.repo_env import (
    get_int_env,
    get_llm_api_key,
    get_llm_base_url,
    get_llm_model,
    get_optional_env,
)

from timetravel.backend_agent import BackendMissionRunner
from timetravel.models import MissionPhase, MissionStatus, TargetProfile, build_target_profiles, load_pwr_lookup
from timetravel.openrouter_runtime import ToolCallingAgentConfig, run_tool_calling_agent


TASK_NAME = "timetravel-backend"
DEFAULT_MODEL = (
    get_optional_env("TIMETRAVEL_BACKEND_MODEL")
    or get_optional_env("TIMETRAVEL_MODEL")
    or get_llm_model()
)
DEFAULT_OPENROUTER_TIMEOUT_SECONDS = get_int_env("OPENROUTER_TIMEOUT_SECONDS", 120) or 120
DEFAULT_MAX_STEPS = get_int_env("TIMETRAVEL_BACKEND_MAX_STEPS", 32) or 32

SYSTEM_PROMPT = """
You are the backend timetravel agent coordinating the CHRONOS-P1 mission.
You control only the official /verify API and the shared mission state.

Mission order is fixed:
1. acquire_batteries -> 2238-11-05
2. return_present -> the device present_date discovered at startup
3. open_tunnel_2024 -> 2024-11-12

Use tools only. Do not invent values.
Always inspect state first, then advance exactly one phase at a time.
For each phase, complete this sequence:
1. prepare_current_phase
2. configure_current_phase
3. wait_for_required_internal_mode
4. wait_for_ui_standby_alignment
5. arm_current_phase
6. wait_for_ui_ready
7. publish_jump_request
8. wait_for_phase_completion
9. advance_phase

Rules:
- Never publish a jump request twice for the same phase.
- Never advance the phase before wait_for_phase_completion succeeds.
- Stop only when the store reports MissionStatus.COMPLETED or FAILED.
- If a tool reports an error, inspect state and recover with the next safe step.
""".strip()

INITIAL_USER_PROMPT = (
    "Start the backend timetravel mission. Inspect the shared state first, then execute the "
    "fixed three-phase mission through tools."
)


OPENROUTER_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "inspect_state",
            "description": "Read the current backend, frontend, and mission state plus recent events.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "prepare_current_phase",
            "description": "Publish standby desired_ui and set mission_state for the current phase.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "configure_current_phase",
            "description": "Configure year/month/day/syncRatio/stabilization for the current phase while backend is in standby.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wait_for_required_internal_mode",
            "description": "Wait until internalMode matches the current phase target year.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wait_for_ui_standby_alignment",
            "description": "Wait until frontend reports the expected standby UI settings for the current phase.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "arm_current_phase",
            "description": "Switch desired_ui to active for the current phase without requesting the jump yet.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wait_for_ui_ready",
            "description": "Wait until backend and frontend both report a jump-ready active configuration.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "publish_jump_request",
            "description": "Publish the next jumpRequest and set MissionStatus.READY_FOR_JUMP.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wait_for_phase_completion",
            "description": "Wait for frontend to consume the jump and for the current phase to complete successfully.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "advance_phase",
            "description": "Advance the local phase cursor after a successful phase completion.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
]


@dataclass(slots=True)
class BackendAgentState:
    profiles: tuple[TargetProfile, ...]
    phase_index: int = 0
    last_stabilization: int | None = None
    last_jump_request_id: int | None = None
    completed: bool = False

    @property
    def current_profile(self) -> TargetProfile | None:
        if self.phase_index >= len(self.profiles):
            return None
        return self.profiles[self.phase_index]


class OpenRouterBackendAgent:
    """Tool-calling backend orchestration over the deterministic mission primitives."""

    def __init__(
        self,
        *,
        runner: BackendMissionRunner,
        transcript_path: Path,
        model: str,
        timeout_seconds: int,
        max_steps: int,
        show_tool_results: bool,
    ) -> None:
        self.runner = runner
        self.transcript_path = transcript_path
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_steps = max_steps
        self.show_tool_results = show_tool_results

    def run(self) -> str | None:
        self.runner._event("INFO", "Starting OpenRouter backend mission runner.")
        self.runner.api_client.help()
        if self.runner.should_reset:
            self.runner._event("INFO", "Reset requested; resetting device before mission start.")
            self.runner.api_client.reset()

        initial_config = self.runner._refresh_config()
        if not initial_config.current_date:
            raise RuntimeError("Missing currentDate in initial getConfig response.")
        present_date = date.fromisoformat(initial_config.current_date)
        self.runner.store.set_mission_state(
            phase=MissionPhase.ACQUIRE_BATTERIES,
            status=MissionStatus.CONFIGURING,
            error=None,
            flag=None,
            present_date=present_date.isoformat(),
        )
        state = BackendAgentState(
            profiles=build_target_profiles(present_date, load_pwr_lookup()),
        )
        handlers = self._build_tool_handlers(state)
        client = self._build_client()
        config = ToolCallingAgentConfig(
            name=TASK_NAME,
            system_prompt=SYSTEM_PROMPT,
            initial_user_prompt=INITIAL_USER_PROMPT,
            tools=OPENROUTER_TOOLS,
            max_steps=self.max_steps,
            transcript_path=self.transcript_path,
            show_tool_results=self.show_tool_results,
        )
        run_tool_calling_agent(
            client=client,
            config=config,
            tool_handlers=handlers,
            finish_predicate=lambda: self._is_finished(state),
        )
        final_state = self.runner.store.get_mission_state()
        return final_state.get("flag")

    def _build_client(self) -> OpenRouterClient:
        api_key = get_llm_api_key().strip()
        base_url = get_llm_base_url().strip()
        if not api_key:
            raise RuntimeError("Missing OpenRouter API key. Set LLM_API_KEY or OPENROUTER_API_KEY.")
        if not base_url:
            raise RuntimeError("Missing OpenRouter base URL. Set LLM_BASE_URL or OPENROUTER_BASE_URL.")
        if not self.model:
            raise RuntimeError(
                "Missing OpenRouter model. Set TIMETRAVEL_BACKEND_MODEL, TIMETRAVEL_MODEL, "
                "OPENROUTER_MODEL, or LLM_MODEL."
            )
        return build_task_openrouter_client(
            __file__,
            api_key=api_key,
            base_url=base_url,
            model=self.model,
            task_name=TASK_NAME,
            timeout_seconds=self.timeout_seconds,
        )

    def _build_tool_handlers(self, state: BackendAgentState) -> dict[str, Any]:
        def inspect_state(_: dict[str, Any]) -> dict[str, Any]:
            profile = state.current_profile
            return {
                "ok": True,
                "phase_index": state.phase_index,
                "current_phase": profile.phase.value if profile else None,
                "current_profile": _serialize_profile(profile),
                "mission_state": self.runner.store.get_mission_state(),
                "desired_ui": asdict(self.runner.store.get_desired_ui()),
                "observed_ui": asdict(self.runner.store.get_observed_ui()),
                "backend_snapshot": self.runner.store.get_backend_snapshot(),
                "recent_events": self.runner.store.get_recent_events(limit=12),
                "last_stabilization": state.last_stabilization,
                "last_jump_request_id": state.last_jump_request_id,
                "completed": state.completed,
            }

        def prepare_current_phase(_: dict[str, Any]) -> dict[str, Any]:
            profile = _require_profile(state)
            self.runner.store.set_mission_state(
                phase=profile.phase,
                status=MissionStatus.CONFIGURING,
                error=None,
            )
            self.runner._publish_desired_ui(
                self.runner.store.get_desired_ui().__class__(
                    PTA=profile.PTA,
                    PTB=profile.PTB,
                    PWR=profile.PWR,
                    mode="standby",
                    jump_request_id=self.runner.store.get_desired_ui().jump_request_id,
                ),
                reason=f"prepare {profile.phase.value} in standby",
            )
            return {
                "ok": True,
                "profile": _serialize_profile(profile),
                "mission_state": self.runner.store.get_mission_state(),
            }

        def configure_current_phase(_: dict[str, Any]) -> dict[str, Any]:
            profile = _require_profile(state)
            self.runner._wait_for_backend_mode("standby")
            stabilization = self.runner._configure_target(profile)
            state.last_stabilization = stabilization
            self.runner._event(
                "INFO",
                f"Configured {profile.target_date} with syncRatio={profile.sync_ratio:.2f} and stabilization={stabilization}.",
            )
            return {
                "ok": True,
                "profile": _serialize_profile(profile),
                "stabilization": stabilization,
                "backend_snapshot": self.runner.store.get_backend_snapshot(),
            }

        def wait_for_required_internal_mode(_: dict[str, Any]) -> dict[str, Any]:
            profile = _require_profile(state)
            self.runner.store.set_mission_state(status=MissionStatus.WAITING_INTERNAL_MODE)
            self.runner._wait_for_internal_mode(profile.required_internal_mode)
            return {
                "ok": True,
                "required_internal_mode": profile.required_internal_mode,
                "backend_snapshot": self.runner.store.get_backend_snapshot(),
            }

        def wait_for_ui_standby_alignment(_: dict[str, Any]) -> dict[str, Any]:
            profile = _require_profile(state)
            self.runner.store.set_mission_state(status=MissionStatus.WAITING_UI)
            self.runner._wait_for_ui_in_standby(profile)
            return {
                "ok": True,
                "observed_ui": asdict(self.runner.store.get_observed_ui()),
                "backend_snapshot": self.runner.store.get_backend_snapshot(),
            }

        def arm_current_phase(_: dict[str, Any]) -> dict[str, Any]:
            profile = _require_profile(state)
            desired_state = self.runner.store.get_desired_ui()
            self.runner._publish_desired_ui(
                desired_state.__class__(
                    PTA=desired_state.PTA,
                    PTB=desired_state.PTB,
                    PWR=desired_state.PWR,
                    mode="active",
                    jump_request_id=desired_state.jump_request_id,
                ),
                reason=f"arm {profile.phase.value} for jump",
            )
            return {"ok": True, "desired_ui": asdict(self.runner.store.get_desired_ui())}

        def wait_for_ui_ready(_: dict[str, Any]) -> dict[str, Any]:
            profile = _require_profile(state)
            self.runner._wait_for_ui_ready(profile)
            return {
                "ok": True,
                "observed_ui": asdict(self.runner.store.get_observed_ui()),
                "backend_snapshot": self.runner.store.get_backend_snapshot(),
            }

        def publish_jump_request(_: dict[str, Any]) -> dict[str, Any]:
            profile = _require_profile(state)
            jump_request_id = self.runner.store.bump_jump_request()
            state.last_jump_request_id = jump_request_id
            self.runner.store.set_mission_state(status=MissionStatus.READY_FOR_JUMP)
            self.runner._event(
                "INFO",
                "Agent channel backend->frontend: "
                f"phase={profile.phase.value} jumpRequest={jump_request_id}",
            )
            return {
                "ok": True,
                "jump_request_id": jump_request_id,
                "mission_state": self.runner.store.get_mission_state(),
            }

        def wait_for_phase_completion(_: dict[str, Any]) -> dict[str, Any]:
            profile = _require_profile(state)
            if state.last_jump_request_id is None:
                raise RuntimeError("Jump request has not been published yet.")
            self.runner._wait_for_jump_consumption(state.last_jump_request_id)
            self.runner.store.set_mission_state(status=MissionStatus.JUMP_IN_PROGRESS)
            self.runner._wait_for_phase_success(profile)
            if profile.phase == MissionPhase.OPEN_TUNNEL_2024:
                state.completed = True
                self.runner.store.set_mission_state(
                    phase=MissionPhase.COMPLETED,
                    status=MissionStatus.COMPLETED,
                )
            else:
                self.runner.store.set_mission_state(status=MissionStatus.LANDED)
            return {
                "ok": True,
                "profile": _serialize_profile(profile),
                "mission_state": self.runner.store.get_mission_state(),
                "backend_snapshot": self.runner.store.get_backend_snapshot(),
            }

        def advance_phase(_: dict[str, Any]) -> dict[str, Any]:
            if state.phase_index < len(state.profiles):
                state.phase_index += 1
            next_profile = state.current_profile
            if next_profile is None:
                state.completed = True
            return {
                "ok": True,
                "phase_index": state.phase_index,
                "next_phase": next_profile.phase.value if next_profile else None,
                "completed": state.completed,
            }

        return {
            "inspect_state": inspect_state,
            "prepare_current_phase": prepare_current_phase,
            "configure_current_phase": configure_current_phase,
            "wait_for_required_internal_mode": wait_for_required_internal_mode,
            "wait_for_ui_standby_alignment": wait_for_ui_standby_alignment,
            "arm_current_phase": arm_current_phase,
            "wait_for_ui_ready": wait_for_ui_ready,
            "publish_jump_request": publish_jump_request,
            "wait_for_phase_completion": wait_for_phase_completion,
            "advance_phase": advance_phase,
        }

    def _is_finished(self, state: BackendAgentState) -> bool:
        mission_state = self.runner.store.get_mission_state()
        if mission_state.get("status") in {MissionStatus.COMPLETED.value, MissionStatus.FAILED.value}:
            return True
        return state.completed


def _require_profile(state: BackendAgentState) -> TargetProfile:
    profile = state.current_profile
    if profile is None:
        raise RuntimeError("No current phase is available.")
    return profile


def _serialize_profile(profile: TargetProfile | None) -> dict[str, Any] | None:
    if profile is None:
        return None
    return {
        "phase": profile.phase.value,
        "target_date": profile.target_date,
        "sync_ratio": profile.sync_ratio,
        "PWR": profile.PWR,
        "PTA": profile.PTA,
        "PTB": profile.PTB,
        "required_internal_mode": profile.required_internal_mode,
        "min_battery_before_jump": profile.min_battery_before_jump,
        "min_battery_after_landing": profile.min_battery_after_landing,
        "tunnel_mode": profile.tunnel_mode,
    }
