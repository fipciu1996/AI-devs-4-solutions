"""OpenRouter-backed frontend agent for the timetravel task."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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

from timetravel.frontend_agent import (
    FrontendAgent,
    UI_SNAPSHOT_SCRIPT,
    merge_observed_ui,
    should_click_orb,
)
from timetravel.models import MissionStatus
from timetravel.openrouter_runtime import ToolCallingAgentConfig, run_tool_calling_agent


TASK_NAME = "timetravel-frontend"
DEFAULT_MODEL = (
    get_optional_env("TIMETRAVEL_FRONTEND_MODEL")
    or get_optional_env("TIMETRAVEL_MODEL")
    or get_llm_model()
)
DEFAULT_OPENROUTER_TIMEOUT_SECONDS = get_int_env("OPENROUTER_TIMEOUT_SECONDS", 120) or 120
DEFAULT_MAX_STEPS = get_int_env("TIMETRAVEL_FRONTEND_MAX_STEPS", 64) or 64

SYSTEM_PROMPT = """
You are the frontend timetravel agent controlling the timetravel preview UI.
You do not use the system browser profile. You operate only through safe tools.

Workflow:
1. inspect_state
2. ensure_preview_ready
3. read_ui_state
4. if desired_ui and observed_ui differ, call apply_desired_ui_step until it returns noop
5. only when mission_status is READY_FOR_JUMP, call consume_jump_if_ready
6. persist_session after meaningful progress

Rules:
- Use tools only.
- Never try to click the orb manually through reasoning; use consume_jump_if_ready.
- Do not keep calling the same noop tool repeatedly without re-reading state.
- Stop only when mission_state is COMPLETED or FAILED.
""".strip()

INITIAL_USER_PROMPT = (
    "Start the frontend timetravel mission. Inspect shared state, make the preview ready, "
    "keep UI controls aligned with desired_ui, and consume jumps only through the safe tool."
)


OPENROUTER_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "inspect_state",
            "description": "Read the shared mission state, desired_ui, observed_ui, backend snapshot, and recent events.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ensure_preview_ready",
            "description": "Launch isolated Playwright Chromium if needed, open timetravel preview, and wait until the page is ready.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_ui_state",
            "description": "Read the current preview UI state, publish it to the shared store, and return the normalized snapshot.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_desired_ui_step",
            "description": "Apply at most one non-jump UI correction towards desired_ui.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "consume_jump_if_ready",
            "description": "Click the orb only if mission_status is READY_FOR_JUMP and the current jumpRequest was not consumed yet.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "persist_session",
            "description": "Save the current Playwright storage_state to disk.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
]


@dataclass(slots=True)
class FrontendAgentState:
    launch: Any | None = None
    playwright_context_manager: Any | None = None
    playwright: Any | None = None


class OpenRouterFrontendAgent:
    """Tool-calling frontend orchestration over the safe preview controls."""

    def __init__(
        self,
        *,
        controller: FrontendAgent,
        transcript_path: Path,
        model: str,
        timeout_seconds: int,
        max_steps: int,
        show_tool_results: bool,
    ) -> None:
        self.controller = controller
        self.transcript_path = transcript_path
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_steps = max_steps
        self.show_tool_results = show_tool_results

    def run(self) -> None:
        state = FrontendAgentState()
        sync_api = self.controller._import_playwright_sync_api()
        state.playwright_context_manager = sync_api.sync_playwright()
        state.playwright = state.playwright_context_manager.__enter__()
        self.controller._last_consumed_jump_id = self.controller.store.get_observed_ui().last_consumed_jump_id
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
            continue_prompt="Re-read shared state, then use one focused tool.",
        )
        try:
            run_tool_calling_agent(
                client=client,
                config=config,
                tool_handlers=handlers,
                finish_predicate=self._is_finished,
            )
        finally:
            self._shutdown(state)

    def _build_client(self) -> OpenRouterClient:
        api_key = get_llm_api_key().strip()
        base_url = get_llm_base_url().strip()
        if not api_key:
            raise RuntimeError("Missing OpenRouter API key. Set LLM_API_KEY or OPENROUTER_API_KEY.")
        if not base_url:
            raise RuntimeError("Missing OpenRouter base URL. Set LLM_BASE_URL or OPENROUTER_BASE_URL.")
        if not self.model:
            raise RuntimeError(
                "Missing OpenRouter model. Set TIMETRAVEL_FRONTEND_MODEL, TIMETRAVEL_MODEL, "
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

    def _build_tool_handlers(self, state: FrontendAgentState) -> dict[str, Any]:
        def inspect_state(_: dict[str, Any]) -> dict[str, Any]:
            return {
                "ok": True,
                "browser_ready": state.launch is not None,
                "mission_state": self.controller.store.get_mission_state(),
                "desired_ui": asdict(self.controller.store.get_desired_ui()),
                "observed_ui": asdict(self.controller.store.get_observed_ui()),
                "backend_snapshot": self.controller.store.get_backend_snapshot(),
                "recent_events": self.controller.store.get_recent_events(limit=12),
            }

        def ensure_preview_ready(_: dict[str, Any]) -> dict[str, Any]:
            if state.launch is None:
                state.launch = self.controller._launch_page(state.playwright)
                self.controller._wait_for_ready_page(state.launch.page)
                self.controller._persist_storage_state(state.launch.context, reason="preview became ready")
            return {
                "ok": True,
                "browser_ready": True,
                "page_url": self.controller._safe_page_url(state.launch.page),
                "source": state.launch.source,
            }

        def read_ui_state(_: dict[str, Any]) -> dict[str, Any]:
            launch = _require_launch(state)
            desired = self.controller.store.get_desired_ui()
            mission_state = self.controller.store.get_mission_state()
            self.controller._log_received_desired_if_changed(desired, mission_state.get("phase"))
            observed = self._read_and_publish_observed(launch.page, phase=mission_state.get("phase"))
            return {
                "ok": True,
                "observed_ui": asdict(observed),
                "page_url": self.controller._safe_page_url(launch.page),
            }

        def apply_desired_ui_step(_: dict[str, Any]) -> dict[str, Any]:
            launch = _require_launch(state)
            desired = self.controller.store.get_desired_ui()
            mission_state = self.controller.store.get_mission_state()
            observed = self._read_and_publish_observed(launch.page, phase=mission_state.get("phase"))
            action = self._apply_non_jump_ui_step(launch.page, desired=desired, observed=observed)
            refreshed = self._read_and_publish_observed(launch.page, phase=mission_state.get("phase"))
            return {
                "ok": True,
                "action": action or "noop",
                "observed_ui": asdict(refreshed),
            }

        def consume_jump_if_ready(_: dict[str, Any]) -> dict[str, Any]:
            launch = _require_launch(state)
            desired = self.controller.store.get_desired_ui()
            mission_state = self.controller.store.get_mission_state()
            observed = self._read_and_publish_observed(launch.page, phase=mission_state.get("phase"))
            if not should_click_orb(
                desired,
                observed,
                mission_status=mission_state.get("status"),
            ):
                return {
                    "ok": True,
                    "clicked": False,
                    "reason": "Jump is not ready yet.",
                    "mission_status": mission_state.get("status"),
                    "observed_ui": asdict(observed),
                }

            launch.page.locator("#orb").click()
            self.controller._last_consumed_jump_id = desired.jump_request_id
            refreshed = self._read_and_publish_observed(launch.page, phase=mission_state.get("phase"))
            self.controller._event("INFO", f"Clicked the orb for jump request {desired.jump_request_id}.")
            return {
                "ok": True,
                "clicked": True,
                "jump_request_id": desired.jump_request_id,
                "observed_ui": asdict(refreshed),
            }

        def persist_session(_: dict[str, Any]) -> dict[str, Any]:
            launch = _require_launch(state)
            self.controller._persist_storage_state(launch.context, reason="tool request")
            return {
                "ok": True,
                "storage_state_path": str(self.controller.storage_state_path),
            }

        return {
            "inspect_state": inspect_state,
            "ensure_preview_ready": ensure_preview_ready,
            "read_ui_state": read_ui_state,
            "apply_desired_ui_step": apply_desired_ui_step,
            "consume_jump_if_ready": consume_jump_if_ready,
            "persist_session": persist_session,
        }

    def _read_and_publish_observed(self, page: Any, *, phase: str | None) -> Any:
        observed = merge_observed_ui(
            self.controller._read_observed_ui(page),
            last_consumed_jump_id=self.controller._last_consumed_jump_id,
        )
        self.controller.store.set_observed_ui(observed)
        self.controller._log_published_observed_if_changed(observed, phase)
        return observed

    def _apply_non_jump_ui_step(self, page: Any, *, desired, observed) -> str | None:
        if desired.mode and observed.mode != desired.mode:
            page.locator("#mainSwitch").click()
            self.controller._event("INFO", f"Applied backend command: switched mode to {desired.mode}.")
            return f"mode:{desired.mode}"

        if desired.PTA is not None and observed.PTA != desired.PTA:
            page.locator("#portA").click()
            self.controller._event("INFO", f"Applied backend command: adjusted PT-A to {desired.PTA}.")
            return f"PT-A:{desired.PTA}"

        if desired.PTB is not None and observed.PTB != desired.PTB:
            if desired.PTA and desired.PTB and observed.battery_level < 2:
                self.controller._event("INFO", "Waiting for battery >= 2/3 before enabling tunnel mode.")
                return "wait:battery"
            page.locator("#portB").click()
            self.controller._event("INFO", f"Applied backend command: adjusted PT-B to {desired.PTB}.")
            return f"PT-B:{desired.PTB}"

        if desired.PWR is not None and observed.PWR != desired.PWR:
            slider = page.locator("#pwrSlider")
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
            self.controller._event("INFO", f"Applied backend command: adjusted PWR to {desired.PWR}.")
            return f"PWR:{desired.PWR}"

        return None

    def _is_finished(self) -> bool:
        mission_state = self.controller.store.get_mission_state()
        return mission_state.get("status") in {
            MissionStatus.COMPLETED.value,
            MissionStatus.FAILED.value,
        }

    def _shutdown(self, state: FrontendAgentState) -> None:
        if state.launch is not None:
            try:
                self.controller._persist_storage_state(state.launch.context, reason="frontend shutdown")
            except Exception:
                pass
            try:
                state.launch.context.close()
            except Exception:
                pass
            try:
                state.launch.browser.close()
            except Exception:
                pass
        if state.playwright_context_manager is not None:
            try:
                state.playwright_context_manager.__exit__(None, None, None)
            except Exception:
                pass


def _require_launch(state: FrontendAgentState) -> Any:
    if state.launch is None:
        raise RuntimeError("Preview is not ready yet. Call ensure_preview_ready first.")
    return state.launch
