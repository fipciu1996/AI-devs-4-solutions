"""Solve the AG3NTS firmware task with an OpenRouter tool-calling agent."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT_HINT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_HINT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_HINT))

from devs_utilities.ag3nts import (
    AG3NTS_SHELL_URL,
    AG3NTS_VERIFY_URL,
    submit_task_answer,
)
from devs_utilities.bootstrap import bootstrap_repo
from devs_utilities.files import write_json
from devs_utilities.flags import extract_flag
from devs_utilities.http import HttpRequestError, JsonResponseError, post_json
from devs_utilities.logging import configure_logging, logger as shared_logger
from devs_utilities.openrouter import (
    OpenRouterClient,
    OpenRouterConfig,
    OpenRouterError,
    ToolCall,
)
from repo_env import (
    get_course_api_key,
    get_env,
    get_int_env,
    get_llm_api_key,
    get_llm_base_url,
    get_optional_env,
)


REPO_ROOT = bootstrap_repo(__file__)
logger = shared_logger.bind(component="firmware")


TASK_NAME = "firmware"
DEFAULT_SHELL_URL = AG3NTS_SHELL_URL
DEFAULT_MODEL = get_env("OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
DEFAULT_API_TIMEOUT_SECONDS = (
    get_int_env(
        "FIRMWARE_TIMEOUT_SECONDS",
        get_int_env("AG3NTS_TIMEOUT_SECONDS", 30) or 30,
    )
    or 30
)
DEFAULT_OPENROUTER_TIMEOUT_SECONDS = get_int_env("OPENROUTER_TIMEOUT_SECONDS", 120) or 120
DEFAULT_MAX_STEPS = get_int_env("FIRMWARE_MAX_STEPS", 18) or 18

COOLER_DIR = "/opt/firmware/cooler"
COOLER_BIN = f"{COOLER_DIR}/cooler.bin"
COOLER_GITIGNORE_PATH = f"{COOLER_DIR}/.gitignore"

OUTPUT_DIR = Path(__file__).resolve().parent
LAST_ANSWER_PATH = OUTPUT_DIR / "last_answer.json"
LAST_SESSION_PATH = OUTPUT_DIR / "last_session.json"
LAST_TRANSCRIPT_PATH = OUTPUT_DIR / "last_transcript.json"
LAST_VERIFY_RESPONSE_PATH = OUTPUT_DIR / "last_verify_response.json"
TOOL_MANIFEST_PATH = OUTPUT_DIR / "shell_tools.json"

CONFIRMATION_PATTERN = re.compile(r"(ECCS-[a-f0-9]{40})", re.IGNORECASE)
EXPECTED_COMMAND_PREFIXES = (
    "help",
    "ls",
    "cat",
    "cd",
    "pwd",
    "rm",
    "editline",
    "reboot",
    "date",
    "uptime",
    "find",
    "history",
    "whoami",
)
FORBIDDEN_OPERATOR_SNIPPETS = (";", "|", "&&", "||", "$(", "`")
RESTRICTED_SYSTEM_PREFIXES = ("/etc", "/root", "/proc")


class FirmwareError(RuntimeError):
    """Base runtime error for the firmware task."""


@dataclass(frozen=True, slots=True)
class ShellBanError(FirmwareError):
    """Raised when the shell API reports a temporary ban."""

    command: str
    reason: str
    seconds_left: int
    reboot_requested: bool

    def __str__(self) -> str:
        return (
            f"Shell API ban for `{self.command}`: {self.reason} "
            f"(wait {self.seconds_left}s, reboot_requested={self.reboot_requested})"
        )


@dataclass(frozen=True, slots=True)
class ShellCommandRecord:
    """One remote shell command with its parsed response."""

    command: str
    response: Any


@dataclass(slots=True)
class AgentState:
    """Mutable state shared across tool calls."""

    final_flag: str | None = None
    final_response: Any = None
    last_confirmation: str | None = None


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Runtime configuration for the firmware agent."""

    api_key: str
    verify_url: str
    shell_url: str
    openrouter_api_key: str
    openrouter_url: str
    model: str
    api_timeout_seconds: int
    openrouter_timeout_seconds: int
    site_url: str | None
    site_name: str | None
    verify: bool
    reboot_first: bool
    max_steps: int
    show_tool_results: bool
    password_hint: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Submit the recovered confirmation code to /verify.",
    )
    parser.add_argument(
        "--skip-reboot",
        action="store_true",
        help="Do not reboot the virtual filesystem before starting the agent.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            f"OpenRouter model override. Defaults to OPENROUTER_MODEL or {DEFAULT_MODEL}."
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"Maximum OpenRouter tool-calling rounds. Default: {DEFAULT_MAX_STEPS}.",
    )
    parser.add_argument(
        "--show-tool-results",
        action="store_true",
        help="Print each OpenRouter tool call and tool result.",
    )
    parser.add_argument(
        "--password-hint",
        default=None,
        help="Optional hint to include in the agent prompt.",
    )
    parser.add_argument(
        "--transcript-path",
        default=str(LAST_TRANSCRIPT_PATH),
        help="Where to save the OpenRouter message transcript.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> AppConfig:
    api_key = get_course_api_key()
    shell_url = DEFAULT_SHELL_URL
    openrouter_api_key = get_llm_api_key()
    openrouter_url = get_llm_base_url()
    model = (args.model or DEFAULT_MODEL).strip()

    api_timeout_raw = (
        get_optional_env("FIRMWARE_TIMEOUT_SECONDS")
        or get_optional_env("AG3NTS_TIMEOUT_SECONDS")
        or str(DEFAULT_API_TIMEOUT_SECONDS)
    )
    try:
        api_timeout_seconds = max(10, int(api_timeout_raw))
    except ValueError as exc:
        raise SystemExit(
            "FIRMWARE_TIMEOUT_SECONDS/AG3NTS_TIMEOUT_SECONDS must be an integer, "
            f"got: {api_timeout_raw}"
        ) from exc
    try:
        openrouter_timeout_seconds = max(10, int(DEFAULT_OPENROUTER_TIMEOUT_SECONDS))
    except ValueError as exc:
        raise SystemExit(
            "OPENROUTER_TIMEOUT_SECONDS must be an integer, "
            f"got: {DEFAULT_OPENROUTER_TIMEOUT_SECONDS}"
        ) from exc

    missing: list[str] = []
    if not api_key:
        missing.append("COURSE_API_KEY")
    if not openrouter_api_key:
        missing.append("LLM_API_KEY")
    if not openrouter_url:
        missing.append("LLM_BASE_URL")
    if not model:
        missing.append("OPENROUTER_MODEL")
    if args.max_steps < 1:
        raise SystemExit("--max-steps must be a positive integer.")

    if missing:
        raise SystemExit(f"Missing required settings: {', '.join(missing)}")

    site_url = get_optional_env("OPENROUTER_SITE_URL") or get_optional_env(
        "OPENROUTER_APP_URL"
    )
    site_name = get_optional_env("OPENROUTER_SITE_NAME") or get_optional_env(
        "OPENROUTER_APP_TITLE"
    )
    password_hint = args.password_hint.strip() if args.password_hint else None
    if password_hint == "":
        password_hint = None

    return AppConfig(
        api_key=api_key,
        verify_url=AG3NTS_VERIFY_URL,
        shell_url=shell_url,
        openrouter_api_key=openrouter_api_key,
        openrouter_url=openrouter_url,
        model=model,
        api_timeout_seconds=api_timeout_seconds,
        openrouter_timeout_seconds=openrouter_timeout_seconds,
        site_url=site_url,
        site_name=site_name,
        verify=args.verify,
        reboot_first=not args.skip_reboot,
        max_steps=args.max_steps,
        show_tool_results=args.show_tool_results,
        password_hint=password_hint,
    )


class ShellClient:
    """Safe wrapper around the firmware shell API."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._history: list[ShellCommandRecord] = []
        self._restricted_exact_paths = {
            f"{COOLER_DIR}/.env",
            f"{COOLER_DIR}/storage.cfg",
        }
        self._restricted_prefix_paths = set(RESTRICTED_SYSTEM_PREFIXES)
        self._allowed_commands = set(EXPECTED_COMMAND_PREFIXES) | {COOLER_BIN}

    @property
    def history(self) -> list[ShellCommandRecord]:
        return list(self._history)

    def add_gitignore_entries(self, entries: list[str]) -> None:
        for entry in entries:
            normalized = entry.strip()
            if not normalized:
                continue
            if normalized.endswith("/"):
                self._restricted_prefix_paths.add(
                    f"{COOLER_DIR}/{normalized.rstrip('/')}"
                )
            else:
                self._restricted_exact_paths.add(f"{COOLER_DIR}/{normalized}")

    def execute(self, command: str) -> dict[str, Any]:
        normalized_command = command.strip()
        self._ensure_safe_command(normalized_command)
        payload = {"apikey": self._config.api_key, "cmd": normalized_command}

        try:
            response = post_json(
                self._config.shell_url,
                payload,
                timeout_seconds=self._config.api_timeout_seconds,
            )
        except HttpRequestError as exc:
            parsed = exc.body_as_json()
            if isinstance(parsed, dict) and isinstance(parsed.get("code"), int):
                self._history.append(
                    ShellCommandRecord(command=normalized_command, response=parsed)
                )
                raise self._to_ban_error(normalized_command, parsed) from exc
            raise FirmwareError(str(exc)) from exc
        except JsonResponseError as exc:
            raise FirmwareError(
                f"Shell API returned invalid JSON for {exc.url}."
            ) from exc

        if not isinstance(response, dict):
            raise FirmwareError("Shell API returned a non-object response.")

        self._history.append(
            ShellCommandRecord(command=normalized_command, response=response)
        )
        return response

    def help(self) -> list[str]:
        response = self.execute("help")
        data = response.get("data")
        if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
            raise FirmwareError("Unexpected help payload from shell API.")
        return data

    def reboot(self) -> dict[str, Any]:
        return self.execute("reboot")

    def read_text(self, path: str) -> str:
        response = self.execute(f"cat {path}")
        data = response.get("data")
        if not isinstance(data, str):
            raise FirmwareError(f"Unexpected file payload for {path}.")
        return data

    def _ensure_safe_command(self, command: str) -> None:
        if not command:
            raise FirmwareError("Refusing to execute an empty remote command.")
        if any(snippet in command for snippet in FORBIDDEN_OPERATOR_SNIPPETS):
            raise FirmwareError(
                "Refusing to execute a command containing shell control operators."
            )

        parts = command.split()
        command_name = parts[0]
        if command_name not in self._allowed_commands:
            raise FirmwareError(f"Unsupported remote command: {command_name}")

        target_path: str | None = None
        if command_name == COOLER_BIN:
            target_path = command_name
        elif command_name in {"ls", "cat", "cd", "rm"} and len(parts) >= 2:
            target_path = parts[1]
        elif command_name == "editline" and len(parts) >= 2:
            target_path = parts[1]

        if target_path and self._is_restricted_path(target_path):
            raise FirmwareError(f"Refusing to touch restricted path: {target_path}")

    def _is_restricted_path(self, path: str) -> bool:
        normalized = path.rstrip("/") or "/"
        if normalized in self._restricted_exact_paths:
            return True
        return any(
            normalized == prefix or normalized.startswith(f"{prefix}/")
            for prefix in self._restricted_prefix_paths
        )

    def _to_ban_error(self, command: str, payload: dict[str, Any]) -> ShellBanError:
        ban_info = payload.get("ban")
        ban_payload = ban_info if isinstance(ban_info, dict) else {}
        seconds_raw = ban_payload.get("seconds_left", ban_payload.get("ttl_seconds", 0))
        try:
            seconds_left = int(seconds_raw)
        except (TypeError, ValueError):
            seconds_left = 0
        return ShellBanError(
            command=command,
            reason=str(ban_payload.get("reason") or payload.get("message") or "ban"),
            seconds_left=max(0, seconds_left),
            reboot_requested=bool(payload.get("reboot")),
        )


class VerifyClient:
    """Submission helper for the AG3NTS verify endpoint."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def submit_confirmation(self, confirmation: str) -> Any:
        return submit_task_answer(
            self._config.verify_url,
            api_key=self._config.api_key,
            task=TASK_NAME,
            answer={"confirmation": confirmation},
            timeout_seconds=self._config.api_timeout_seconds,
        )


def parse_gitignore_entries(text: str) -> list[str]:
    entries: list[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        entries.append(stripped)
    return entries


def ensure_expected_help(help_lines: list[str]) -> None:
    missing = [
        prefix
        for prefix in EXPECTED_COMMAND_PREFIXES
        if not any(line.startswith(prefix) for line in help_lines)
    ]
    if missing:
        raise FirmwareError(
            "Shell help is missing required commands: " + ", ".join(sorted(missing))
        )


def extract_confirmation(text: str) -> str | None:
    match = CONFIRMATION_PATTERN.search(text)
    if not match:
        return None
    return match.group(1)


def build_tool_manifest(
    help_lines: list[str],
    gitignore_entries: list[str],
    *,
    verify_enabled: bool,
) -> dict[str, Any]:
    shell_commands: list[dict[str, str]] = []
    for raw_line in help_lines:
        if not isinstance(raw_line, str) or " - " not in raw_line:
            continue
        command_spec, description = raw_line.split(" - ", 1)
        shell_commands.append(
            {
                "remote_command": command_spec,
                "description": description.strip(),
            }
        )

    return {
        "task": TASK_NAME,
        "shell_commands": shell_commands,
        "gitignore_entries": gitignore_entries,
        "openrouter_tools": [
            {
                "name": "run_shell_command",
                "description": "Execute one raw command on the restricted firmware shell.",
            },
            {
                "name": "submit_confirmation",
                "description": (
                    "Submit the recovered ECCS confirmation to /verify."
                    if verify_enabled
                    else "Store the recovered ECCS confirmation locally."
                ),
            },
        ],
    }


def build_openrouter_tools(*, verify_enabled: bool) -> list[dict[str, Any]]:
    submit_description = (
        "Submit the recovered ECCS confirmation to the AG3NTS verify endpoint. "
        "Use it immediately after you see an ECCS-... code."
        if verify_enabled
        else "Store the recovered ECCS confirmation locally and stop after that."
    )
    return [
        {
            "type": "function",
            "function": {
                "name": "run_shell_command",
                "description": (
                    "Execute one raw command through the restricted firmware shell API. "
                    "Supported commands come from help, plus direct execution of "
                    "/opt/firmware/cooler/cooler.bin."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": (
                                "A single shell command such as `help`, "
                                "`cat /home/operator/notes/pass.txt`, "
                                "`editline /opt/firmware/cooler/settings.ini 2 SAFETY_CHECK=pass`, "
                                "or `/opt/firmware/cooler/cooler.bin admin1`."
                            ),
                        }
                    },
                    "required": ["command"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "submit_confirmation",
                "description": submit_description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "confirmation": {
                            "type": "string",
                            "description": "The recovered ECCS confirmation code.",
                        }
                    },
                    "required": ["confirmation"],
                    "additionalProperties": False,
                },
            },
        },
    ]


def build_system_prompt(
    help_lines: list[str],
    gitignore_entries: list[str],
    *,
    verify_enabled: bool,
    password_hint: str | None,
) -> str:
    help_text = "\n".join(f"- {line}" for line in help_lines)
    gitignore_text = "\n".join(f"- {entry}" for entry in gitignore_entries)
    verification_line = (
        "Stop only after submit_confirmation returns a flag."
        if verify_enabled
        else "Stop after submit_confirmation stores the confirmation locally."
    )
    hint_line = (
        f"User supplied password hint: {password_hint}\n"
        if password_hint
        else ""
    )
    return (
        "You are a firmware recovery agent working through tools only.\n\n"
        "Goal:\n"
        "- recover the password needed by /opt/firmware/cooler/cooler.bin\n"
        "- fix the configuration so the cooler can start\n"
        "- run the binary and recover the ECCS confirmation code\n"
        "- submit that confirmation with the submit_confirmation tool\n\n"
        "Hard rules:\n"
        "- Never touch /etc, /root, or /proc.\n"
        "- Respect /opt/firmware/cooler/.gitignore exactly.\n"
        "- If a command returns a refusal or ban, do not repeat that command.\n"
        "- After any reboot, assume previous filesystem edits are gone.\n"
        "- Prefer absolute paths.\n"
        "- Useful safe clues often live in shell history, /home/operator, "
        "and /opt/firmware/cooler/settings.ini.\n"
        "- If you change settings.ini, only edit specific lines with editline.\n"
        "- If a lock file blocks startup, fix the root cause first and then remove the lock.\n"
        f"- {verification_line}\n\n"
        "Fresh shell help:\n"
        f"{help_text}\n\n"
        "Entries ignored by /opt/firmware/cooler/.gitignore:\n"
        f"{gitignore_text}\n\n"
        f"{hint_line}"
        "Be concise. Think with tools."
    )


def build_initial_user_prompt(*, rebooted: bool) -> str:
    reboot_line = (
        "The VM has already been rebooted once, so you are starting from a clean state."
        if rebooted
        else "The VM has not been rebooted for you yet."
    )
    return (
        "Solve the firmware task through the shell tool. "
        "Inspect the safe clues, repair the configuration, run cooler.bin, "
        "and submit the ECCS confirmation.\n\n"
        f"{reboot_line}"
    )


def handle_shell_ban(error: ShellBanError, shell: ShellClient) -> dict[str, Any]:
    wait_seconds = max(1, error.seconds_left + 1)
    logger.warning("Shell ban encountered: {}", error)
    logger.info("Waiting {}s for the ban to expire.", wait_seconds)
    time.sleep(wait_seconds)

    rebooted = False
    reboot_response: Any = None
    if error.reboot_requested:
        logger.info("Rebooting the virtual filesystem after the ban.")
        reboot_response = shell.reboot()
        rebooted = True

    return {
        "ok": False,
        "error": str(error),
        "command": error.command,
        "ban_reason": error.reason,
        "waited_seconds": wait_seconds,
        "vm_rebooted": rebooted,
        "reboot_response": reboot_response,
        "agent_guidance": (
            "Avoid the banned command. If the VM was rebooted, re-apply any needed "
            "safe configuration changes before continuing."
        ),
    }


def build_tool_handlers(
    *,
    config: AppConfig,
    shell: ShellClient,
    verify_client: VerifyClient,
    state: AgentState,
) -> dict[str, Any]:
    def run_shell_command(arguments: dict[str, Any]) -> Any:
        command = str(arguments.get("command", "")).strip()
        if not command:
            return {
                "ok": False,
                "error": "Missing required `command` argument.",
            }
        try:
            response = shell.execute(command)
        except ShellBanError as exc:
            return handle_shell_ban(exc, shell)
        except FirmwareError as exc:
            return {
                "ok": False,
                "command": command,
                "error": str(exc),
            }

        return {
            "ok": True,
            "command": command,
            "response": response,
        }

    def submit_confirmation(arguments: dict[str, Any]) -> Any:
        raw_confirmation = str(arguments.get("confirmation", "")).strip()
        confirmation = extract_confirmation(raw_confirmation)
        if confirmation is None:
            return {
                "ok": False,
                "error": (
                    "The provided confirmation does not match the expected "
                    "ECCS-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx format."
                ),
                "provided_confirmation": raw_confirmation,
            }

        state.last_confirmation = confirmation
        payload = {
            "apikey": config.api_key,
            "task": TASK_NAME,
            "answer": {"confirmation": confirmation},
        }
        write_json(LAST_ANSWER_PATH, payload)

        if not config.verify:
            response = {
                "ok": True,
                "verification_skipped": True,
                "confirmation": confirmation,
            }
            state.final_response = response
            write_json(LAST_VERIFY_RESPONSE_PATH, response)
            return response

        try:
            response = verify_client.submit_confirmation(confirmation)
        except HttpRequestError as exc:
            response = exc.to_response_dict()
            state.final_response = response
            write_json(LAST_VERIFY_RESPONSE_PATH, response)
            return response

        state.final_response = response
        write_json(LAST_VERIFY_RESPONSE_PATH, response)
        flag = extract_flag(response)
        if flag:
            state.final_flag = flag
        return response

    return {
        "run_shell_command": run_shell_command,
        "submit_confirmation": submit_confirmation,
    }


def execute_tool_call(
    tool_call: ToolCall,
    handlers: dict[str, Any],
    *,
    show_tool_results: bool,
) -> dict[str, Any]:
    if tool_call.name not in handlers:
        raise FirmwareError(f"Model called an unknown tool: {tool_call.name}")

    result = handlers[tool_call.name](tool_call.arguments)
    if show_tool_results:
        logger.info(
            "Tool {} args:\n{}\nTool result:\n{}",
            tool_call.name,
            json.dumps(tool_call.arguments, ensure_ascii=False, indent=2),
            json.dumps(result, ensure_ascii=False, indent=2),
        )

    return {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "name": tool_call.name,
        "content": json.dumps(result, ensure_ascii=False),
    }


def build_session_payload(
    *,
    config: AppConfig,
    help_lines: list[str],
    gitignore_entries: list[str],
    state: AgentState,
    shell: ShellClient,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "task": TASK_NAME,
        "model": config.model,
        "verify_enabled": config.verify,
        "help": help_lines,
        "gitignore_entries": gitignore_entries,
        "last_confirmation": state.last_confirmation,
        "final_response": state.final_response,
        "final_flag": state.final_flag,
        "shell_transcript": [
            {"command": item.command, "response": item.response}
            for item in shell.history
        ],
        "agent_messages": messages,
    }


def maybe_finish_from_plain_text(
    *,
    completion_content: str | None,
    handlers: dict[str, Any],
    state: AgentState,
    config: AppConfig,
) -> bool:
    text = (completion_content or "").strip()
    if not text:
        return False

    flag = extract_flag(text)
    if flag:
        state.final_flag = flag
        return True

    confirmation = extract_confirmation(text)
    if confirmation is None or state.last_confirmation is not None:
        return False

    logger.warning(
        "Model returned a confirmation in plain text without using the tool. "
        "Submitting it automatically."
    )
    response = handlers["submit_confirmation"]({"confirmation": confirmation})
    auto_flag = extract_flag(response)
    if auto_flag:
        state.final_flag = auto_flag
    return (not config.verify and state.last_confirmation is not None) or (
        state.final_flag is not None
    )


def run_agent(
    *,
    config: AppConfig,
    transcript_path: Path,
) -> tuple[AgentState, dict[str, Any]]:
    shell = ShellClient(config)
    if config.reboot_first:
        logger.info("Rebooting the virtual filesystem before starting the agent.")
        shell.reboot()

    help_lines = shell.help()
    ensure_expected_help(help_lines)
    gitignore_entries = parse_gitignore_entries(shell.read_text(COOLER_GITIGNORE_PATH))
    shell.add_gitignore_entries(gitignore_entries)
    write_json(
        TOOL_MANIFEST_PATH,
        build_tool_manifest(
            help_lines,
            gitignore_entries,
            verify_enabled=config.verify,
        ),
    )

    client = OpenRouterClient(
        OpenRouterConfig(
            api_key=config.openrouter_api_key,
            base_url=config.openrouter_url,
            model=config.model,
            timeout_seconds=config.openrouter_timeout_seconds,
            site_url=config.site_url,
            site_name=config.site_name,
        )
    )
    verify_client = VerifyClient(config)
    state = AgentState()
    handlers = build_tool_handlers(
        config=config,
        shell=shell,
        verify_client=verify_client,
        state=state,
    )
    tools = build_openrouter_tools(verify_enabled=config.verify)
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": build_system_prompt(
                help_lines,
                gitignore_entries,
                verify_enabled=config.verify,
                password_hint=config.password_hint,
            ),
        },
        {
            "role": "user",
            "content": build_initial_user_prompt(rebooted=config.reboot_first),
        },
    ]

    try:
        for _step in range(config.max_steps):
            completion = client.create_completion(messages, tools=tools)
            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": completion.content or "",
            }
            if completion.tool_calls:
                assistant_message["tool_calls"] = [
                    tool_call.to_message_dict() for tool_call in completion.tool_calls
                ]
            messages.append(assistant_message)

            if completion.tool_calls:
                for tool_call in completion.tool_calls:
                    tool_message = execute_tool_call(
                        tool_call,
                        handlers,
                        show_tool_results=config.show_tool_results,
                    )
                    messages.append(tool_message)
                    if state.final_flag:
                        break
                    if not config.verify and state.last_confirmation:
                        break

                if state.final_flag or (not config.verify and state.last_confirmation):
                    break
                continue

            if maybe_finish_from_plain_text(
                completion_content=completion.content,
                handlers=handlers,
                state=state,
                config=config,
            ):
                break

        else:
            raise FirmwareError(
                f"Agent did not finish within {config.max_steps} tool-calling rounds."
            )
    finally:
        write_json(transcript_path, messages)
        write_json(
            LAST_SESSION_PATH,
            build_session_payload(
                config=config,
                help_lines=help_lines,
                gitignore_entries=gitignore_entries,
                state=state,
                shell=shell,
                messages=messages,
            ),
        )

    return state, build_session_payload(
        config=config,
        help_lines=help_lines,
        gitignore_entries=gitignore_entries,
        state=state,
        shell=shell,
        messages=messages,
    )


def main() -> int:
    args = parse_args()
    config = build_config(args)
    transcript_path = Path(args.transcript_path).resolve()
    configure_logging(name="firmware", verbose=config.show_tool_results)

    try:
        state, _session = run_agent(config=config, transcript_path=transcript_path)
    except (FirmwareError, OpenRouterError) as exc:
        logger.error("Error: {}", exc)
        return 1

    logger.info("Model: {}", config.model)
    if state.last_confirmation:
        logger.success("Confirmation: {}", state.last_confirmation)
    if state.final_response is not None:
        logger.info(
            "Verify response:\n{}",
            json.dumps(state.final_response, ensure_ascii=False, indent=2),
        )
    if state.final_flag:
        logger.success("Flag: {}", state.final_flag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
