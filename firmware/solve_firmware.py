"""Solve the AG3NTS firmware task through the restricted shell API."""

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

from devs_utilities.ag3nts import submit_task_answer
from devs_utilities.bootstrap import bootstrap_repo
from devs_utilities.files import write_json
from devs_utilities.flags import extract_flag
from devs_utilities.http import HttpRequestError, JsonResponseError, post_json
from devs_utilities.logging import configure_logging, logger as shared_logger
from repo_env import get_env, get_optional_env


REPO_ROOT = bootstrap_repo(__file__)
logger = shared_logger.bind(component="firmware")


TASK_NAME = "firmware"
DEFAULT_SHELL_URL = "https://example.invalid/api/shell"
DEFAULT_TIMEOUT_SECONDS = 30

COOLER_DIR = "/opt/firmware/cooler"
COOLER_BIN = f"{COOLER_DIR}/cooler.bin"
SETTINGS_PATH = f"{COOLER_DIR}/settings.ini"
LOCK_PATH = f"{COOLER_DIR}/cooler-is-blocked.lock"
PASSWORD_NOTE_PATH = "/home/operator/notes/pass.txt"

OUTPUT_DIR = Path(__file__).resolve().parent
LAST_ANSWER_PATH = OUTPUT_DIR / "last_answer.json"
LAST_SESSION_PATH = OUTPUT_DIR / "last_session.json"
LAST_VERIFY_RESPONSE_PATH = OUTPUT_DIR / "last_verify_response.json"
TOOL_MANIFEST_PATH = OUTPUT_DIR / "shell_tools.json"

CONFIRMATION_PATTERN = re.compile(r"(ECCS-[a-f0-9]{40})", re.IGNORECASE)
PASSWORD_ATTEMPT_PATTERN = re.compile(
    rf"{re.escape(COOLER_BIN)}\s+([A-Za-z0-9._-]+)"
)

RESTRICTED_PATH_PREFIXES = ("/etc", "/root", "/proc")
RESTRICTED_REMOTE_PATHS = {
    f"{COOLER_DIR}/.env",
    f"{COOLER_DIR}/storage.cfg",
    f"{COOLER_DIR}/logs",
}

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


class FirmwareError(RuntimeError):
    """Base runtime error for the firmware task."""


@dataclass(frozen=True, slots=True)
class ShellBanError(FirmwareError):
    """Raised when the shell API reports a temporary ban."""

    command: str
    reason: str
    seconds_left: int
    reboot_requested: bool
    code: int

    def __str__(self) -> str:
        return (
            f"Shell API ban for `{self.command}`: {self.reason} "
            f"(wait {self.seconds_left}s, reboot_requested={self.reboot_requested})"
        )


@dataclass(frozen=True, slots=True)
class ShellCommandRecord:
    """A single shell command and its parsed response."""

    command: str
    response: Any


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Runtime settings for the firmware solver."""

    api_key: str
    verify_url: str
    shell_url: str
    timeout_seconds: int
    reboot_first: bool
    password_override: str | None
    verify: bool


@dataclass(frozen=True, slots=True)
class RunResult:
    """Successful firmware run details."""

    password: str
    confirmation: str
    launch_output: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Submit the recovered confirmation code to /verify.",
    )
    parser.add_argument(
        "--password",
        default=None,
        help="Optional password override for cooler.bin.",
    )
    parser.add_argument(
        "--skip-reboot",
        action="store_true",
        help="Do not reboot the virtual filesystem before solving.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> AppConfig:
    api_key = get_env("AG3NTS_API_KEY")
    verify_url = get_env("AG3NTS_VERIFY_URL")
    shell_url = get_optional_env("AG3NTS_SHELL_URL") or DEFAULT_SHELL_URL
    timeout_raw = get_optional_env("AG3NTS_TIMEOUT_SECONDS") or str(
        DEFAULT_TIMEOUT_SECONDS
    )

    try:
        timeout_seconds = max(5, int(timeout_raw))
    except ValueError as exc:
        raise SystemExit(
            f"AG3NTS_TIMEOUT_SECONDS must be an integer, got: {timeout_raw}"
        ) from exc

    missing: list[str] = []
    if not api_key:
        missing.append("AG3NTS_API_KEY")
    if not verify_url:
        missing.append("AG3NTS_VERIFY_URL")
    if not shell_url:
        missing.append("AG3NTS_SHELL_URL")

    if missing:
        raise SystemExit(f"Missing required settings: {', '.join(missing)}")

    password_override = args.password.strip() if args.password else None
    if password_override == "":
        password_override = None

    return AppConfig(
        api_key=api_key,
        verify_url=verify_url,
        shell_url=shell_url,
        timeout_seconds=timeout_seconds,
        reboot_first=not args.skip_reboot,
        password_override=password_override,
        verify=args.verify,
    )


def build_tool_manifest(help_lines: list[str]) -> dict[str, Any]:
    """Convert help output into a compact tool manifest for agent use."""

    tools: list[dict[str, str]] = []
    for raw_line in help_lines:
        if not isinstance(raw_line, str) or " - " not in raw_line:
            continue
        command_spec, description = raw_line.split(" - ", 1)
        command_name = command_spec.split(" ", 1)[0]
        tools.append(
            {
                "tool_name": f"shell_{command_name}",
                "remote_command": command_spec,
                "description": description.strip(),
            }
        )
    return {"count": len(tools), "tools": tools}


class ShellClient:
    """Thin wrapper around the firmware shell API with safety rails."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._history: list[ShellCommandRecord] = []

    @property
    def history(self) -> list[ShellCommandRecord]:
        return list(self._history)

    def execute(self, command: str) -> dict[str, Any]:
        self._ensure_safe_command(command)
        payload = {"apikey": self._config.api_key, "cmd": command}
        try:
            response = post_json(
                self._config.shell_url,
                payload,
                timeout_seconds=self._config.timeout_seconds,
            )
        except HttpRequestError as exc:
            parsed = exc.body_as_json()
            if isinstance(parsed, dict) and isinstance(parsed.get("code"), int):
                self._history.append(ShellCommandRecord(command=command, response=parsed))
                raise self._to_ban_error(command, parsed) from exc
            raise FirmwareError(str(exc)) from exc
        except JsonResponseError as exc:
            raise FirmwareError(
                f"Shell API returned invalid JSON for {exc.url}."
            ) from exc

        if not isinstance(response, dict):
            raise FirmwareError("Shell API returned a non-object response.")

        self._history.append(ShellCommandRecord(command=command, response=response))
        return response

    def help(self) -> list[str]:
        response = self.execute("help")
        data = response.get("data")
        if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
            raise FirmwareError("Unexpected help payload from shell API.")
        return data

    def reboot(self) -> dict[str, Any]:
        return self.execute("reboot")

    def history_entries(self) -> list[str]:
        response = self.execute("history")
        data = response.get("data")
        if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
            raise FirmwareError("Unexpected history payload from shell API.")
        return data

    def list_dir(self, path: str) -> list[str]:
        response = self.execute(f"ls {path}")
        data = response.get("data")
        if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
            raise FirmwareError(f"Unexpected directory listing payload for {path}.")
        return data

    def read_text(self, path: str) -> str:
        response = self.execute(f"cat {path}")
        data = response.get("data")
        if not isinstance(data, str):
            raise FirmwareError(f"Unexpected file payload for {path}.")
        return data

    def edit_line(self, path: str, line_number: int, content: str) -> dict[str, Any]:
        if line_number < 1:
            raise FirmwareError("edit_line requires a positive line number.")
        return self.execute(f"editline {path} {line_number} {content}")

    def remove_file(self, path: str) -> dict[str, Any]:
        return self.execute(f"rm {path}")

    def run_binary(self, password: str) -> str:
        response = self.execute(f"{COOLER_BIN} {password}")
        data = response.get("data")
        if not isinstance(data, str):
            raise FirmwareError("Unexpected executable output payload.")
        return data

    def _ensure_safe_command(self, command: str) -> None:
        parts = command.split()
        if not parts:
            raise FirmwareError("Refusing to execute an empty remote command.")

        command_name = parts[0]
        target_path: str | None = None
        if command_name in {"ls", "cat", "cd", "rm"} and len(parts) >= 2:
            target_path = parts[1]
        elif command_name == "editline" and len(parts) >= 2:
            target_path = parts[1]

        if target_path and self._is_restricted_path(target_path):
            raise FirmwareError(f"Refusing to touch restricted path: {target_path}")

    def _is_restricted_path(self, path: str) -> bool:
        normalized = path.rstrip("/") or "/"
        if normalized in RESTRICTED_REMOTE_PATHS:
            return True
        return any(
            normalized == prefix or normalized.startswith(f"{prefix}/")
            for prefix in RESTRICTED_PATH_PREFIXES
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
            code=int(payload.get("code", -1)),
        )


def handle_shell_ban(error: ShellBanError, shell: ShellClient) -> None:
    """Sleep out an existing ban, then reboot to reset the VM state."""

    logger.warning("Shell ban encountered: {}", error)
    wait_seconds = max(1, error.seconds_left + 1)
    logger.info("Waiting {}s for the ban to expire.", wait_seconds)
    time.sleep(wait_seconds)
    if error.reboot_requested:
        logger.info("Rebooting the virtual filesystem after the ban.")
        shell.reboot()


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


def parse_settings_lines(settings_text: str) -> dict[tuple[str, str], int]:
    """Map `(section, key)` to 1-based line numbers in settings.ini."""

    line_map: dict[tuple[str, str], int] = {}
    current_section = ""
    for line_number, raw_line in enumerate(settings_text.splitlines(), start=1):
        stripped = raw_line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current_section = stripped[1:-1].strip()
            continue
        if "=" not in stripped:
            continue
        key = stripped.split("=", 1)[0].lstrip("#").strip()
        line_map[(current_section, key)] = line_number
    return line_map


def ensure_correct_settings(shell: ShellClient) -> dict[str, Any]:
    """Fix the known bad settings in cooler/settings.ini."""

    settings_text = shell.read_text(SETTINGS_PATH)
    current_lines = settings_text.splitlines()
    line_map = parse_settings_lines(settings_text)

    desired_lines: dict[tuple[str, str], str] = {
        ("main", "SAFETY_CHECK"): "SAFETY_CHECK=pass",
        ("test_mode", "enabled"): "enabled=false",
        ("cooling", "enabled"): "enabled=true",
    }

    changed: list[dict[str, Any]] = []
    for key, desired_text in desired_lines.items():
        line_number = line_map.get(key)
        if line_number is None:
            section, setting = key
            raise FirmwareError(
                f"Cannot find `{setting}` inside the `{section}` section of settings.ini."
            )
        current_text = current_lines[line_number - 1].strip()
        if current_text == desired_text:
            continue
        shell.edit_line(SETTINGS_PATH, line_number, desired_text)
        current_lines[line_number - 1] = desired_text
        changed.append(
            {
                "section": key[0],
                "key": key[1],
                "line_number": line_number,
                "before": current_text,
                "after": desired_text,
            }
        )

    return {
        "changed": changed,
        "final_preview": "\n".join(current_lines),
    }


def collect_password_candidates(
    shell: ShellClient, password_override: str | None
) -> list[str]:
    """Gather likely passwords from safe places in the VM state."""

    candidates: list[str] = []

    def add_candidate(value: str) -> None:
        normalized = value.strip()
        if normalized and normalized not in candidates:
            candidates.append(normalized)

    if password_override:
        add_candidate(password_override)

    note_password = shell.read_text(PASSWORD_NOTE_PATH).strip()
    if note_password:
        add_candidate(note_password)

    for entry in shell.history_entries():
        match = PASSWORD_ATTEMPT_PATTERN.search(entry)
        if not match:
            continue
        candidate = match.group(1).strip()
        if candidate.startswith("-"):
            continue
        add_candidate(candidate)

    if not candidates:
        raise FirmwareError("No password candidates recovered from the VM.")

    return candidates


def remove_lock_if_present(shell: ShellClient) -> bool:
    listing = shell.list_dir(COOLER_DIR)
    if "cooler-is-blocked.lock" not in listing:
        return False
    shell.remove_file(LOCK_PATH)
    return True


def launch_cooler(shell: ShellClient, passwords: list[str]) -> RunResult:
    """Try password candidates until cooler.bin prints the confirmation code."""

    last_output = ""
    for password in passwords:
        output = shell.run_binary(password)
        last_output = output
        confirmation = extract_confirmation(output)
        if confirmation:
            return RunResult(
                password=password,
                confirmation=confirmation,
                launch_output=output,
            )

        if "Lock file exists" in output:
            logger.warning("Detected a stale lock file. Removing it and retrying.")
            remove_lock_if_present(shell)
            retry_output = shell.run_binary(password)
            last_output = retry_output
            confirmation = extract_confirmation(retry_output)
            if confirmation:
                return RunResult(
                    password=password,
                    confirmation=confirmation,
                    launch_output=retry_output,
                )

    raise FirmwareError(
        "cooler.bin did not return an ECCS confirmation code. "
        f"Last output:\n{last_output}"
    )


def extract_confirmation(output: str) -> str | None:
    match = CONFIRMATION_PATTERN.search(output)
    if not match:
        return None
    return match.group(1)


def solve_firmware(config: AppConfig) -> tuple[RunResult, dict[str, Any]]:
    shell = ShellClient(config)
    if config.reboot_first:
        logger.info("Rebooting the virtual filesystem before solving.")
        shell.reboot()

    try:
        help_lines = shell.help()
    except ShellBanError as exc:
        handle_shell_ban(exc, shell)
        help_lines = shell.help()

    ensure_expected_help(help_lines)
    tool_manifest = build_tool_manifest(help_lines)
    write_json(TOOL_MANIFEST_PATH, tool_manifest)

    try:
        settings_report = ensure_correct_settings(shell)
        password_candidates = collect_password_candidates(
            shell, config.password_override
        )
        lock_removed = remove_lock_if_present(shell)
        run_result = launch_cooler(shell, password_candidates)
    except ShellBanError as exc:
        handle_shell_ban(exc, shell)
        raise FirmwareError(
            "The solver hit a shell ban during a supposedly safe operation. "
            "The remote VM was rebooted; rerun the script."
        ) from exc

    session_payload = {
        "help": help_lines,
        "settings_report": settings_report,
        "password_candidates": password_candidates,
        "lock_removed_before_run": lock_removed,
        "run_result": {
            "password": run_result.password,
            "confirmation": run_result.confirmation,
            "launch_output": run_result.launch_output,
        },
        "transcript": [
            {"command": item.command, "response": item.response}
            for item in shell.history
        ],
    }
    write_json(LAST_SESSION_PATH, session_payload)
    return run_result, session_payload


def verify_confirmation(config: AppConfig, confirmation: str) -> Any:
    answer = {"confirmation": confirmation}
    write_json(
        LAST_ANSWER_PATH,
        {
            "apikey": config.api_key,
            "task": TASK_NAME,
            "answer": answer,
        },
    )

    response = submit_task_answer(
        config.verify_url,
        api_key=config.api_key,
        task=TASK_NAME,
        answer=answer,
        timeout_seconds=config.timeout_seconds,
    )
    write_json(LAST_VERIFY_RESPONSE_PATH, response)
    return response


def main() -> int:
    args = parse_args()
    configure_logging(name="firmware")
    config = build_config(args)

    try:
        run_result, _session = solve_firmware(config)
    except FirmwareError as exc:
        logger.error("Error: {}", exc)
        return 1

    logger.info("Recovered password candidate: {}", run_result.password)
    logger.success("Confirmation: {}", run_result.confirmation)

    if not config.verify:
        return 0

    try:
        verify_response = verify_confirmation(config, run_result.confirmation)
    except HttpRequestError as exc:
        payload = exc.to_response_dict()
        write_json(LAST_VERIFY_RESPONSE_PATH, payload)
        logger.error("Verify request failed: {}", json.dumps(payload, ensure_ascii=False))
        return 1

    logger.info(
        "Verify response:\n{}",
        json.dumps(verify_response, ensure_ascii=False, indent=2),
    )
    flag = extract_flag(verify_response)
    if flag:
        logger.success("Flag: {}", flag)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
