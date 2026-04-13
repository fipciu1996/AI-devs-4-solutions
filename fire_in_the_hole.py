"""Run the repository solvers in parallel, while keeping task pipelines ordered."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from devs_utilities.bootstrap import resolve_repo_python
from devs_utilities.logging import configure_logging, logger as shared_logger
from devs_utilities.repo_env import get_ngrok_auth_token, get_optional_env, load_repo_env


REPO_ROOT = Path(__file__).resolve().parent
PYTHON = str(resolve_repo_python(__file__))
DEFAULT_LOG_DIR = REPO_ROOT / "fire_in_the_hole_logs"
DEFAULT_COST_DIR = REPO_ROOT / "fire_in_the_hole_costs"
DEFAULT_RAILWAY_PROMPT = "Aktywuj trase X-01"
DEFAULT_NEGOTIATIONS_CHECK_DELAY_SECONDS = 5.0
DEFAULT_NEGOTIATIONS_POLL_ATTEMPTS = 12
logger = shared_logger.bind(component="fire_in_the_hole")

load_repo_env(__file__)
FLAG_PATTERN = re.compile(r"\{FLG:[^}]+\}")
SECRET_LABEL_PATTERN = re.compile(r"\b(?:sekret|secret)\b\s*:", re.IGNORECASE)
STEP_HEADER_PATTERN = re.compile(r"^=== Step \d+/\d+: (?P<label>.+?) ===$", re.MULTILINE)
SUMMARY_STEP_ALIASES = {
    "people": {
        "findhim": "findhim",
    },
}
TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cached_tokens",
    "reasoning_tokens",
    "cache_write_tokens",
    "total_tokens",
)


@dataclass(frozen=True, slots=True)
class StepSpec:
    label: str
    command: tuple[str, ...]
    cwd: Path = REPO_ROOT
    delay_after_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class TaskSpec:
    name: str
    description: str
    build_steps: Callable[[argparse.Namespace], list[StepSpec]]
    mode_label: str = "single-step"
    is_enabled: Callable[[argparse.Namespace], tuple[bool, str | None]] = lambda _args: (True, None)


@dataclass(frozen=True, slots=True)
class TaskResult:
    name: str
    status: str
    duration_seconds: float
    step_count: int
    cost_path: Path | None = None
    log_path: Path | None = None
    detail: str | None = None
    failed_step: str | None = None
    exit_code: int | None = None


@dataclass(frozen=True, slots=True)
class TaskFlags:
    task_name: str
    main_flag: str | None = None
    secret_flag: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Names of tasks to run. Accepts whitespace or comma-separated values. Default: all.",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=None,
        help="Names of tasks to skip. Accepts whitespace or comma-separated values.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of tasks to run in parallel. Default: 4.",
    )
    parser.add_argument(
        "--verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add --verify to tasks that support it. Enabled by default.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the registered solver tasks and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show commands that would run without starting subprocesses.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help=f"Directory for per-task logs. Default: {DEFAULT_LOG_DIR.name}",
    )
    parser.add_argument(
        "--cost-dir",
        type=Path,
        default=DEFAULT_COST_DIR,
        help=f"Directory for per-task token cost reports. Default: {DEFAULT_COST_DIR.name}",
    )
    parser.add_argument(
        "--railway-prompt",
        default=DEFAULT_RAILWAY_PROMPT,
        help=f"Prompt passed to railway\\route_agent.py. Default: {DEFAULT_RAILWAY_PROMPT}",
    )
    parser.add_argument(
        "--negotiations-public-base-url",
        default=None,
        help="Public base URL used by negotiations\\submit_tools.py.",
    )
    parser.add_argument(
        "--negotiations-use-ngrok",
        action="store_true",
        help="Expose the local negotiations tool with the ngrok Python SDK. Auto-enabled when NGROK_AUTH_TOKEN is set.",
    )
    parser.add_argument(
        "--negotiations-ngrok-domain",
        default=None,
        help="Optional reserved ngrok domain for the negotiations task.",
    )
    parser.add_argument(
        "--negotiations-check-delay-seconds",
        type=float,
        default=DEFAULT_NEGOTIATIONS_CHECK_DELAY_SECONDS,
        help=(
            "Delay between negotiations registration and status check. "
            f"Default: {DEFAULT_NEGOTIATIONS_CHECK_DELAY_SECONDS:.0f}s."
        ),
    )
    parser.add_argument(
        "--negotiations-check-attempts",
        type=int,
        default=DEFAULT_NEGOTIATIONS_POLL_ATTEMPTS,
        help=(
            "Maximum number of async status polls when negotiations uses ngrok. "
            f"Default: {DEFAULT_NEGOTIATIONS_POLL_ATTEMPTS}."
        ),
    )
    args = parser.parse_args()
    if args.max_workers < 1:
        parser.error("--max-workers must be >= 1.")
    if args.negotiations_check_delay_seconds < 0:
        parser.error("--negotiations-check-delay-seconds must be >= 0.")
    if args.negotiations_check_attempts < 1:
        parser.error("--negotiations-check-attempts must be >= 1.")
    args.log_dir = resolve_path(args.log_dir)
    args.cost_dir = resolve_path(args.cost_dir)
    return args


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def normalize_task_names(raw_values: list[str] | None) -> list[str]:
    if not raw_values:
        return []
    names: list[str] = []
    for raw_value in raw_values:
        for chunk in raw_value.split(","):
            name = chunk.strip().casefold()
            if name:
                names.append(name)
    return names


def python_command(script: str | Path, *args: str) -> tuple[str, ...]:
    script_path = Path(script)
    if not script_path.is_absolute():
        script_path = REPO_ROOT / script_path
    return (PYTHON, str(script_path.relative_to(REPO_ROOT)), *args)


def maybe_verify(args: argparse.Namespace, supports_verify: bool) -> tuple[str, ...]:
    if supports_verify and args.verify:
        return ("--verify",)
    return ()


def build_single_script_task(
    name: str,
    description: str,
    script: str | Path,
    *,
    supports_verify: bool = False,
    static_args: tuple[str, ...] = (),
) -> TaskSpec:
    def build_steps(args: argparse.Namespace) -> list[StepSpec]:
        command = python_command(
            script,
            *static_args,
            *maybe_verify(args, supports_verify),
        )
        return [StepSpec(label=name, command=command)]

    return TaskSpec(name=name, description=description, build_steps=build_steps)


def build_people_task() -> TaskSpec:
    def is_enabled(_args: argparse.Namespace) -> tuple[bool, str | None]:
        csv_path = REPO_ROOT / "people" / "people.csv"
        if csv_path.exists():
            return True, None
        return False, f"missing private dataset: {csv_path}"

    def build_steps(args: argparse.Namespace) -> list[StepSpec]:
        verify_args = maybe_verify(args, True)
        return [
            StepSpec(
                label="people-filter",
                command=python_command("people/filter_people.py", *verify_args),
            ),
            StepSpec(
                label="findhim",
                command=python_command("findhim/solve_findhim.py", *verify_args),
            ),
        ]

    return TaskSpec(
        name="people",
        description="Run the people candidate filter from people/ and then the findhim lookup from findhim/.",
        build_steps=build_steps,
        mode_label="pipeline",
        is_enabled=is_enabled,
    )


def build_sendit_task() -> TaskSpec:
    def build_steps(_args: argparse.Namespace) -> list[StepSpec]:
        return [
            StepSpec(
                label="sendit-download",
                command=python_command(
                    "sendit/download_attachments.py",
                    "--index",
                    "sendit/index.md",
                ),
            ),
            StepSpec(
                label="sendit-analyze",
                command=python_command(
                    "sendit/analyze_attachments_openrouter.py",
                    "--input-dir",
                    "sendit/attachments",
                    "--output-dir",
                    "sendit/analysis",
                ),
            ),
            StepSpec(
                label="sendit-draft",
                command=python_command(
                    "sendit/build_declaration_draft.py",
                    "--shipment-file",
                    "sendit/shipment.example.json",
                    "--analysis-dir",
                    "sendit/analysis",
                    "--attachments-dir",
                    "sendit/attachments",
                    "--output-dir",
                    "sendit/draft",
                ),
            ),
            StepSpec(
                label="sendit-legal",
                command=python_command(
                    "sendit/generate_legal_declaration.py",
                    "--shipment-file",
                    "sendit/shipment.legal.example.json",
                    "--output-dir",
                    "sendit/legal_output",
                ),
            ),
        ]

    return TaskSpec(
        name="sendit",
        description="Run the sendit download/analyze/draft/legal pipeline.",
        build_steps=build_steps,
        mode_label="pipeline",
    )


def build_railway_task() -> TaskSpec:
    def build_steps(args: argparse.Namespace) -> list[StepSpec]:
        return [
            StepSpec(
                label="railway",
                command=python_command("railway/route_agent.py", args.railway_prompt),
            )
        ]

    return TaskSpec(
        name="railway",
        description="Run the railway route agent with a configurable prompt.",
        build_steps=build_steps,
    )


def build_negotiations_task() -> TaskSpec:
    def is_enabled(args: argparse.Namespace) -> tuple[bool, str | None]:
        if args.negotiations_public_base_url:
            return True, None
        if args.negotiations_use_ngrok:
            return True, None
        return (
            False,
            "missing --negotiations-public-base-url or NGROK_AUTH_TOKEN / Ngrok config",
        )

    def build_steps(args: argparse.Namespace) -> list[StepSpec]:
        if args.negotiations_use_ngrok and not args.negotiations_public_base_url:
            command = [
                *python_command(
                    "negotiations/submit_tools.py",
                    "--use-ngrok",
                    "--launch-server",
                    "--wait-for-check",
                    "--check-interval-seconds",
                    str(args.negotiations_check_delay_seconds),
                    "--check-attempts",
                    str(args.negotiations_check_attempts),
                )
            ]
            if args.negotiations_ngrok_domain:
                command.extend(("--ngrok-domain", args.negotiations_ngrok_domain))
            return [
                StepSpec(
                    label="negotiations-ngrok",
                    command=tuple(command),
                )
            ]

        return [
            StepSpec(
                label="negotiations-register",
                command=python_command(
                    "negotiations/submit_tools.py",
                    "--public-base-url",
                    args.negotiations_public_base_url,
                ),
                delay_after_seconds=args.negotiations_check_delay_seconds,
            ),
            StepSpec(
                label="negotiations-check",
                command=python_command(
                    "negotiations/submit_tools.py",
                    "--check",
                ),
            ),
        ]

    return TaskSpec(
        name="negotiations",
        description="Register the negotiations tool via a manual public URL or Ngrok SDK and check the async status.",
        build_steps=build_steps,
        mode_label="pipeline",
        is_enabled=is_enabled,
    )


def build_task_registry() -> dict[str, TaskSpec]:
    tasks = [
        build_single_script_task(
            "categorize",
            "Run the categorize solver.",
            "categorize/solve_categorize.py",
        ),
        build_single_script_task(
            "domatowo",
            "Run the domatowo solver.",
            "domatowo/solve_domatowo.py",
            supports_verify=True,
        ),
        build_single_script_task(
            "drone",
            "Run the drone solver.",
            "drone/solve_drone.py",
        ),
        build_single_script_task(
            "electricity",
            "Run the electricity solver.",
            "electricity/solve_electricity.py",
        ),
        build_single_script_task(
            "failure",
            "Run the failure solver.",
            "failure/solve_failure.py",
            supports_verify=True,
        ),
        build_single_script_task(
            "filesystem",
            "Run the filesystem solver.",
            "filesystem/solve_filesystem.py",
        ),
        build_single_script_task(
            "firmware",
            "Run the firmware solver.",
            "firmware/solve_firmware.py",
            supports_verify=True,
        ),
        build_single_script_task(
            "foodwarehouse",
            "Run the foodwarehouse solver.",
            "foodwarehouse/solve_foodwarehouse.py",
            supports_verify=True,
        ),
        build_single_script_task(
            "goingthere",
            "Run the goingthere solver.",
            "goingthere/solve_goingthere.py",
        ),
        build_single_script_task(
            "mailbox",
            "Run the mailbox solver.",
            "mailbox/solve_mailbox.py",
        ),
        build_negotiations_task(),
        build_single_script_task(
            "okoeditor",
            "Run the okoeditor solver.",
            "okoeditor/solve_okoeditor.py",
        ),
        build_people_task(),
        build_single_script_task(
            "phonecall",
            "Run the phonecall solver.",
            "phonecall/solve_phonecall.py",
        ),
        build_single_script_task(
            "radiomonitoring",
            "Run the radiomonitoring solver.",
            "radiomonitoring/solve_radiomonitoring.py",
            supports_verify=True,
        ),
        build_railway_task(),
        build_single_script_task(
            "reactor",
            "Run the reactor solver.",
            "reactor/solve_reactor.py",
        ),
        build_single_script_task(
            "savethem",
            "Run the savethem solver.",
            "savethem/solve_savethem.py",
            supports_verify=True,
        ),
        build_single_script_task(
            "shellaccess",
            "Run the shellaccess solver.",
            "shellaccess/solve_shellaccess.py",
        ),
        build_sendit_task(),
        build_single_script_task(
            "sensors",
            "Run the sensors solver.",
            "sensors/solve_sensors.py",
            supports_verify=True,
        ),
        build_single_script_task(
            "timetravel",
            "Run the timetravel solver.",
            "timetravel/solve_timetravel.py",
        ),
        build_single_script_task(
            "windpower",
            "Run the windpower solver.",
            "windpower/solve_windpower.py",
        ),
    ]
    return {task.name: task for task in tasks}


def resolve_selected_tasks(
    registry: dict[str, TaskSpec],
    args: argparse.Namespace,
) -> list[TaskSpec]:
    include_names = normalize_task_names(args.tasks)
    exclude_names = set(normalize_task_names(args.exclude))
    unknown_names = sorted(
        {
            name
            for name in include_names + list(exclude_names)
            if name not in registry
        }
    )
    if unknown_names:
        raise ValueError(
            "Unknown task names: "
            + ", ".join(unknown_names)
            + ". Known tasks: "
            + ", ".join(registry)
        )

    selected_names = include_names or list(registry)
    selected_tasks = [registry[name] for name in selected_names if name not in exclude_names]
    if not selected_tasks:
        raise ValueError("No tasks selected.")
    return selected_tasks


def list_tasks(tasks: list[TaskSpec], args: argparse.Namespace) -> int:
    logger.info("Registered solver tasks:")
    for task in tasks:
        enabled, reason = task.is_enabled(args)
        status = "enabled" if enabled else f"skipped ({reason})"
        logger.info("- {} [{}] - {}", task.name, task.mode_label, status)
        logger.info("  {}", task.description)
    logger.info("`proxy` is intentionally not included because it is a service, not a solver.")
    return 0


def show_dry_run(tasks: list[TaskSpec], args: argparse.Namespace) -> int:
    logger.info("Dry run for {} selected tasks:", len(tasks))
    for task in tasks:
        enabled, reason = task.is_enabled(args)
        if not enabled:
            logger.warning("- {} skipped: {}", task.name, reason)
            continue
        logger.info("- {}", task.name)
        for step in task.build_steps(args):
            logger.info("  [{}] {}", step.label, subprocess.list2cmdline(list(step.command)))
    return 0


def empty_token_totals() -> dict[str, int]:
    return {field: 0 for field in TOKEN_FIELDS}


def normalize_token_totals(payload: object) -> dict[str, int]:
    normalized = empty_token_totals()
    if not isinstance(payload, dict):
        return normalized
    for field in TOKEN_FIELDS:
        value = payload.get(field, 0)
        if isinstance(value, bool):
            normalized[field] = 0
        elif isinstance(value, int):
            normalized[field] = value
        elif isinstance(value, float):
            normalized[field] = int(value)
        elif isinstance(value, str) and value.strip().isdigit():
            normalized[field] = int(value.strip())
    return normalized


def add_token_totals(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    return {
        field: left[field] + right[field]
        for field in TOKEN_FIELDS
    }


def task_cost_path(cost_dir: Path, task_name: str) -> Path:
    return cost_dir / f"{task_name}.json"


def prepare_cost_dir(cost_dir: Path) -> None:
    cost_dir.mkdir(parents=True, exist_ok=True)
    for json_path in cost_dir.glob("*.json"):
        json_path.unlink()


def load_cost_report(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def write_task_cost_report(result: TaskResult) -> dict[str, object] | None:
    if result.cost_path is None:
        return None

    payload = load_cost_report(result.cost_path)
    payload["task"] = result.name
    payload["status"] = result.status
    payload["duration_seconds"] = round(result.duration_seconds, 3)
    payload["step_count"] = result.step_count
    payload["request_count"] = int(payload.get("request_count") or 0)
    raw_models = payload.get("models")
    if not isinstance(raw_models, list):
        raw_models = []
    payload["models"] = sorted(
        model
        for model in raw_models
        if isinstance(model, str) and model.strip()
    )
    payload["totals"] = normalize_token_totals(payload.get("totals"))
    payload["usage_by_model"] = (
        payload.get("usage_by_model")
        if isinstance(payload.get("usage_by_model"), dict)
        else {}
    )
    payload["calls"] = (
        payload.get("calls")
        if isinstance(payload.get("calls"), list)
        else []
    )
    payload["source"] = "fire_in_the_hole"
    if result.log_path is not None:
        payload["log_path"] = str(result.log_path)
    if result.detail:
        payload["detail"] = result.detail
    if result.failed_step:
        payload["failed_step"] = result.failed_step
    if result.exit_code is not None:
        payload["exit_code"] = result.exit_code

    result.cost_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def write_run_cost_summary(
    tasks: list[TaskSpec],
    results_by_name: dict[str, TaskResult],
    *,
    cost_dir: Path,
) -> Path:
    totals = empty_token_totals()
    success_count = 0
    skipped_count = 0
    failed_count = 0
    task_reports: dict[str, dict[str, object]] = {}

    for task in tasks:
        result = results_by_name[task.name]
        report = write_task_cost_report(result) or {}
        report_totals = normalize_token_totals(report.get("totals"))
        totals = add_token_totals(totals, report_totals)
        task_reports[task.name] = {
            "status": result.status,
            "duration_seconds": round(result.duration_seconds, 3),
            "request_count": int(report.get("request_count") or 0),
            "models": report.get("models", []),
            "totals": report_totals,
            "log_path": str(result.log_path) if result.log_path is not None else None,
            "cost_path": str(result.cost_path) if result.cost_path is not None else None,
        }
        if result.status == "success":
            success_count += 1
        elif result.status == "skipped":
            skipped_count += 1
        else:
            failed_count += 1

    summary_path = cost_dir / "fire_in_the_hole_total.json"
    summary_payload = {
        "run": "fire_in_the_hole",
        "task_count": len(tasks),
        "success_count": success_count,
        "skipped_count": skipped_count,
        "failed_count": failed_count,
        "totals": totals,
        "tasks": task_reports,
    }
    summary_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary_path


def run_task(task: TaskSpec, args: argparse.Namespace) -> TaskResult:
    cost_path = task_cost_path(args.cost_dir, task.name)
    enabled, reason = task.is_enabled(args)
    if not enabled:
        result = TaskResult(
            name=task.name,
            status="skipped",
            duration_seconds=0.0,
            step_count=0,
            cost_path=cost_path,
            detail=reason,
        )
        write_task_cost_report(result)
        return result

    steps = task.build_steps(args)
    log_path = args.log_dir / f"{task.name}.log"
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.cost_dir.mkdir(parents=True, exist_ok=True)
    if cost_path.exists():
        cost_path.unlink()
    start_time = time.perf_counter()

    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"Task: {task.name}\n")
        handle.write(f"Description: {task.description}\n")
        handle.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        handle.write(f"Step count: {len(steps)}\n\n")

    logger.info("Starting {} ({} step(s)).", task.name, len(steps))

    for index, step in enumerate(steps, start=1):
        command_string = subprocess.list2cmdline(list(step.command))
        logger.info("[{} {}/{}] {}", task.name, index, len(steps), command_string)
        step_env = os.environ.copy()
        step_env["OPENROUTER_USAGE_OUTPUT_PATH"] = str(cost_path)
        step_env["OPENROUTER_USAGE_TASK_NAME"] = task.name
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"=== Step {index}/{len(steps)}: {step.label} ===\n")
            handle.write(f"CWD: {step.cwd}\n")
            handle.write(f"CMD: {command_string}\n\n")
            handle.flush()
            completed = subprocess.run(
                step.command,
                cwd=step.cwd,
                check=False,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=step_env,
            )
            handle.write("\n")

        if completed.returncode != 0:
            result = TaskResult(
                name=task.name,
                status="failed",
                duration_seconds=time.perf_counter() - start_time,
                step_count=len(steps),
                cost_path=cost_path,
                log_path=log_path,
                detail=f"step `{step.label}` exited with {completed.returncode}",
                failed_step=step.label,
                exit_code=completed.returncode,
            )
            write_task_cost_report(result)
            return result

        if step.delay_after_seconds > 0 and index < len(steps):
            logger.info(
                "[{}] Waiting {:.1f}s before the next step.",
                task.name,
                step.delay_after_seconds,
            )
            time.sleep(step.delay_after_seconds)

    result = TaskResult(
        name=task.name,
        status="success",
        duration_seconds=time.perf_counter() - start_time,
        step_count=len(steps),
        cost_path=cost_path,
        log_path=log_path,
    )
    write_task_cost_report(result)
    return result


def extract_flags_from_log_text(task_name: str, log_text: str) -> TaskFlags | None:
    main_flag: str | None = None
    secret_flag: str | None = None
    ordered_flags: list[str] = []

    for line in log_text.splitlines():
        flags_in_line = FLAG_PATTERN.findall(line)
        if not flags_in_line:
            continue

        secret_match = SECRET_LABEL_PATTERN.search(line)
        if secret_match:
            before_secret = line[: secret_match.start()]
            after_secret = line[secret_match.end() :]
            before_flags = FLAG_PATTERN.findall(before_secret)
            after_flags = FLAG_PATTERN.findall(after_secret)
            ordered_flags.extend(before_flags)
            if before_flags and main_flag is None:
                main_flag = before_flags[-1]
            if after_flags and secret_flag is None:
                secret_flag = after_flags[0]
            ordered_flags.extend(after_flags)
            continue

        ordered_flags.extend(flags_in_line)
        if main_flag is None:
            main_flag = flags_in_line[0]

    unique_flags: list[str] = []
    for flag in ordered_flags:
        if flag not in unique_flags:
            unique_flags.append(flag)

    if main_flag is None and unique_flags:
        main_flag = unique_flags[0]
    if secret_flag is None and len(unique_flags) > 1:
        secret_flag = unique_flags[1]

    if main_flag is None and secret_flag is None:
        return None
    return TaskFlags(task_name=task_name, main_flag=main_flag, secret_flag=secret_flag)


def read_task_log_text(result: TaskResult) -> str | None:
    if result.log_path is None or not result.log_path.exists():
        return None
    return result.log_path.read_text(encoding="utf-8", errors="replace")


def extract_step_log_sections(log_text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    matches = list(STEP_HEADER_PATTERN.finditer(log_text))
    for index, match in enumerate(matches):
        step_label = match.group("label").strip()
        step_start = match.end()
        step_end = matches[index + 1].start() if index + 1 < len(matches) else len(log_text)
        sections[step_label] = log_text[step_start:step_end]
    return sections


def extract_task_flags(result: TaskResult) -> TaskFlags | None:
    log_text = read_task_log_text(result)
    if log_text is None:
        return None
    return extract_flags_from_log_text(result.name, log_text)


def extract_additional_task_flags(result: TaskResult) -> list[TaskFlags]:
    step_aliases = SUMMARY_STEP_ALIASES.get(result.name)
    if not step_aliases:
        return []

    log_text = read_task_log_text(result)
    if log_text is None:
        return []

    step_logs = extract_step_log_sections(log_text)
    additional_flags: list[TaskFlags] = []
    for step_label, summary_name in step_aliases.items():
        step_log_text = step_logs.get(step_label)
        if step_log_text is None:
            continue
        task_flags = extract_flags_from_log_text(summary_name, step_log_text)
        if task_flags is not None:
            additional_flags.append(task_flags)
    return additional_flags


def collect_success_flags(
    tasks: list[TaskSpec],
    results_by_name: dict[str, TaskResult],
) -> list[TaskFlags]:
    collected_flags: list[TaskFlags] = []
    for task in tasks:
        result = results_by_name[task.name]
        task_flags = extract_task_flags(result)
        if result.status == "success":
            collected_flags.append(task_flags or TaskFlags(task_name=task.name))
        elif task_flags is not None:
            collected_flags.append(task_flags)
        collected_flags.extend(extract_additional_task_flags(result))
    return collected_flags


def render_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def format_row(columns: tuple[str, ...]) -> str:
        return " | ".join(
            value.ljust(widths[index])
            for index, value in enumerate(columns)
        )

    separator = "-+-".join("-" * width for width in widths)
    lines = [
        format_row(headers),
        separator,
        *(format_row(row) for row in rows),
    ]
    return "\n".join(lines)


def render_flags_table(flags: list[TaskFlags]) -> str:
    headers = ("Task name", "Main flag", "Secret flag")
    rows = [
        (
            item.task_name,
            item.main_flag or "-",
            item.secret_flag or "-",
        )
        for item in flags
    ]
    return render_table(headers, rows)


def format_duration_seconds(seconds: float) -> str:
    return f"{seconds:.1f}s"


def extract_models_used(result: TaskResult) -> list[str]:
    if result.cost_path is None:
        return []
    report = load_cost_report(result.cost_path)
    raw_models = report.get("models")
    if not isinstance(raw_models, list):
        return []
    return sorted(
        model
        for model in raw_models
        if isinstance(model, str) and model.strip()
    )


def did_task_fetch_flag(result: TaskResult) -> bool:
    return extract_task_flags(result) is not None or bool(extract_additional_task_flags(result))


def render_task_run_table(
    tasks: list[TaskSpec],
    results_by_name: dict[str, TaskResult],
) -> str:
    headers = ("Task name", "Time from starting task", "Models used", "If flag fetched")
    rows = [
        (
            task.name,
            format_duration_seconds(results_by_name[task.name].duration_seconds),
            ", ".join(extract_models_used(results_by_name[task.name])) or "-",
            "yes" if did_task_fetch_flag(results_by_name[task.name]) else "no",
        )
        for task in tasks
    ]
    return render_table(headers, rows)


def summarize_results(tasks: list[TaskSpec], results_by_name: dict[str, TaskResult]) -> int:
    failure_count = 0
    skipped_count = 0
    success_count = 0
    collected_flags = collect_success_flags(tasks, results_by_name)

    logger.info("")
    logger.info("Summary:")
    for task in tasks:
        result = results_by_name[task.name]
        if result.status == "success":
            success_count += 1
            logger.success(
                "- {} OK in {:.1f}s (log: {})",
                task.name,
                result.duration_seconds,
                result.log_path,
            )
            continue
        if result.status == "skipped":
            skipped_count += 1
            logger.warning("- {} skipped: {}", task.name, result.detail)
            continue

        failure_count += 1
        logger.error(
            "- {} failed: {} (log: {})",
            task.name,
            result.detail,
            result.log_path,
        )

    logger.info(
        "Totals: {} success, {} skipped, {} failed.",
        success_count,
        skipped_count,
        failure_count,
    )
    logger.info("")
    logger.info("Task runs:")
    logger.info("\n{}", render_task_run_table(tasks, results_by_name))
    if collected_flags:
        logger.info("")
        logger.info("Collected flags:")
        logger.info("\n{}", render_flags_table(collected_flags))
    return 1 if failure_count else 0


def apply_negotiations_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Fill negotiations launcher defaults from the repository environment."""

    if not args.negotiations_public_base_url:
        args.negotiations_public_base_url = get_optional_env("NEGOTIATIONS_PUBLIC_BASE_URL")
    if not args.negotiations_ngrok_domain:
        args.negotiations_ngrok_domain = get_optional_env("NGROK_DOMAIN")
    if not args.negotiations_use_ngrok:
        args.negotiations_use_ngrok = bool(get_ngrok_auth_token())
    return args


def main() -> int:
    configure_logging(name="fire_in_the_hole")
    args = parse_args()
    apply_negotiations_defaults(args)
    current_python = str(Path(sys.executable).resolve())
    if current_python != PYTHON:
        logger.info("Using repository interpreter {} instead of current {}.", PYTHON, current_python)

    registry = build_task_registry()
    try:
        selected_tasks = resolve_selected_tasks(registry, args)
    except ValueError as exc:
        logger.error("{}", exc)
        return 1

    if args.list:
        return list_tasks(selected_tasks, args)
    if args.dry_run:
        return show_dry_run(selected_tasks, args)

    max_workers = min(args.max_workers, len(selected_tasks))
    logger.info(
        "Running {} task(s) with up to {} worker(s). Logs: {} Costs: {}",
        len(selected_tasks),
        max_workers,
        args.log_dir,
        args.cost_dir,
    )
    prepare_cost_dir(args.cost_dir)

    results_by_name: dict[str, TaskResult] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task: dict[Future[TaskResult], TaskSpec] = {
            executor.submit(run_task, task, args): task
            for task in selected_tasks
        }
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover - defensive launcher guard
                result = TaskResult(
                    name=task.name,
                    status="failed",
                    duration_seconds=0.0,
                    step_count=0,
                    detail=f"launcher exception: {exc}",
                )

            results_by_name[result.name] = result
            if result.status == "success":
                logger.success("{} finished in {:.1f}s.", result.name, result.duration_seconds)
            elif result.status == "skipped":
                logger.warning("{} skipped: {}", result.name, result.detail)
            else:
                logger.error("{} failed: {}", result.name, result.detail)

    summary_cost_path = write_run_cost_summary(
        selected_tasks,
        results_by_name,
        cost_dir=args.cost_dir,
    )
    logger.info("Token cost reports saved to {}", args.cost_dir)
    logger.info("Total token usage saved to {}", summary_cost_path)
    return summarize_results(selected_tasks, results_by_name)


if __name__ == "__main__":
    raise SystemExit(main())
