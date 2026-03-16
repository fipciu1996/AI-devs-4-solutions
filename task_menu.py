"""Interactive launcher for the repository tasks."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from repo_env import load_repo_env


REPO_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
DEFAULT_CATEGORIZE_ORDER = "J-D-I-B-A-C-G-E-H-F"

load_repo_env(__file__)


@dataclass(frozen=True, slots=True)
class CommandSpec:
    command: list[str]
    cwd: Path


@dataclass(frozen=True, slots=True)
class MenuAction:
    label: str
    description: str
    builder: Callable[[], list[CommandSpec] | None]


def prompt_text(label: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{label}{suffix}: ").strip()
    if value:
        return value
    return default or ""


def prompt_yes_no(label: str, default: bool = False) -> bool:
    default_hint = "Y/n" if default else "y/N"
    value = input(f"{label} [{default_hint}]: ").strip().lower()
    if not value:
        return default
    return value in {"y", "yes", "t", "true", "1"}


def pause() -> None:
    input("\nPress Enter to continue...")


def run_commands(commands: list[CommandSpec]) -> int:
    last_return_code = 0
    for command in commands:
        print(f"\nRunning in {command.cwd}: {' '.join(command.command)}\n")
        try:
            result = subprocess.run(command.command, cwd=command.cwd, check=False)
        except FileNotFoundError as exc:
            print(f"Could not start command: {exc}")
            return 1
        last_return_code = result.returncode
        if last_return_code != 0:
            print(f"\nCommand finished with exit code {last_return_code}.")
            return last_return_code
    return last_return_code


def build_python_command(script: str, *args: str, cwd: Path | None = None) -> CommandSpec:
    return CommandSpec(
        command=[PYTHON, script, *args],
        cwd=cwd or REPO_ROOT,
    )


def categorize_default() -> list[CommandSpec]:
    return [build_python_command("categorize\\solve_categorize.py")]


def categorize_reset() -> list[CommandSpec]:
    return [build_python_command("categorize\\solve_categorize.py", "--reset")]


def categorize_custom() -> list[CommandSpec]:
    order_mode = prompt_text("Order mode", "csv_position")
    order = prompt_text("Order", DEFAULT_CATEGORIZE_ORDER)
    args = [f"--order-mode={order_mode}", f"--order={order}"]
    if prompt_yes_no("Reset before sending?", False):
        args.append("--reset")
    return [build_python_command("categorize\\solve_categorize.py", *args)]


def people_filter_dry_run() -> list[CommandSpec]:
    return [build_python_command("people\\filter_people.py", "--dry-run")]


def people_filter_save() -> list[CommandSpec]:
    output_path = prompt_text("Output file", "people\\people_result.json")
    return [build_python_command("people\\filter_people.py", "--output", output_path)]


def people_filter_verify() -> list[CommandSpec]:
    output_path = prompt_text("Output file", "people\\people_result.json")
    return [
        build_python_command(
            "people\\filter_people.py",
            "--output",
            output_path,
            "--verify",
        )
    ]


def people_find_agent_dry_run() -> list[CommandSpec]:
    suspects = prompt_text("Suspects file", "people\\people_result.json")
    return [build_python_command("people\\find-agent.py", "--suspects", suspects, "--dry-run")]


def people_find_agent_save() -> list[CommandSpec]:
    suspects = prompt_text("Suspects file", "people\\people_result.json")
    output_path = prompt_text("Output file", "people\\findhim_result.json")
    args = ["--suspects", suspects, "--output", output_path]
    if prompt_yes_no("Refresh plants cache?", False):
        args.append("--refresh-plants")
    return [build_python_command("people\\find-agent.py", *args)]


def people_find_agent_verify() -> list[CommandSpec]:
    suspects = prompt_text("Suspects file", "people\\people_result.json")
    output_path = prompt_text("Output file", "people\\findhim_result.json")
    args = ["--suspects", suspects, "--output", output_path, "--verify"]
    if prompt_yes_no("Refresh plants cache?", False):
        args.append("--refresh-plants")
    return [build_python_command("people\\find-agent.py", *args)]


def people_full_pipeline() -> list[CommandSpec]:
    people_output = prompt_text("People output file", "people\\people_result.json")
    findhim_output = prompt_text("Findhim output file", "people\\findhim_result.json")
    commands = [
        build_python_command("people\\filter_people.py", "--output", people_output),
        build_python_command(
            "people\\find-agent.py",
            "--suspects",
            people_output,
            "--output",
            findhim_output,
        ),
    ]
    return commands


def railway_run_prompt() -> list[CommandSpec] | None:
    prompt = prompt_text("Railway prompt")
    if not prompt:
        print("Prompt is required.")
        return None
    args = ["railway\\route_agent.py"]
    if prompt_yes_no("Show tool results?", False):
        args.append("--show-tool-results")
    args.append(prompt)
    return [build_python_command(*args)]


def sendit_download() -> list[CommandSpec]:
    index_path = prompt_text("Index file", "sendit\\index.md")
    return [build_python_command("sendit\\download_attachments.py", "--index", index_path)]


def sendit_analyze() -> list[CommandSpec]:
    input_dir = prompt_text("Input dir", "sendit\\attachments")
    output_dir = prompt_text("Output dir", "sendit\\analysis")
    return [
        build_python_command(
            "sendit\\analyze_attachments_openrouter.py",
            "--input-dir",
            input_dir,
            "--output-dir",
            output_dir,
        )
    ]


def sendit_build_draft() -> list[CommandSpec]:
    shipment_file = prompt_text("Shipment file", "sendit\\shipment.example.json")
    return [
        build_python_command(
            "sendit\\build_declaration_draft.py",
            "--shipment-file",
            shipment_file,
            "--analysis-dir",
            "sendit\\analysis",
            "--attachments-dir",
            "sendit\\attachments",
            "--output-dir",
            "sendit\\draft",
        )
    ]


def sendit_generate_legal() -> list[CommandSpec]:
    shipment_file = prompt_text("Shipment file", "sendit\\shipment.legal.example.json")
    output_dir = prompt_text("Output dir", "sendit\\legal_output")
    return [
        build_python_command(
            "sendit\\generate_legal_declaration.py",
            "--shipment-file",
            shipment_file,
            "--output-dir",
            output_dir,
        )
    ]


def sendit_full_pipeline() -> list[CommandSpec]:
    shipment_file = prompt_text("Draft shipment file", "sendit\\shipment.example.json")
    legal_shipment_file = prompt_text(
        "Legal shipment file",
        "sendit\\shipment.legal.example.json",
    )
    return [
        build_python_command("sendit\\download_attachments.py", "--index", "sendit\\index.md"),
        build_python_command(
            "sendit\\analyze_attachments_openrouter.py",
            "--input-dir",
            "sendit\\attachments",
            "--output-dir",
            "sendit\\analysis",
        ),
        build_python_command(
            "sendit\\build_declaration_draft.py",
            "--shipment-file",
            shipment_file,
            "--analysis-dir",
            "sendit\\analysis",
            "--attachments-dir",
            "sendit\\attachments",
            "--output-dir",
            "sendit\\draft",
        ),
        build_python_command(
            "sendit\\generate_legal_declaration.py",
            "--shipment-file",
            legal_shipment_file,
            "--output-dir",
            "sendit\\legal_output",
        ),
    ]


def proxy_local_server() -> list[CommandSpec]:
    return [
        build_python_command(
            "-m",
            "proxy_api_server.server",
            cwd=REPO_ROOT / "proxy",
        )
    ]


def proxy_docker_compose() -> list[CommandSpec]:
    return [
        CommandSpec(
            command=["docker", "compose", "--env-file", "..\\.env", "up", "--build"],
            cwd=REPO_ROOT / "proxy",
        )
    ]


TASK_MENUS: dict[str, list[MenuAction]] = {
    "Categorize": [
        MenuAction("Run default", "Send prompts using current defaults.", categorize_default),
        MenuAction("Run with reset", "Reset the budget and send prompts again.", categorize_reset),
        MenuAction("Custom run", "Pick order mode and order interactively.", categorize_custom),
    ],
    "People": [
        MenuAction("Filter dry-run", "Preview filtered candidates only.", people_filter_dry_run),
        MenuAction("Filter and save", "Build people_result.json.", people_filter_save),
        MenuAction("Filter and verify", "Build and verify people_result.json.", people_filter_verify),
        MenuAction("Find agent dry-run", "Preview findhim step without verify.", people_find_agent_dry_run),
        MenuAction("Find agent and save", "Build findhim_result.json.", people_find_agent_save),
        MenuAction("Find agent and verify", "Build and verify findhim_result.json.", people_find_agent_verify),
        MenuAction("Full pipeline", "Run filter_people and find-agent sequentially.", people_full_pipeline),
    ],
    "Railway": [
        MenuAction("Run prompt", "Send a natural-language railway instruction.", railway_run_prompt),
    ],
    "Sendit": [
        MenuAction("Download attachments", "Resolve and copy/download files from index.md.", sendit_download),
        MenuAction("Analyze attachments", "Create OpenRouter analysis reports.", sendit_analyze),
        MenuAction("Build draft", "Create the working declaration draft.", sendit_build_draft),
        MenuAction("Generate legal declaration", "Run local legal validation/generation.", sendit_generate_legal),
        MenuAction("Full pipeline", "Run the common sendit flow end-to-end.", sendit_full_pipeline),
    ],
    "Proxy": [
        MenuAction("Run local server", "Start proxy_api_server with Python.", proxy_local_server),
        MenuAction("Run docker compose", "Start the proxy stack with Docker Compose.", proxy_docker_compose),
    ],
}


def show_main_menu() -> str:
    print("\nInteractive Task Menu\n")
    task_names = list(TASK_MENUS)
    for index, task_name in enumerate(task_names, start=1):
        print(f"{index}. {task_name}")
    print("q. Quit")
    choice = input("\nSelect task: ").strip().lower()
    if choice == "q":
        return choice
    if choice.isdigit():
        selected_index = int(choice) - 1
        if 0 <= selected_index < len(task_names):
            return task_names[selected_index]
    print("Invalid choice.")
    return ""


def show_task_menu(task_name: str, actions: list[MenuAction]) -> bool:
    while True:
        print(f"\n{task_name}\n")
        for index, action in enumerate(actions, start=1):
            print(f"{index}. {action.label} - {action.description}")
        print("b. Back")
        print("q. Quit")
        choice = input("\nSelect action: ").strip().lower()
        if choice == "b":
            return True
        if choice == "q":
            return False
        if not choice.isdigit():
            print("Invalid choice.")
            continue

        selected_index = int(choice) - 1
        if not 0 <= selected_index < len(actions):
            print("Invalid choice.")
            continue

        commands = actions[selected_index].builder()
        if commands is None:
            pause()
            continue
        exit_code = run_commands(commands)
        print(f"\nFinished with exit code {exit_code}.")
        pause()


def main() -> int:
    while True:
        selected_task = show_main_menu()
        if not selected_task:
            continue
        if selected_task == "q":
            print("Goodbye.")
            return 0
        should_continue = show_task_menu(selected_task, TASK_MENUS[selected_task])
        if not should_continue:
            print("Goodbye.")
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
