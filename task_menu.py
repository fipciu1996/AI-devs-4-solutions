"""Interactive launcher built dynamically from the current repository state."""

from __future__ import annotations

import shlex
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from devs_utilities.logging import configure_logging, logger as shared_logger
from repo_env import load_repo_env


REPO_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
IGNORED_DIRS = {
    ".git",
    ".idea",
    ".venv",
    "__pycache__",
    "devs_utilities",
}
logger = shared_logger.bind(component="task_menu")

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


@dataclass(frozen=True, slots=True)
class TaskMenu:
    name: str
    path: Path
    actions: tuple[MenuAction, ...]


def prompt_text(label: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{label}{suffix}: ").strip()
    if value:
        return value
    return default or ""


def pause() -> None:
    input("\nPress Enter to continue...")


def run_commands(commands: list[CommandSpec]) -> int:
    last_return_code = 0
    for command in commands:
        logger.info("Running in {}: {}", command.cwd, " ".join(command.command))
        try:
            result = subprocess.run(command.command, cwd=command.cwd, check=False)
        except FileNotFoundError as exc:
            logger.error("Could not start command: {}", exc)
            return 1
        last_return_code = result.returncode
        if last_return_code != 0:
            logger.warning("Command finished with exit code {}.", last_return_code)
            return last_return_code
    return last_return_code


def build_python_command(script_path: Path, *args: str, cwd: Path | None = None) -> CommandSpec:
    relative_script = script_path.relative_to(REPO_ROOT)
    return CommandSpec(
        command=[PYTHON, str(relative_script), *args],
        cwd=cwd or REPO_ROOT,
    )


def split_args(raw_args: str) -> list[str]:
    """Split a free-form argument string into CLI tokens."""

    if not raw_args.strip():
        return []
    return shlex.split(raw_args, posix=False)


def titleize_name(raw_name: str) -> str:
    """Convert a directory or file stem into a readable label."""

    return raw_name.replace("-", " ").replace("_", " ").title()


def script_sort_key(script_path: Path) -> tuple[int, str]:
    """Prefer obvious entrypoints before generic helpers."""

    name = script_path.name.casefold()
    if name.startswith("solve_"):
        return (0, name)
    if name == "route_agent.py":
        return (1, name)
    if name == "submit_tools.py":
        return (2, name)
    if name.startswith("test_"):
        return (9, name)
    return (3, name)


def module_sort_key(module_path: Path) -> tuple[int, str]:
    return (0, module_path.parent.name.casefold(), module_path.name.casefold())


def discover_task_dirs() -> list[Path]:
    """Return top-level task directories worth showing in the menu."""

    task_dirs: list[Path] = []
    for child in sorted(REPO_ROOT.iterdir(), key=lambda path: path.name.casefold()):
        if not child.is_dir() or child.name in IGNORED_DIRS or child.name.startswith("."):
            continue
        has_scripts = any(
            script.name != "__init__.py"
            for script in child.glob("*.py")
        )
        has_compose = (child / "docker-compose.yml").exists()
        has_server_module = any(child.glob("src/*/server.py"))
        if has_scripts or has_compose or has_server_module:
            task_dirs.append(child)
    return task_dirs


def build_script_action(script_path: Path) -> MenuAction:
    """Create a generic runner for a discovered Python script."""

    relative_script = script_path.relative_to(REPO_ROOT)
    label = f"Run {script_path.name}"
    description = f"Run {relative_script} with optional extra arguments."

    def builder() -> list[CommandSpec]:
        extra_args = split_args(prompt_text("Extra args", ""))
        return [build_python_command(script_path, *extra_args)]

    return MenuAction(label, description, builder)


def build_module_server_action(task_dir: Path, server_path: Path) -> MenuAction:
    """Create an action for `python -m package.server` entrypoints."""

    package_dir = server_path.parent
    module_name = f"{package_dir.name}.server"
    label = f"Run {module_name}"
    description = f"Start the local server module in {task_dir.name}."

    def builder() -> list[CommandSpec]:
        extra_args = split_args(prompt_text("Extra args", ""))
        return [
            CommandSpec(
                command=[PYTHON, "-m", module_name, *extra_args],
                cwd=task_dir,
            )
        ]

    return MenuAction(label, description, builder)


def build_docker_compose_action(task_dir: Path) -> MenuAction:
    """Create an action for a discovered docker-compose file."""

    def builder() -> list[CommandSpec]:
        extra_args = split_args(prompt_text("Compose args", "up --build"))
        return [
            CommandSpec(
                command=["docker", "compose", "--env-file", "..\\.env", *extra_args],
                cwd=task_dir,
            )
        ]

    return MenuAction(
        "Run docker compose",
        f"Run docker compose in {task_dir.name}.",
        builder,
    )


def discover_actions(task_dir: Path) -> tuple[MenuAction, ...]:
    """Build all actions available for a task directory."""

    actions: list[MenuAction] = []

    scripts = sorted(
        (
            path
            for path in task_dir.glob("*.py")
            if path.name != "__init__.py" and not path.name.startswith("test_")
        ),
        key=script_sort_key,
    )
    actions.extend(build_script_action(script_path) for script_path in scripts)

    server_modules = sorted(task_dir.glob("src/*/server.py"), key=module_sort_key)
    actions.extend(
        build_module_server_action(task_dir, server_path)
        for server_path in server_modules
    )

    if (task_dir / "docker-compose.yml").exists():
        actions.append(build_docker_compose_action(task_dir))

    return tuple(actions)


def discover_task_menus() -> list[TaskMenu]:
    """Discover current tasks and actions from the filesystem."""

    task_menus: list[TaskMenu] = []
    for task_dir in discover_task_dirs():
        actions = discover_actions(task_dir)
        if actions:
            task_menus.append(
                TaskMenu(
                    name=titleize_name(task_dir.name),
                    path=task_dir,
                    actions=actions,
                )
            )
    return task_menus


def show_main_menu(task_menus: list[TaskMenu]) -> str:
    logger.info("Interactive Task Menu")
    for index, task_menu in enumerate(task_menus, start=1):
        logger.info("{}. {}", index, task_menu.name)
    logger.info("q. Quit")
    choice = input("\nSelect task: ").strip().lower()
    if choice == "q":
        return choice
    if choice.isdigit():
        selected_index = int(choice) - 1
        if 0 <= selected_index < len(task_menus):
            return str(selected_index)
    logger.warning("Invalid choice.")
    return ""


def show_task_menu(task_menu: TaskMenu) -> bool:
    while True:
        logger.info("{}", task_menu.name)
        logger.info("Path: {}", task_menu.path)
        for index, action in enumerate(task_menu.actions, start=1):
            logger.info("{}. {} - {}", index, action.label, action.description)
        logger.info("b. Back")
        logger.info("q. Quit")
        choice = input("\nSelect action: ").strip().lower()
        if choice == "b":
            return True
        if choice == "q":
            return False
        if not choice.isdigit():
            logger.warning("Invalid choice.")
            continue

        selected_index = int(choice) - 1
        if not 0 <= selected_index < len(task_menu.actions):
            logger.warning("Invalid choice.")
            continue

        commands = task_menu.actions[selected_index].builder()
        if commands is None:
            pause()
            continue
        exit_code = run_commands(commands)
        logger.info("Finished with exit code {}.", exit_code)
        pause()


def main() -> int:
    configure_logging(name="task_menu")
    while True:
        task_menus = discover_task_menus()
        if not task_menus:
            logger.warning("No runnable tasks were discovered.")
            return 1

        selected_task = show_main_menu(task_menus)
        if not selected_task:
            continue
        if selected_task == "q":
            logger.info("Goodbye.")
            return 0

        task_menu = task_menus[int(selected_task)]
        should_continue = show_task_menu(task_menu)
        if not should_continue:
            logger.info("Goodbye.")
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
