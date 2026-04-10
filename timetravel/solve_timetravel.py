"""Run the timetravel backend and frontend agents together."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT_HINT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_HINT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_HINT))

from devs_utilities.bootstrap import bootstrap_repo
from devs_utilities.logging import configure_logging, logger as shared_logger

from timetravel.models import MissionStatus
from timetravel.store import SharedStateStore


REPO_ROOT = bootstrap_repo(__file__)
logger = shared_logger.bind(component="timetravel")
PYTHON = sys.executable
DEFAULT_RUNTIME_DIR = Path(__file__).resolve().parent / "runtime"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent-runtime",
        choices=("auto", "deterministic", "openrouter"),
        default="auto",
        help="Execution runtime for backend and frontend agents. Default: auto",
    )
    parser.add_argument(
        "--show-tool-results",
        action="store_true",
        help="Print OpenRouter tool results for both agents.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_RUNTIME_DIR / "timetravel_state.sqlite3",
        help="Path to the shared SQLite state database.",
    )
    parser.add_argument(
        "--storage-state-path",
        type=Path,
        default=DEFAULT_RUNTIME_DIR / "frontend_storage_state.json",
        help="Path to the persisted Playwright storage_state JSON file.",
    )
    parser.add_argument(
        "--manual-login",
        action="store_true",
        help="Wait for an interactive preview login instead of requiring an inherited session.",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=1.0,
        help="Polling interval used by both agents. Default: 1.0",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset the timetravel device before starting the mission.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for all processes.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(name="timetravel", verbose=args.verbose)
    try:
        import playwright  # noqa: F401
    except ImportError:
        logger.error(
            "Missing optional dependency `playwright`. Run `.venv\\Scripts\\python.exe -m pip install -e .` first."
        )
        return 1
    args.db_path.parent.mkdir(parents=True, exist_ok=True)
    if args.db_path.exists():
        args.db_path.unlink()

    store = SharedStateStore(args.db_path)

    backend_command = [
        PYTHON,
        str(Path(__file__).with_name("backend_agent.py")),
        "--db-path",
        str(args.db_path),
        "--agent-runtime",
        args.agent_runtime,
        "--poll-interval-seconds",
        str(args.poll_interval_seconds),
    ]
    frontend_command = [
        PYTHON,
        str(Path(__file__).with_name("frontend_agent.py")),
        "--db-path",
        str(args.db_path),
        "--agent-runtime",
        args.agent_runtime,
        "--storage-state-path",
        str(args.storage_state_path),
        "--poll-interval-seconds",
        str(args.poll_interval_seconds),
    ]
    if args.manual_login:
        frontend_command.append("--manual-login")
    if args.reset:
        backend_command.append("--reset")
    if args.show_tool_results:
        backend_command.append("--show-tool-results")
        frontend_command.append("--show-tool-results")
    if args.verbose:
        backend_command.append("--verbose")
        frontend_command.append("--verbose")

    backend_process = subprocess.Popen(backend_command, cwd=REPO_ROOT)
    frontend_process = subprocess.Popen(frontend_command, cwd=REPO_ROOT)
    try:
        while True:
            mission_state = store.get_mission_state()
            if mission_state.get("flag"):
                flag = mission_state["flag"]
                logger.info("Flag captured: {}", flag)
                print(flag)
                return 0
            if mission_state.get("status") == MissionStatus.FAILED.value:
                error = mission_state.get("error") or "Unknown mission failure."
                logger.error("Mission failed: {}", error)
                return 1
            if backend_process.poll() is not None and backend_process.returncode not in (0, None):
                logger.error("Backend agent exited with code {}.", backend_process.returncode)
                return backend_process.returncode or 1
            if frontend_process.poll() is not None and frontend_process.returncode not in (0, None):
                logger.error("Frontend agent exited with code {}.", frontend_process.returncode)
                return frontend_process.returncode or 1
            time.sleep(args.poll_interval_seconds)
    finally:
        for process in (backend_process, frontend_process):
            if process.poll() is None:
                process.terminate()
        for process in (backend_process, frontend_process):
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    raise SystemExit(main())
