"""Submit or check the negotiations tool registration in AG3NTS."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

REPO_ROOT_HINT = Path(__file__).resolve().parents[1]
NEGOTIATIONS_DIR = Path(__file__).resolve().parent
NEGOTIATIONS_SRC_DIR = NEGOTIATIONS_DIR / "src"
for candidate in (
    str(REPO_ROOT_HINT),
    str(NEGOTIATIONS_DIR),
    str(NEGOTIATIONS_SRC_DIR),
):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from devs_utilities.ag3nts import AG3NTS_VERIFY_URL, submit_task_answer
from devs_utilities.bootstrap import bootstrap_repo
from devs_utilities.http import HttpRequestError, get_json
from negotiations_api.config import load_settings
from repo_env import get_course_api_key, get_optional_env


bootstrap_repo(__file__)
TASK_NAME = "negotiations"
DEFAULT_SETTINGS = load_settings()
DEFAULT_LOCAL_HOST = "127.0.0.1"
DEFAULT_SERVER_STARTUP_TIMEOUT_SECONDS = 20.0
DEFAULT_CHECK_INTERVAL_SECONDS = 5.0
DEFAULT_CHECK_ATTEMPTS = 12
PENDING_CHECK_MARKERS = (
    "pending",
    "processing",
    "in progress",
    "not ready",
    "retry",
    "wait",
    "try again",
    "sprawdz",
    "sprawdź",
    "sprobuj",
    "spróbuj",
    "w toku",
    "jeszcze",
)


def build_answer(
    public_base_url: str,
    *,
    tool_path: str = DEFAULT_SETTINGS.api_tool_path,
) -> dict[str, object]:
    """Build the tool declaration payload."""

    normalized_base = public_base_url.rstrip("/") + "/"
    tool_url = urljoin(normalized_base, tool_path.lstrip("/"))
    description = (
        "Szukaj miast dla jednego przedmiotu opisanego naturalnie w polu "
        "params. Podaj pojedynczy produkt z parametrami, np. "
        "'akumulator AGM 48V 150Ah' albo 'przetwornica 48V 3000W'. "
        "Odpowiedz zawiera dopasowany produkt i liste miast."
    )
    return {
        "tools": [
            {
                "URL": tool_url,
                "description": description,
            }
        ]
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-base-url")
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--use-ngrok",
        action="store_true",
        help="Expose the local negotiations API through the ngrok Python SDK.",
    )
    parser.add_argument(
        "--ngrok-authtoken",
        default=get_optional_env("NGROK_AUTHTOKEN"),
        help="ngrok authtoken. Defaults to NGROK_AUTHTOKEN.",
    )
    parser.add_argument(
        "--ngrok-domain",
        default=get_optional_env("NGROK_DOMAIN"),
        help="Optional reserved ngrok domain. Defaults to NGROK_DOMAIN.",
    )
    parser.add_argument(
        "--launch-server",
        action="store_true",
        help="Launch the local negotiations API server as a subprocess.",
    )
    parser.add_argument(
        "--wait-for-check",
        action="store_true",
        help="Keep the ngrok tunnel open and poll the verification status.",
    )
    parser.add_argument(
        "--local-host",
        default=DEFAULT_LOCAL_HOST,
        help=f"Host used for local health checks. Default: {DEFAULT_LOCAL_HOST}",
    )
    parser.add_argument(
        "--api-host",
        default=DEFAULT_SETTINGS.api_host,
        help=f"Host bind for the local server. Default: {DEFAULT_SETTINGS.api_host}",
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=DEFAULT_SETTINGS.api_port,
        help=f"Local port for the negotiations API. Default: {DEFAULT_SETTINGS.api_port}",
    )
    parser.add_argument(
        "--api-tool-path",
        default=DEFAULT_SETTINGS.api_tool_path,
        help=f"Tool path exposed by the server. Default: {DEFAULT_SETTINGS.api_tool_path}",
    )
    parser.add_argument(
        "--server-startup-timeout-seconds",
        type=float,
        default=DEFAULT_SERVER_STARTUP_TIMEOUT_SECONDS,
        help=(
            "How long to wait for the local server healthcheck. "
            f"Default: {DEFAULT_SERVER_STARTUP_TIMEOUT_SECONDS:.0f}s"
        ),
    )
    parser.add_argument(
        "--check-interval-seconds",
        type=float,
        default=DEFAULT_CHECK_INTERVAL_SECONDS,
        help=(
            "Delay between verification status polls when using --wait-for-check. "
            f"Default: {DEFAULT_CHECK_INTERVAL_SECONDS:.0f}s"
        ),
    )
    parser.add_argument(
        "--check-attempts",
        type=int,
        default=DEFAULT_CHECK_ATTEMPTS,
        help=(
            "Maximum number of status checks when using --wait-for-check. "
            f"Default: {DEFAULT_CHECK_ATTEMPTS}"
        ),
    )
    args = parser.parse_args()

    if args.api_port < 1:
        parser.error("--api-port must be >= 1.")
    if args.server_startup_timeout_seconds <= 0:
        parser.error("--server-startup-timeout-seconds must be > 0.")
    if args.check_interval_seconds < 0:
        parser.error("--check-interval-seconds must be >= 0.")
    if args.check_attempts < 1:
        parser.error("--check-attempts must be >= 1.")
    if args.check and any(
        (
            args.public_base_url,
            args.use_ngrok,
            args.launch_server,
            args.wait_for_check,
        )
    ):
        parser.error("--check cannot be combined with registration/tunnel options.")
    if not args.check and not args.public_base_url and not args.use_ngrok:
        parser.error("Provide --public-base-url or --use-ngrok unless --check is used.")
    if args.wait_for_check and not args.use_ngrok:
        parser.error("--wait-for-check requires --use-ngrok.")
    if args.public_base_url and args.use_ngrok:
        parser.error("--public-base-url cannot be combined with --use-ngrok.")
    return args


def submit_answer(api_key: str, answer: dict[str, object]) -> Any:
    """Submit a task answer payload."""

    return submit_task_answer(
        AG3NTS_VERIFY_URL,
        api_key=api_key,
        task=TASK_NAME,
        answer=answer,
        timeout_seconds=30,
    )


def check_status(api_key: str) -> Any:
    """Check the asynchronous task status."""

    return submit_answer(api_key, {"action": "check"})


def build_local_base_url(host: str, port: int) -> str:
    """Build a base URL for the local negotiations API."""

    return f"http://{host}:{port}/"


def build_healthcheck_url(host: str, port: int) -> str:
    """Build the health endpoint URL."""

    return urljoin(build_local_base_url(host, port), "health")


def build_ngrok_forward_kwargs(
    port: int,
    *,
    authtoken: str | None,
    domain: str | None,
) -> dict[str, object]:
    """Build keyword arguments for ngrok.forward()."""

    kwargs: dict[str, object] = {"addr": port}
    if authtoken:
        kwargs["authtoken"] = authtoken
    if domain:
        kwargs["domain"] = domain
    return kwargs


def extract_listener_url(listener: object) -> str:
    """Read the public URL from an ngrok listener object."""

    raw_url = getattr(listener, "url", None)
    if callable(raw_url):
        raw_url = raw_url()
    if not isinstance(raw_url, str) or not raw_url.strip():
        raise RuntimeError("The ngrok listener did not expose a usable public URL.")
    return raw_url.rstrip("/")


def close_listener(listener: object) -> None:
    """Close an ngrok listener if the SDK exposes a close method."""

    close = getattr(listener, "close", None)
    if callable(close):
        close()


def serialize_response(response: Any) -> str:
    """Serialize an AG3NTS response to text for logging and heuristics."""

    return json.dumps(response, ensure_ascii=False, sort_keys=True).casefold()


def is_pending_check_response(response: Any) -> bool:
    """Heuristically detect a still-pending asynchronous check response."""

    serialized = serialize_response(response)
    if "flg:" in serialized:
        return False
    return any(marker in serialized for marker in PENDING_CHECK_MARKERS)


def print_json_response(label: str, response: Any) -> None:
    """Print a labeled JSON response block."""

    print(f"{label}:")
    print(json.dumps(response, ensure_ascii=False, indent=2))


def build_server_environment(args: argparse.Namespace) -> dict[str, str]:
    """Build the environment for a spawned negotiations server process."""

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    pythonpath_parts = [str(NEGOTIATIONS_SRC_DIR)]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env["API_HOST"] = args.api_host
    env["API_PORT"] = str(args.api_port)
    env["API_TOOL_PATH"] = args.api_tool_path
    return env


def wait_for_server(host: str, port: int, timeout_seconds: float) -> None:
    """Poll the local health endpoint until the server is ready."""

    health_url = build_healthcheck_url(host, port)
    deadline = time.monotonic() + timeout_seconds
    last_error: str | None = None
    while time.monotonic() < deadline:
        try:
            response = get_json(health_url, timeout_seconds=3.0)
        except HttpRequestError as exc:
            last_error = str(exc)
            time.sleep(0.5)
            continue

        if isinstance(response, dict) and response.get("status") == "ok":
            return
        last_error = f"unexpected health payload: {response!r}"
        time.sleep(0.5)

    detail = last_error or "no response"
    raise RuntimeError(f"Negotiations server did not become healthy within {timeout_seconds}s: {detail}")


@contextmanager
def launched_server(args: argparse.Namespace) -> Iterator[subprocess.Popen[str] | None]:
    """Optionally launch the local negotiations server."""

    if not args.launch_server:
        yield None
        return

    process = subprocess.Popen(
        [sys.executable, "-m", "negotiations_api.server"],
        cwd=NEGOTIATIONS_DIR,
        env=build_server_environment(args),
        text=True,
    )
    try:
        wait_for_server(
            args.local_host,
            args.api_port,
            args.server_startup_timeout_seconds,
        )
        yield process
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


@contextmanager
def ngrok_tunnel(
    *,
    port: int,
    authtoken: str | None,
    domain: str | None,
) -> Iterator[str]:
    """Open a public tunnel using the ngrok Python SDK."""

    try:
        import ngrok  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on local install
        raise RuntimeError(
            "Missing the `ngrok` package. Install dependencies with `pip install -e .` "
            "or `pip install ngrok`."
        ) from exc

    listener = ngrok.forward(**build_ngrok_forward_kwargs(port, authtoken=authtoken, domain=domain))
    try:
        yield extract_listener_url(listener)
    finally:
        close_listener(listener)


def wait_for_check_result(
    api_key: str,
    *,
    interval_seconds: float,
    attempts: int,
) -> tuple[Any, bool]:
    """Poll the asynchronous status endpoint while the tunnel remains open."""

    last_response: Any = None
    for attempt in range(1, attempts + 1):
        if attempt == 1:
            time.sleep(interval_seconds)
        else:
            time.sleep(interval_seconds)
        last_response = check_status(api_key)
        print_json_response(f"Check response {attempt}/{attempts}", last_response)
        if not is_pending_check_response(last_response):
            return last_response, True
    return last_response, False


def register_with_public_url(
    api_key: str,
    *,
    public_base_url: str,
    tool_path: str,
) -> Any:
    """Register the tool declaration for a concrete public base URL."""

    answer = build_answer(public_base_url, tool_path=tool_path)
    response = submit_answer(api_key, answer)
    print_json_response("Registration response", response)
    return response


def main() -> int:
    """Parse arguments and submit the tool definition."""

    args = parse_args()
    api_key = get_course_api_key()

    if args.check:
        print_json_response("Check response", check_status(api_key))
        return 0

    if args.use_ngrok:
        with launched_server(args):
            if not args.launch_server:
                wait_for_server(
                    args.local_host,
                    args.api_port,
                    args.server_startup_timeout_seconds,
                )
            with ngrok_tunnel(
                port=args.api_port,
                authtoken=args.ngrok_authtoken,
                domain=args.ngrok_domain,
            ) as public_base_url:
                print(f"Ngrok public base URL: {public_base_url}")
                register_with_public_url(
                    api_key,
                    public_base_url=public_base_url,
                    tool_path=args.api_tool_path,
                )
                if not args.wait_for_check:
                    return 0
                _, completed = wait_for_check_result(
                    api_key,
                    interval_seconds=args.check_interval_seconds,
                    attempts=args.check_attempts,
                )
                return 0 if completed else 1

    register_with_public_url(
        api_key,
        public_base_url=args.public_base_url,
        tool_path=args.api_tool_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
