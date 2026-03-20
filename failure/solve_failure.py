"""Solve the AG3NTS failure task by condensing relevant outage logs."""

from __future__ import annotations

import argparse
import json
import sys
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from repo_env import get_env, load_repo_env


load_repo_env(__file__)


TASK_NAME = "failure"
VERIFY_URL = get_env("AG3NTS_VERIFY_URL")
DATA_BASE_URL = get_env("AG3NTS_DATA_BASE_URL")
LOG_PATTERN = re.compile(
    r"^\[(?P<timestamp>[^\]]+)\] \[(?P<level>[^\]]+)\] (?P<message>.*)$"
)

RAW_LOG_PATH = Path(__file__).with_name("failure.log")
CONDENSED_LOG_PATH = Path(__file__).with_name("final_logs.txt")
RESPONSE_PATH = Path(__file__).with_name("submission_response.json")

EXCLUDED_MESSAGES = {
    "Pressure jitter near STMTURB12 is above baseline. Automatic damping remains engaged.",
    "FIRMWARE watchdog acknowledged delayed subsystem poll. Retry timer is active.",
    "WTRPMP duty cycle is elevated for current load. Extended operation may reduce efficiency.",
    "Level sensor reconciliation for WTANK07 returned minor mismatch. Secondary read is requested.",
    "FIRMWARE reports a trend outside preferred startup envelope. Monitoring intensity has been increased.",
    "Advisory threshold crossed on WTANK07. Control loop continues with reduced tolerance.",
    "WSTPOOL2 shows moderate parameter drift during initialization. Automatic correction remains active.",
    "Preventive warning issued for WTRPMP due to unstable short-term readings. Escalation rules are armed.",
    "WTANK07 reports a trend outside preferred startup envelope. Monitoring intensity has been increased.",
    "ECCS8 shows moderate parameter drift during initialization. Automatic correction remains active.",
    "Preventive warning issued for WTANK07 due to unstable short-term readings. Escalation rules are armed.",
}

MESSAGE_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    (
        "Thermal drift on ",
        "",
    ),
    (
        " exceeds advisory threshold. Corrective ramp is queued.",
        " thermal drift above advisory limit; corrective ramp queued.",
    ),
    (
        " reported runaway outlet temperature. Protection interlock initiated reactor trip.",
        " runaway outlet temp; protection interlock tripped reactor.",
    ),
    (
        "Fill trajectory in ",
        "",
    ),
    (
        " is slower than expected. Cooling reserve may become constrained.",
        " refill slower than expected; cooling reserve tightening.",
    ),
    (
        " validation queue returned nonblocking fault set. Runtime proceeds in constrained mode.",
        " nonblocking faults in validation queue; runtime constrained.",
    ),
    (
        "Flow margin on ",
        "",
    ),
    (
        " is below preferred startup profile. Monitoring continues without immediate trip.",
        " flow margin below startup profile; no immediate trip.",
    ),
    (
        "Input ripple on ",
        "",
    ),
    (
        " crossed warning limits. Stability window is narrowed.",
        " input ripple crossed warning limit; stability window narrowed.",
    ),
    (
        " absorption path reached emergency boundary. Heat rejection is no longer sufficient.",
        " absorption path hit emergency boundary; heat rejection insufficient.",
    ),
    (
        " reports rising return temperature. Cooling headroom is decreasing.",
        " return temp rising; cooling headroom falling.",
    ),
    (
        "Waste heat relay to ",
        "",
    ),
    (
        " is approaching soft cap. Throughput tuning is required.",
        " waste-heat relay nearing soft cap; tuning required.",
    ),
    (
        " transient disturbed auxiliary pump control. Recovery completed with degraded margin.",
        " transient upset aux pump control; recovered with degraded margin.",
    ),
    (
        " indicates unstable refill trend. Available coolant inventory is no longer guaranteed.",
        " refill trend unstable; coolant inventory no longer guaranteed.",
    ),
    (
        " feedback loop exceeded correction budget. Thermal conversion rate is reduced.",
        " feedback loop exceeded correction budget; thermal conversion reduced.",
    ),
    (
        " suction profile is inconsistent with expected coolant volume. Mechanical stress is increasing.",
        " suction profile inconsistent with coolant volume; mechanical stress rising.",
    ),
    (
        " core cooling cannot maintain safe gradient. Immediate protective actions are required.",
        " core cooling cannot maintain safe gradient; immediate protection required.",
    ),
    (
        "Heat transfer path to ",
        "",
    ),
    (
        " is saturated. Dissipation lag continues to accumulate.",
        " heat-transfer path saturated; dissipation lag growing.",
    ),
    (
        "Cooling efficiency on ",
        "",
    ),
    (
        " dropped below operational target. Compensating commands did not recover nominal state.",
        " cooling efficiency below operational target; compensation failed.",
    ),
    (
        " level estimate dropped near minimum reserve line. Automatic refill request timed out.",
        " level near minimum reserve; auto-refill timed out.",
    ),
    (
        " return circuit temperature rose faster than prediction. Emergency bias remains armed.",
        " return-circuit temp rose faster than predicted; emergency bias armed.",
    ),
    (
        " reported repeated cavitation signatures. Output pressure cannot be held at requested level.",
        " repeated cavitation; output pressure cannot hold setpoint.",
    ),
    (
        " coolant level is below critical threshold. Shutdown logic is moving to hard trip stage.",
        " coolant level below critical threshold; hard-trip logic engaged.",
    ),
    (
        " lost stable prime under peak thermal demand. Core loop continuity is compromised.",
        " lost stable prime under peak thermal demand; core loop continuity compromised.",
    ),
    (
        " entered emergency guard branch after repeated safety faults. Manual override is locked.",
        " entered emergency guard after repeated safety faults; manual override locked.",
    ),
    (
        " decoupling sequence forced by thermal risk. Energy conversion is terminated.",
        " decoupling forced by thermal risk; energy conversion terminated.",
    ),
    (
        "Cross-check between ",
        "Cross-check ",
    ),
    (
        " and hardware interface map did not complete successfully. Compatibility verification remains unresolved for startup state.",
        " vs hardware interface map failed; startup compatibility unresolved.",
    ),
    (
        " can no longer sustain stable feed for cooling auxiliaries. Critical loads are shedding.",
        " can no longer sustain stable feed for cooling auxiliaries; critical loads shedding.",
    ),
    (
        "Power stability on ",
        "",
    ),
    (
        " is highly unstable under startup load. Adding an additional power source is strongly recommended.",
        " power highly unstable under startup load; extra power source recommended.",
    ),
    (
        "Safety bootstrap read missing environment marker SAFETY_CHECK=pass. FIRMWARE continues in restricted validation mode.",
        "FIRMWARE missing SAFETY_CHECK=pass; restricted validation mode.",
    ),
    (
        " entered critical protection state during startup. Immediate shutdown safeguards remain active.",
        " entered critical protection during startup; shutdown safeguards active.",
    ),
    (
        " failed a recovery step in the active sequence. The subsystem remains in degraded operation mode.",
        " failed a recovery step; subsystem remains degraded.",
    ),
    (
        "Cooling reserve trend in ",
        "",
    ),
    (
        " keeps falling during load rise. ECCS8 is approaching a nonrecoverable limit.",
        " cooling reserve falling during load rise; ECCS8 nearing nonrecoverable limit.",
    ),
    (
        " remains partially filled. Shutdown criteria are approaching.",
        " remains partially filled; shutdown criteria nearing.",
    ),
    (
        "Operational fault persisted on ",
        "",
    ),
    (
        " after retry cycle. Performance constraints are now enforced.",
        " fault persisted after retry cycle; performance constrained.",
    ),
    (
        " returned inconsistent feedback under load. Automatic fallback path has been applied.",
        " returned inconsistent feedback under load; auto-fallback applied.",
    ),
    (
        "Coolant level in ",
        "",
    ),
    (
        " is below critical reserve for sustained operation. Protective shutdown path is being enforced.",
        " coolant level below critical reserve for sustained operation; protective shutdown enforced.",
    ),
    (
        "Control response from ",
        "",
    ),
    (
        " exceeded error budget. Further recovery attempts are limited.",
        " control response exceeded error budget; further recovery limited.",
    ),
    (
        " cannot remove heat with the current ",
        " cannot remove heat with current ",
    ),
    (
        " volume. Reactor protection initiates critical stop.",
        " volume; reactor protection critical stop.",
    ),
    (
        "Critical boundary exceeded on ECCS8. Emergency interlock keeps the reactor in protected mode.",
        "ECCS8 critical boundary exceeded; emergency interlock keeps reactor in protected mode.",
    ),
    (
        "Coolant inventory in ",
        "",
    ),
    (
        " is below critical threshold for full-loop operation. ECCS8 cannot guarantee reactor heat removal and automatic shutdown is mandatory.",
        " coolant inventory below critical threshold for full-loop operation; ECCS8 cannot guarantee heat removal, auto shutdown mandatory.",
    ),
    (
        "Insufficient cooling capacity confirmed after incomplete ",
        "Cooling capacity insufficient after incomplete ",
    ),
    (
        " refill. Reactor protection system executes final shutdown sequence.",
        " refill; protection system executes final shutdown.",
    ),
    (
        "Final trip complete because ",
        "Final trip: ",
    ),
    (
        " remained under critical water level. FIRMWARE confirms safe shutdown state with all core operations halted.",
        " remained under critical water level; FIRMWARE confirms safe shutdown, core halted.",
    ),
)


@dataclass(slots=True)
class LogEntry:
    timestamp: str
    level: str
    message: str

    @property
    def minute_timestamp(self) -> str:
        return self.timestamp[:16]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help=f"Save the downloaded source log to {RAW_LOG_PATH.name}.",
    )
    parser.add_argument(
        "--print-logs",
        action="store_true",
        help="Print the condensed log payload to stdout.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Submit the condensed logs to the AG3NTS verify endpoint.",
    )
    return parser.parse_args()


def get_api_key() -> str:
    api_key = get_env("AG3NTS_API_KEY")
    if not api_key:
        raise ValueError("Missing AG3NTS_API_KEY in .env.")
    return api_key


def build_log_url(api_key: str) -> str:
    if not DATA_BASE_URL:
        raise ValueError("Missing AG3NTS_DATA_BASE_URL in .env.")
    return f"{DATA_BASE_URL.rstrip('/')}/{api_key}/{TASK_NAME}.log"


def http_get_text(url: str) -> str:
    request = Request(url, headers={"Accept": "text/plain,*/*;q=0.8"})
    try:
        with urlopen(request, timeout=60) as response:
            return response.read().decode("utf-8")
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} while fetching {url}: {details}") from exc
    except URLError as exc:
        raise RuntimeError(f"Network error while fetching {url}: {exc.reason}") from exc


def http_post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=60) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"raw": raw}
        payload["http_status"] = exc.code
        raise RuntimeError(json.dumps(payload, ensure_ascii=False, indent=2)) from exc

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}


def parse_entries(raw_text: str) -> list[LogEntry]:
    entries: list[LogEntry] = []
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = LOG_PATTERN.match(stripped)
        if not match:
            raise ValueError(f"Unexpected log line format: {stripped}")
        entries.append(
            LogEntry(
                timestamp=match.group("timestamp"),
                level=match.group("level"),
                message=match.group("message"),
            )
        )
    return entries


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def select_relevant_entries(entries: list[LogEntry]) -> list[LogEntry]:
    seen_messages: set[str] = set()
    selected: list[LogEntry] = []

    for entry in entries:
        if entry.level == "INFO":
            continue
        if entry.message in EXCLUDED_MESSAGES:
            continue
        if entry.message in seen_messages:
            continue
        seen_messages.add(entry.message)
        selected.append(entry)

    return selected


def compress_message(message: str) -> str:
    compressed = message
    for old, new in MESSAGE_REPLACEMENTS:
        compressed = compressed.replace(old, new)
    compressed = re.sub(r"\s+", " ", compressed).strip()
    return compressed


def build_condensed_logs(entries: list[LogEntry]) -> str:
    selected = select_relevant_entries(entries)
    condensed_lines = [
        f"[{entry.minute_timestamp}] [{entry.level}] {compress_message(entry.message)}"
        for entry in selected
    ]
    return "\n".join(condensed_lines)


def summarize_source(entries: list[LogEntry], raw_text: str, condensed_logs: str) -> str:
    level_counts: dict[str, int] = {}
    for entry in entries:
        level_counts[entry.level] = level_counts.get(entry.level, 0) + 1

    selected_entries = condensed_logs.splitlines()
    summary = {
        "source_lines": len(entries),
        "source_chars": len(raw_text),
        "source_approx_tokens": estimate_tokens(raw_text),
        "levels": level_counts,
        "selected_lines": len(selected_entries),
        "selected_chars": len(condensed_logs),
        "selected_approx_tokens": estimate_tokens(condensed_logs),
        "output_file": str(CONDENSED_LOG_PATH),
    }
    return json.dumps(summary, ensure_ascii=False, indent=2)


def extract_flag(payload: Any) -> str | None:
    if isinstance(payload, str):
        return payload if payload.startswith("{FLG:") else None

    if isinstance(payload, dict):
        for value in payload.values():
            flag = extract_flag(value)
            if flag:
                return flag
        return None

    if isinstance(payload, list):
        for item in payload:
            flag = extract_flag(item)
            if flag:
                return flag

    return None


def main() -> int:
    args = parse_args()
    api_key = get_api_key()
    raw_text = http_get_text(build_log_url(api_key))
    entries = parse_entries(raw_text)
    condensed_logs = build_condensed_logs(entries)

    if args.save_raw:
        RAW_LOG_PATH.write_text(raw_text, encoding="utf-8")

    CONDENSED_LOG_PATH.write_text(condensed_logs + "\n", encoding="utf-8")
    print(summarize_source(entries, raw_text, condensed_logs))

    if args.print_logs:
        print("\n--- condensed logs ---")
        print(condensed_logs)

    if not args.verify:
        return 0

    if not VERIFY_URL:
        raise ValueError("Missing AG3NTS_VERIFY_URL in .env.")

    payload = {
        "apikey": api_key,
        "task": TASK_NAME,
        "answer": {"logs": condensed_logs},
    }
    response = http_post_json(VERIFY_URL, payload)
    RESPONSE_PATH.write_text(
        json.dumps(response, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\n--- verify response ---")
    print(json.dumps(response, ensure_ascii=False, indent=2))

    flag = extract_flag(response)
    if flag:
        print(f"\nFlag: {flag}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
