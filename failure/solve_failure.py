"""Solve the AG3NTS failure task by condensing relevant outage logs."""

from __future__ import annotations

import argparse
import json
import sys
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT_HINT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_HINT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_HINT))

from devs_utilities.ag3nts import (
    AG3NTS_VERIFY_URL,
    build_ag3nts_task_data_url,
    submit_task_answer,
)
from devs_utilities.bootstrap import bootstrap_repo
from devs_utilities.files import write_json
from devs_utilities.flags import extract_flag
from devs_utilities.http import HttpRequestError, get_text
from devs_utilities.logging import configure_logging, logger as shared_logger
from devs_utilities.openrouter import (
    build_task_openrouter_client,
    OpenRouterClient,
    OpenRouterError,
    parse_json_object_content,
)
from repo_env import get_course_api_key, get_env, get_int_env, get_optional_env


REPO_ROOT = bootstrap_repo(__file__)
logger = shared_logger.bind(component="failure")


TASK_NAME = "failure"
REQUEST_TIMEOUT_SECONDS = get_int_env("FAILURE_TIMEOUT_SECONDS", 60) or 60
DEFAULT_MODEL = (
    get_optional_env("OPENROUTER_MODEL")
    or get_optional_env("LLM_MODEL")
    or "openai/gpt-4.1-mini"
)
DEFAULT_OPENROUTER_TIMEOUT_SECONDS = get_int_env("OPENROUTER_TIMEOUT_SECONDS", 60) or 60
LOG_PATTERN = re.compile(
    r"^\[(?P<timestamp>[^\]]+)\] \[(?P<level>[^\]]+)\] (?P<message>.*)$"
)

RAW_LOG_PATH = Path(__file__).with_name("failure.log")
CONDENSED_LOG_PATH = Path(__file__).with_name("final_logs.txt")
RESPONSE_PATH = Path(__file__).with_name("submission_response.json")
MODEL_PLAN_PATH = Path(__file__).with_name("last_model_logs.json")
MODEL_SYSTEM_PROMPT = """You compress already-selected outage log lines.

Rules:
- Do not invent, remove, reorder, or merge distinct incidents unless the meaning
  stays exactly intact.
- Keep one incident per output line.
- Preserve the leading timestamp and level block in each line.
- Keep component names and causal facts.
- Make wording shorter and more technical.
- Stay within 1500 tokens total.

Return JSON only:
{"logs":["[2026-04-01 10:15] [ERROR] ..."],"reason":"short explanation"}
"""

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
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Disable OpenRouter log compression and use deterministic condensation only.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"OpenRouter model override. Default: {DEFAULT_MODEL}.",
    )
    return parser.parse_args()


def get_api_key() -> str:
    api_key = get_course_api_key()
    if not api_key:
        raise ValueError("Missing COURSE_API_KEY in the local repository config.")
    return api_key


def build_log_url(api_key: str) -> str:
    return build_ag3nts_task_data_url(api_key, f"{TASK_NAME}.log")


def http_get_text(url: str) -> str:
    return get_text(
        url,
        headers={"Accept": "text/plain,*/*;q=0.8"},
        timeout_seconds=REQUEST_TIMEOUT_SECONDS,
        errors="strict",
    )


def http_post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        return submit_task_answer(
            url,
            api_key=str(payload["apikey"]),
            task=str(payload["task"]),
            answer=dict(payload["answer"]),
            timeout_seconds=REQUEST_TIMEOUT_SECONDS,
        )
    except HttpRequestError as exc:
        raise RuntimeError(json.dumps(exc.to_response_dict(), ensure_ascii=False, indent=2)) from exc


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


def build_optional_openrouter_client(args: argparse.Namespace) -> OpenRouterClient | None:
    if args.skip_model:
        return None

    api_key = (
        get_optional_env("OPENROUTER_API_KEY")
        or get_optional_env("LLM_API_KEY")
        or ""
    ).strip()
    base_url = (
        get_optional_env("OPENROUTER_BASE_URL")
        or get_optional_env("LLM_BASE_URL")
        or ""
    ).strip()
    model = (args.model or DEFAULT_MODEL).strip()
    if not api_key or not base_url or not model:
        return None

    return build_task_openrouter_client(
        __file__,
        api_key=api_key,
        base_url=base_url,
        model=model,
        task_name=TASK_NAME,
        timeout_seconds=float(max(30, DEFAULT_OPENROUTER_TIMEOUT_SECONDS)),
    )


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


def build_condensed_log_lines(entries: list[LogEntry]) -> list[str]:
    selected = select_relevant_entries(entries)
    return [
        f"[{entry.minute_timestamp}] [{entry.level}] {compress_message(entry.message)}"
        for entry in selected
    ]


def build_condensed_logs(entries: list[LogEntry]) -> str:
    return "\n".join(build_condensed_log_lines(entries))


def validate_model_log_lines(raw_lines: Any) -> list[str]:
    if not isinstance(raw_lines, list) or not raw_lines:
        raise OpenRouterError("Model log payload must be a non-empty list.")
    normalized: list[str] = []
    for line in raw_lines:
        if not isinstance(line, str):
            raise OpenRouterError("Every condensed log line must be a string.")
        stripped = " ".join(line.split())
        if not stripped or not stripped.startswith("["):
            raise OpenRouterError("Condensed log line must preserve the log prefix.")
        normalized.append(stripped)
    if estimate_tokens("\n".join(normalized)) > 1500:
        raise OpenRouterError("Model-condensed logs exceeded the 1500-token budget.")
    return normalized


def condense_logs_with_openrouter(
    condensed_lines: list[str],
    client: OpenRouterClient | None,
) -> tuple[list[str], str, str]:
    if client is None:
        return condensed_lines, "deterministic", "OpenRouter unavailable."

    completion = client.create_completion(
        [
            {"role": "system", "content": MODEL_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Condense these outage log lines without changing their meaning:\n"
                    + "\n".join(condensed_lines)
                ),
            },
        ]
    )
    if not completion.content:
        raise OpenRouterError("Log compressor returned no content.")
    payload = parse_json_object_content(completion.content)
    model_lines = validate_model_log_lines(payload.get("logs"))
    reason = str(payload.get("reason", "")).strip() or "OpenRouter compressed the selected logs."
    return model_lines, "openrouter", reason


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


def main() -> int:
    configure_logging(name="failure")
    args = parse_args()
    api_key = get_api_key()
    raw_text = http_get_text(build_log_url(api_key))
    entries = parse_entries(raw_text)
    condensed_lines = build_condensed_log_lines(entries)
    condensed_logs = "\n".join(condensed_lines)

    log_source = "deterministic"
    log_reason = "Deterministic condensation."
    try:
        condensed_lines, log_source, log_reason = condense_logs_with_openrouter(
            condensed_lines,
            build_optional_openrouter_client(args),
        )
        condensed_logs = "\n".join(condensed_lines)
    except OpenRouterError as exc:
        logger.warning("OpenRouter condensation failed, using deterministic logs: {}", exc)

    write_json(
        MODEL_PLAN_PATH,
        {
            "source": log_source,
            "reason": log_reason,
            "estimated_tokens": estimate_tokens(condensed_logs),
            "line_count": len(condensed_lines),
        },
    )

    if args.save_raw:
        RAW_LOG_PATH.write_text(raw_text, encoding="utf-8")

    CONDENSED_LOG_PATH.write_text(condensed_logs + "\n", encoding="utf-8")
    logger.info("Summary:\n{}", summarize_source(entries, raw_text, condensed_logs))
    logger.info("Condensation source: {} ({})", log_source, log_reason)

    if args.print_logs:
        logger.info("Condensed logs:\n{}", condensed_logs)

    if not args.verify:
        return 0

    payload = {
        "apikey": api_key,
        "task": TASK_NAME,
        "answer": {"logs": condensed_logs},
    }
    response = http_post_json(AG3NTS_VERIFY_URL, payload)
    write_json(RESPONSE_PATH, response)
    logger.info("Verify response:\n{}", json.dumps(response, ensure_ascii=False, indent=2))

    flag = extract_flag(response)
    if flag:
        logger.success("Flag: {}", flag)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
