"""Solve the AG3NTS evaluation task by combining static checks with OpenRouter."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar

REPO_ROOT_HINT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_HINT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_HINT))

from devs_utilities.ag3nts import (
    AG3NTS_VERIFY_URL,
    build_ag3nts_public_data_url,
    build_task_answer_payload,
    submit_task_answer,
)
from devs_utilities.bootstrap import bootstrap_repo
from devs_utilities.files import write_json
from devs_utilities.http import HttpRequestError, get_bytes
from devs_utilities.logging import configure_logging, logger as shared_logger
from devs_utilities.openrouter import (
    build_task_openrouter_client,
    build_task_site_name,
    OpenRouterClient,
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
logger = shared_logger.bind(component="sensors")


TASK_NAME = "evaluation"
DATA_URL = build_ag3nts_public_data_url("sensors.zip")
DEFAULT_MODEL = get_env("OPENROUTER_MODEL", "xiaomi/mimo-v2-omni") or "xiaomi/mimo-v2-omni"
DEFAULT_BATCH_SIZE = get_int_env("SENSORS_BATCH_SIZE", 10) or 10
DOWNLOAD_TIMEOUT_SECONDS = get_int_env("SENSORS_DOWNLOAD_TIMEOUT_SECONDS", 60) or 60
VERIFY_TIMEOUT_SECONDS = get_int_env("AG3NTS_TIMEOUT_SECONDS", 30) or 30

OUTPUT_DIR = Path(__file__).resolve().parent
ZIP_PATH = OUTPUT_DIR / "sensors.zip"
DATASET_DIR = OUTPUT_DIR / "dataset"
STATIC_ANALYSIS_PATH = OUTPUT_DIR / "static_analysis.json"
NOTE_CANDIDATES_PATH = OUTPUT_DIR / "note_candidates.json"
MODEL_REVIEW_PATH = OUTPUT_DIR / "model_review.json"
NOTE_CACHE_PATH = OUTPUT_DIR / "note_claim_cache.json"
FINAL_ANSWER_PATH = OUTPUT_DIR / "final_answer.json"
VERIFY_RESPONSE_PATH = OUTPUT_DIR / "last_verify_response.json"

NotePolarity = Literal["positive", "negative", "uncertain"]


@dataclass(frozen=True, slots=True)
class MeasurementRule:
    sensor_name: str
    field_name: str
    minimum: float
    maximum: float


MEASUREMENT_RULES = (
    MeasurementRule("temperature", "temperature_K", 553, 873),
    MeasurementRule("pressure", "pressure_bar", 60, 160),
    MeasurementRule("water", "water_level_meters", 5.0, 15.0),
    MeasurementRule("voltage", "voltage_supply_v", 229.0, 231.0),
    MeasurementRule("humidity", "humidity_percent", 40.0, 80.0),
)

NEGATIVE_NOTE_MARKERS = (
    "looks unstable",
    "feel inconsistent",
    "behavior is concerning",
    "readings look suspicious",
    "not the pattern i expected",
    "did not look right",
    "looks unusual",
    "raises serious doubts",
    "seems unreliable",
    "clear irregularity",
    "does not look healthy",
    "visible anomaly",
    "unexpected pattern",
    "questionable behavior",
    "requires attention",
    "quality is doubtful",
    "not comfortable with this result",
    "clearly off",
    "investigated immediately",
    "data flow appears compromised",
    "unstable characteristics",
    "looks technically doubtful",
    "does not match healthy history",
    "suggests a potential fault",
    "indicates probable degradation",
    "cannot be treated as normal",
    "signs of malfunction",
    "confidence in this report is low",
    "operating picture is not trustworthy",
    "strong signs of an issue",
    "does not pass a safety-minded review",
    "conflicts with our baseline",
    "changed in a risky way",
    "outside expected behavior",
    "consistency is clearly broken",
    "too erratic for approval",
    "root-cause analysis",
    "engineering analysis",
    "urgent verification",
    "replacement assessment",
    "revalidation",
    "deeper diagnostic task",
    "maintenance follow-up",
    "quality audit",
    "troubleshooting queue",
    "investigation is completed",
    "probable fault",
    "focused technical review",
    "on-site inspection",
)

POSITIVE_NOTE_MARKERS = (
    "appears nominal",
    "checks out",
    "looks reliable",
    "calm and predictable",
    "stays normal",
    "no irregular behavior",
    "quality is high",
    "no concerning drift",
    "fully stable",
    "diagnostics are positive",
    "no warning signs",
    "snapshot is reassuring",
    "remains healthy",
    "indicators remain strong",
    "without surprises",
    "looks clean",
    "state is consistent",
    "picture is solid",
    "confirms stability",
    "stay controlled",
    "stayed balanced",
    "looks steady",
    "condition is excellent",
    "remains coherent",
    "healthy cycles",
    "nothing suggests a fault condition",
    "normal operation continues without drift",
    "behaves exactly as intended",
    "pattern remains trustworthy",
    "align with normal patterns",
    "consistency is maintained across the board",
    "all values follow expected distribution",
    "operating envelope is respected",
    "everything remains inside expected limits",
    "baseline behavior is preserved",
    "all control checks passed cleanly",
    "safe operating zone",
    "system response remains predictable",
    "all measured channels stay in tolerance",
    "signal quality remains smooth and stable",
    "there is no sign of abnormal activity",
    "runtime conditions are comfortably normal",
    "there are no deviations to flag",
    "latest sample fits reference behavior",
    "status stays green",
    "confirmed regular operation",
    "approved as-is",
    "standard pass",
    "approved the report as normal",
    "left the setup untouched",
    "no intervention is necessary",
    "full approval",
    "no escalation was triggered",
    "no corrective steps were needed",
    "logged it as routine",
    "signed off this inspection",
    "case is cleared",
    "monitoring continues unchanged",
    "shift can proceed as planned",
    "only routine observation continues",
    "kept the system in normal mode",
    "keep the same operating plan",
    "cycle as healthy",
    "looks completely normal",
)

MODEL_SYSTEM_PROMPT = """You classify operator notes from power-plant sensor files.

Your task:
- Read only operator_notes.
- Decide whether the note claims everything is OK, claims there is a problem,
  or is too ambiguous to tell.
- Ignore unusual writing style and focus on meaning.
- Use tool calling before giving the final answer.

Use:
- "ok" when the note says the system looks normal, healthy, stable, approved,
  routine, or no action is needed.
- "problem" when the note says the result is suspicious, unstable, faulty,
  concerning, under investigation, escalated, or needs verification.
- "uncertain" only when the note does not clearly lean either way.

Return JSON only:
{
  "results": [
    {
      "key": "abc123",
      "note_claim": "ok|problem|uncertain",
      "reason": "short explanation"
    }
  ]
}
"""
MODEL_MAX_STEPS = 4

LEGACY_MODEL_ALIASES = {
    "openrouter/healer-alpha": "xiaomi/mimo-v2-omni",
}


@dataclass(frozen=True, slots=True)
class SensorRecord:
    file_id: str
    sensor_type: str
    timestamp: int
    temperature_K: float
    pressure_bar: float
    water_level_meters: float
    voltage_supply_v: float
    humidity_percent: float
    operator_notes: str

    @property
    def active_sensors(self) -> set[str]:
        return set(self.sensor_type.split("/"))

    def measurement_payload(self) -> dict[str, float]:
        return {
            "temperature_K": self.temperature_K,
            "pressure_bar": self.pressure_bar,
            "water_level_meters": self.water_level_meters,
            "voltage_supply_v": self.voltage_supply_v,
            "humidity_percent": self.humidity_percent,
        }

    def to_prompt_payload(self) -> dict[str, Any]:
        return {
            "id": self.file_id,
            "sensor_type": self.sensor_type,
            "timestamp": self.timestamp,
            **self.measurement_payload(),
            "operator_notes": self.operator_notes,
        }


@dataclass(frozen=True, slots=True)
class StaticFinding:
    file_id: str
    measurement_reasons: tuple[str, ...]
    note_polarity: NotePolarity
    note_parts: tuple[str, ...]

    @property
    def has_measurement_anomaly(self) -> bool:
        return bool(self.measurement_reasons)


@dataclass(frozen=True, slots=True)
class ModelCandidate:
    record: SensorRecord
    finding: StaticFinding
    candidate_reason: str

    def to_prompt_payload(self) -> dict[str, Any]:
        return {
            **self.record.to_prompt_payload(),
            "measurement_reasons": list(self.finding.measurement_reasons),
            "heuristic_note_polarity": self.finding.note_polarity,
            "candidate_reason": self.candidate_reason,
            "note_cache_key": note_cache_key(self.record.operator_notes),
        }


@dataclass(frozen=True, slots=True)
class ModelDecision:
    file_id: str
    is_anomaly: bool
    note_claim: str
    reason: str
    source: str


@dataclass(frozen=True, slots=True)
class NoteReview:
    key: str
    operator_notes: str
    note_claim: str
    reason: str
    source: str


T = TypeVar("T")


def note_cache_key(note: str) -> str:
    return hashlib.sha256(note.encode("utf-8")).hexdigest()[:16]


def load_note_cache(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}

    parsed = json.loads(path.read_text(encoding="utf-8"))
    items = parsed.get("items")
    if not isinstance(items, list):
        return {}

    cache: dict[str, dict[str, str]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        key = item.get("key")
        operator_notes = item.get("operator_notes")
        note_claim = item.get("note_claim")
        reason = item.get("reason")
        if (
            isinstance(key, str)
            and isinstance(operator_notes, str)
            and isinstance(note_claim, str)
            and isinstance(reason, str)
        ):
            cache[key] = {
                "operator_notes": operator_notes,
                "note_claim": note_claim,
                "reason": reason,
            }
    return cache


def save_note_cache(path: Path, cache: dict[str, dict[str, str]]) -> None:
    items = [
        {
            "key": key,
            "operator_notes": value["operator_notes"],
            "note_claim": value["note_claim"],
            "reason": value["reason"],
        }
        for key, value in sorted(cache.items())
    ]
    write_json(path, {"item_count": len(items), "items": items})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=None,
        help=(
            f"OpenRouter model override. Defaults to OPENROUTER_MODEL or {DEFAULT_MODEL}."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"How many candidate files to send in one OpenRouter batch. Default: {DEFAULT_BATCH_SIZE}.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Submit the computed answer to the AG3NTS verify endpoint.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Assume sensors.zip already exists locally.",
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Skip OpenRouter review and rely only on deterministic heuristics.",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Re-extract the dataset even when the file count already matches.",
    )
    return parser.parse_args()


def expected_file_count(zip_path: Path) -> int:
    with zipfile.ZipFile(zip_path) as archive:
        return sum(1 for name in archive.namelist() if not name.endswith("/"))


def count_dataset_files(dataset_dir: Path) -> int:
    return sum(1 for _ in dataset_dir.glob("*.json"))


def ensure_dataset(*, skip_download: bool, force_extract: bool) -> None:
    if not ZIP_PATH.exists():
        if skip_download:
            raise SystemExit(f"Missing dataset archive: {ZIP_PATH}")
        logger.info("Downloading dataset from {}", DATA_URL)
        ZIP_PATH.write_bytes(get_bytes(DATA_URL, timeout_seconds=DOWNLOAD_TIMEOUT_SECONDS))

    wanted_count = expected_file_count(ZIP_PATH)
    current_count = count_dataset_files(DATASET_DIR) if DATASET_DIR.exists() else 0
    if force_extract or current_count != wanted_count:
        logger.info("Extracting dataset to {} ({} files expected)", DATASET_DIR, wanted_count)
        if DATASET_DIR.exists():
            for path in DATASET_DIR.glob("*.json"):
                path.unlink()
        else:
            DATASET_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ZIP_PATH) as archive:
            archive.extractall(DATASET_DIR)
        current_count = count_dataset_files(DATASET_DIR)
        if current_count != wanted_count:
            raise SystemExit(
                f"Incomplete extraction: expected {wanted_count} JSON files, got {current_count}."
            )


def load_records(dataset_dir: Path) -> list[SensorRecord]:
    records: list[SensorRecord] = []
    for path in sorted(dataset_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.append(
            SensorRecord(
                file_id=path.stem,
                sensor_type=str(payload["sensor_type"]),
                timestamp=int(payload["timestamp"]),
                temperature_K=float(payload["temperature_K"]),
                pressure_bar=float(payload["pressure_bar"]),
                water_level_meters=float(payload["water_level_meters"]),
                voltage_supply_v=float(payload["voltage_supply_v"]),
                humidity_percent=float(payload["humidity_percent"]),
                operator_notes=str(payload["operator_notes"]).strip(),
            )
        )
    return records


def split_note_parts(note: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in note.split(", ") if part.strip())


def classify_note_polarity(note: str) -> NotePolarity:
    lowered = note.casefold()
    if any(marker in lowered for marker in NEGATIVE_NOTE_MARKERS):
        return "negative"
    if any(marker in lowered for marker in POSITIVE_NOTE_MARKERS):
        return "positive"
    return "uncertain"


def analyze_measurements(record: SensorRecord) -> tuple[str, ...]:
    reasons: list[str] = []
    active = record.active_sensors
    values = record.measurement_payload()

    for rule in MEASUREMENT_RULES:
        value = values[rule.field_name]
        if rule.sensor_name in active:
            if value == 0:
                reasons.append(f"{rule.field_name}:active_zero")
            elif value < rule.minimum or value > rule.maximum:
                reasons.append(f"{rule.field_name}:out_of_range")
        elif value != 0:
            reasons.append(f"{rule.field_name}:inactive_nonzero")

    return tuple(reasons)


def build_static_findings(records: list[SensorRecord]) -> dict[str, StaticFinding]:
    findings: dict[str, StaticFinding] = {}
    for record in records:
        findings[record.file_id] = StaticFinding(
            file_id=record.file_id,
            measurement_reasons=analyze_measurements(record),
            note_polarity=classify_note_polarity(record.operator_notes),
            note_parts=split_note_parts(record.operator_notes),
        )
    return findings


def build_note_candidates(
    records: list[SensorRecord],
    findings: dict[str, StaticFinding],
) -> list[ModelCandidate]:
    candidates: list[ModelCandidate] = []

    for record in records:
        finding = findings[record.file_id]
        part_count = len(finding.note_parts)

        if part_count != 3:
            candidates.append(
                ModelCandidate(
                    record=record,
                    finding=finding,
                    candidate_reason="unusual_note_structure",
                )
            )
            continue

        data_bad = finding.has_measurement_anomaly
        note_negative = finding.note_polarity == "negative"
        note_positive = finding.note_polarity == "positive"

        if data_bad and not note_negative:
            candidates.append(
                ModelCandidate(
                    record=record,
                    finding=finding,
                    candidate_reason="measurement_anomaly_with_non_alarm_note",
                )
            )
        elif not data_bad and not note_positive:
            candidates.append(
                ModelCandidate(
                    record=record,
                    finding=finding,
                    candidate_reason="healthy_measurements_with_non_normal_note",
                )
            )

    return candidates


def chunked(items: list[T], size: int) -> list[list[T]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def build_openrouter_client(model_override: str | None) -> OpenRouterClient:
    api_key = get_llm_api_key()
    base_url = get_llm_base_url()
    model = model_override or get_optional_env("OPENROUTER_MODEL") or DEFAULT_MODEL
    resolved_model = LEGACY_MODEL_ALIASES.get(model, model)
    if resolved_model != model:
        logger.warning(
            "Model {} is no longer available on OpenRouter, using {} instead.",
            model,
            resolved_model,
        )
    model = resolved_model
    timeout_raw = get_optional_env("OPENROUTER_TIMEOUT_SECONDS") or "60"
    try:
        timeout_seconds = max(10, int(timeout_raw))
    except ValueError as exc:
        raise SystemExit(
            f"OPENROUTER_TIMEOUT_SECONDS must be an integer, got: {timeout_raw}"
        ) from exc

    missing: list[str] = []
    if not api_key:
        missing.append("LLM_API_KEY")
    if not base_url:
        missing.append("LLM_BASE_URL")
    if missing:
        raise SystemExit(f"Missing required OpenRouter settings: {', '.join(missing)}")

    return build_task_openrouter_client(
        __file__,
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout_seconds=timeout_seconds,
    )


def parse_model_review(payload: str) -> list[NoteReview]:
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise OpenRouterError("OpenRouter returned invalid JSON for model review.") from exc

    results = parsed.get("results")
    if not isinstance(results, list):
        raise OpenRouterError("Model review payload is missing the results list.")

    reviews: list[NoteReview] = []
    for item in results:
        if not isinstance(item, dict):
            raise OpenRouterError("Model review item must be an object.")
        key = item.get("key")
        note_claim = item.get("note_claim")
        reason = item.get("reason")
        if not isinstance(key, str) or not key:
            raise OpenRouterError("Model review item is missing a key.")
        if not isinstance(note_claim, str) or not note_claim:
            raise OpenRouterError(f"Model review item {key} is missing note_claim.")
        if not isinstance(reason, str) or not reason:
            raise OpenRouterError(f"Model review item {key} is missing reason.")
        reviews.append(
            NoteReview(
                key=key,
                operator_notes="",
                note_claim=note_claim,
                reason=reason,
                source="model",
            )
        )

    return reviews


SENSOR_REVIEW_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_notes_batch_context",
            "description": "Return the operator notes that need classification in the current batch.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_review_payload",
            "description": "Validate that the proposed review payload contains exactly the expected note keys.",
            "parameters": {
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "key": {"type": "string"},
                                "note_claim": {"type": "string"},
                                "reason": {"type": "string"},
                            },
                            "required": ["key", "note_claim", "reason"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["results"],
                "additionalProperties": False,
            },
        },
    },
]


def build_sensor_review_handlers(batch: list[dict[str, str]]) -> dict[str, Any]:
    expected_keys = {item["key"] for item in batch}

    def get_notes_batch_context(_: dict[str, Any]) -> dict[str, Any]:
        return {"notes": batch}

    def validate_review_payload(arguments: dict[str, Any]) -> dict[str, Any]:
        raw_results = arguments.get("results")
        if not isinstance(raw_results, list):
            return {"is_valid": False, "message": "results must be a list"}
        received_keys = {
            item.get("key")
            for item in raw_results
            if isinstance(item, dict) and isinstance(item.get("key"), str)
        }
        return {
            "is_valid": received_keys == expected_keys,
            "expected_keys": sorted(expected_keys),
            "received_keys": sorted(str(key) for key in received_keys if isinstance(key, str)),
        }

    return {
        "get_notes_batch_context": get_notes_batch_context,
        "validate_review_payload": validate_review_payload,
    }


def execute_sensor_tool_call(tool_call: ToolCall, handlers: dict[str, Any]) -> dict[str, Any]:
    if tool_call.name not in handlers:
        raise OpenRouterError(f"Unknown sensor review tool: {tool_call.name!r}")
    result = handlers[tool_call.name](tool_call.arguments)
    return {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "name": tool_call.name,
        "content": json.dumps(result, ensure_ascii=False),
    }


def review_note_batch_with_tool_calling(
    client: OpenRouterClient,
    batch: list[dict[str, str]],
) -> list[NoteReview]:
    handlers = build_sensor_review_handlers(batch)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": MODEL_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "Classify the current batch of operator notes and return JSON only.",
        },
    ]
    for _ in range(MODEL_MAX_STEPS):
        result = client.create_completion(messages, tools=SENSOR_REVIEW_TOOLS)
        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": result.content or "",
        }
        if result.tool_calls:
            assistant_message["tool_calls"] = [
                tool_call.to_message_dict() for tool_call in result.tool_calls
            ]
            messages.append(assistant_message)
            for tool_call in result.tool_calls:
                messages.append(execute_sensor_tool_call(tool_call, handlers))
            continue
        if not result.content:
            raise OpenRouterError("OpenRouter returned no content for note review.")
        return parse_model_review(result.content)
    raise OpenRouterError("OpenRouter tool calling did not finish for sensor review.")


def build_file_decisions(
    candidates: list[ModelCandidate],
    note_reviews: dict[str, NoteReview],
) -> list[ModelDecision]:
    decisions: list[ModelDecision] = []

    for candidate in candidates:
        note_key = note_cache_key(candidate.record.operator_notes)
        review = note_reviews[note_key]
        measurement_bad = candidate.finding.has_measurement_anomaly

        if measurement_bad:
            if review.note_claim == "ok":
                reason = "Measurements are anomalous and the note claims everything is OK."
            elif review.note_claim == "problem":
                reason = "Measurements are anomalous and the note also reports a problem."
            else:
                reason = "Measurements are anomalous; the note is not clearly decisive."
            is_anomaly = True
        else:
            is_anomaly = review.note_claim == "problem"
            if review.note_claim == "problem":
                reason = "Measurements are normal but the note reports a problem."
            elif review.note_claim == "ok":
                reason = "Measurements are normal and the note also sounds normal."
            else:
                reason = "Measurements are normal and the note stays ambiguous."

        decisions.append(
            ModelDecision(
                file_id=candidate.record.file_id,
                is_anomaly=is_anomaly,
                note_claim=review.note_claim,
                reason=reason,
                source=review.source,
            )
        )

    return decisions


def review_note_candidates(
    client: OpenRouterClient,
    candidates: list[ModelCandidate],
    *,
    batch_size: int,
) -> tuple[list[ModelDecision], dict[str, int]]:
    cache = load_note_cache(NOTE_CACHE_PATH)
    note_reviews: dict[str, NoteReview] = {}
    unique_notes: dict[str, str] = {}
    pending_payloads: list[dict[str, str]] = []
    cache_hits = 0

    for candidate in candidates:
        note_key = note_cache_key(candidate.record.operator_notes)
        unique_notes.setdefault(note_key, candidate.record.operator_notes)

    for note_key, operator_notes in unique_notes.items():
        cached = cache.get(note_key)
        if cached and cached.get("operator_notes") == operator_notes:
            note_reviews[note_key] = NoteReview(
                key=note_key,
                operator_notes=operator_notes,
                note_claim=cached["note_claim"],
                reason=cached["reason"],
                source="cache",
            )
            cache_hits += 1
        else:
            pending_payloads.append(
                {"key": note_key, "operator_notes": operator_notes}
            )

    for batch_index, batch in enumerate(chunked(pending_payloads, batch_size), start=1):
        reviews = review_note_batch_with_tool_calling(client, batch)
        for review in reviews:
            operator_notes = unique_notes.get(review.key)
            if operator_notes is None:
                raise OpenRouterError(f"Model returned an unknown cache key: {review.key}")
            hydrated = NoteReview(
                key=review.key,
                operator_notes=operator_notes,
                note_claim=review.note_claim,
                reason=review.reason,
                source="model",
            )
            note_reviews[review.key] = hydrated
            cache[review.key] = {
                "operator_notes": operator_notes,
                "note_claim": review.note_claim,
                "reason": review.reason,
            }
        logger.info(
            "Reviewed note batch {}/{} ({} unique notes)",
            batch_index,
            len(chunked(pending_payloads, batch_size)),
            len(batch),
        )

    save_note_cache(NOTE_CACHE_PATH, cache)

    return build_file_decisions(candidates, note_reviews), {
        "candidate_count": len(candidates),
        "unique_note_count": len(unique_notes),
        "cache_hit_count": cache_hits,
        "model_note_count": len(pending_payloads),
    }


def heuristic_candidate_decisions(candidates: list[ModelCandidate]) -> list[ModelDecision]:
    note_reviews: dict[str, NoteReview] = {}
    for candidate in candidates:
        note_key = note_cache_key(candidate.record.operator_notes)
        if note_key in note_reviews:
            continue
        note_claim_map = {
            "positive": "ok",
            "negative": "problem",
            "uncertain": "uncertain",
        }
        note_reviews[note_key] = NoteReview(
            key=note_key,
            operator_notes=candidate.record.operator_notes,
            note_claim=note_claim_map[candidate.finding.note_polarity],
            reason=candidate.candidate_reason,
            source="heuristic",
        )
    return build_file_decisions(candidates, note_reviews)


def build_static_report(
    records: list[SensorRecord],
    findings: dict[str, StaticFinding],
    candidates: list[ModelCandidate],
) -> dict[str, Any]:
    measurement_anomalies = [
        {
            "id": finding.file_id,
            "measurement_reasons": list(finding.measurement_reasons),
            "note_polarity": finding.note_polarity,
        }
        for finding in findings.values()
        if finding.has_measurement_anomaly
    ]
    note_candidate_payload = [
        {
            "id": candidate.record.file_id,
            "note_cache_key": note_cache_key(candidate.record.operator_notes),
            "candidate_reason": candidate.candidate_reason,
            "note_polarity": candidate.finding.note_polarity,
            "measurement_reasons": list(candidate.finding.measurement_reasons),
        }
        for candidate in candidates
    ]
    return {
        "record_count": len(records),
        "measurement_anomaly_count": len(measurement_anomalies),
        "measurement_anomalies": measurement_anomalies,
        "note_candidate_count": len(note_candidate_payload),
        "note_candidate_unique_note_count": len(
            {candidate.record.operator_notes for candidate in candidates}
        ),
        "note_candidates": note_candidate_payload,
    }


def build_model_report(
    decisions: list[ModelDecision],
    cache_stats: dict[str, int] | None = None,
) -> dict[str, Any]:
    return {
        "cache_stats": cache_stats or {},
        "reviewed_file_count": len(decisions),
        "results": [
            {
                "id": decision.file_id,
                "is_anomaly": decision.is_anomaly,
                "note_claim": decision.note_claim,
                "reason": decision.reason,
                "source": decision.source,
            }
            for decision in decisions
        ],
    }


def build_final_answer(
    findings: dict[str, StaticFinding],
    decisions: list[ModelDecision],
) -> dict[str, Any]:
    final_ids = {
        file_id
        for file_id, finding in findings.items()
        if finding.has_measurement_anomaly
    }
    final_ids.update(
        decision.file_id for decision in decisions if decision.is_anomaly
    )
    answer = {"recheck": sorted(final_ids)}
    return build_task_answer_payload(get_course_api_key(), TASK_NAME, answer)


def verify_answer(payload: dict[str, Any]) -> Any:
    return submit_task_answer(
        AG3NTS_VERIFY_URL,
        api_key=str(payload["apikey"]),
        task=str(payload["task"]),
        answer=dict(payload["answer"]),
        timeout_seconds=VERIFY_TIMEOUT_SECONDS,
    )


def main() -> int:
    configure_logging(name="sensors")
    args = parse_args()

    ensure_dataset(skip_download=args.skip_download, force_extract=args.force_extract)
    records = load_records(DATASET_DIR)
    findings = build_static_findings(records)
    candidates = build_note_candidates(records, findings)

    static_report = build_static_report(records, findings, candidates)
    write_json(STATIC_ANALYSIS_PATH, static_report)
    write_json(NOTE_CANDIDATES_PATH, static_report["note_candidates"])

    logger.info(
        "Static analysis found {} measurement anomalies and {} note candidates.",
        static_report["measurement_anomaly_count"],
        static_report["note_candidate_count"],
    )

    decisions: list[ModelDecision] = []
    cache_stats: dict[str, int] = {}
    if candidates and not args.skip_model:
        client = build_openrouter_client(args.model)
        decisions, cache_stats = review_note_candidates(
            client,
            candidates,
            batch_size=max(1, args.batch_size),
        )
        logger.info(
            "Note cache stats: {} candidate files, {} unique notes, {} cache hits, {} model calls.",
            cache_stats["candidate_count"],
            cache_stats["unique_note_count"],
            cache_stats["cache_hit_count"],
            cache_stats["model_note_count"],
        )
        write_json(MODEL_REVIEW_PATH, build_model_report(decisions, cache_stats))
    else:
        if candidates:
            logger.warning(
                "Skipping OpenRouter review; using heuristic decisions for {} candidate notes.",
                len(candidates),
            )
            decisions = heuristic_candidate_decisions(candidates)
            cache_stats = {
                "candidate_count": len(candidates),
                "unique_note_count": len({candidate.record.operator_notes for candidate in candidates}),
                "cache_hit_count": 0,
                "model_note_count": 0,
            }
        write_json(MODEL_REVIEW_PATH, build_model_report(decisions, cache_stats))

    final_payload = build_final_answer(findings, decisions)
    write_json(FINAL_ANSWER_PATH, final_payload)
    logger.info(
        "Prepared final answer with {} file ids.",
        len(final_payload["answer"]["recheck"]),
    )

    if not args.verify:
        return 0

    try:
        response = verify_answer(final_payload)
    except HttpRequestError as exc:
        write_json(VERIFY_RESPONSE_PATH, exc.to_response_dict())
        raise SystemExit(str(exc)) from exc

    write_json(VERIFY_RESPONSE_PATH, response)
    logger.success("Verify response: {}", response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
