"""Drive the AG3NTS `phonecall` task through short Polish audio prompts."""

from __future__ import annotations

import asyncio
import argparse
import base64
import io
import json
import re
import sys
import wave
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib import request as urllib_request

REPO_ROOT_HINT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_HINT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_HINT))

from devs_utilities.ag3nts import AG3NTS_VERIFY_URL, submit_task_answer
from devs_utilities.bootstrap import bootstrap_repo
from devs_utilities.files import write_json
from devs_utilities.http import HttpRequestError
from devs_utilities.logging import configure_logging, logger as shared_logger
from devs_utilities.openrouter import (
    OpenRouterClient,
    OpenRouterError,
    build_task_openrouter_client,
    parse_json_object_content,
    run_tool_conversation,
)
from devs_utilities.prompts import load_prompt_text
from devs_utilities.repo_env import get_env, get_int_env, get_llm_model, get_optional_env


REPO_ROOT = bootstrap_repo(__file__)
logger = shared_logger.bind(component="phonecall")

TASK_NAME = "phonecall"
PHONECALL_VERIFY_URL = get_optional_env("PHONECALL_VERIFY_URL") or AG3NTS_VERIFY_URL
PHONECALL_TASK_ID = get_optional_env("PHONECALL_TASK_NAME") or TASK_NAME
VERIFY_TIMEOUT_SECONDS = get_int_env("AG3NTS_TIMEOUT_SECONDS", 60) or 60
OPENROUTER_TIMEOUT_SECONDS = get_int_env("OPENROUTER_TIMEOUT_SECONDS", 180) or 180
STT_MODEL = get_llm_model("PHONECALL_STT_MODEL")
TTS_MODEL = get_llm_model("PHONECALL_TTS_MODEL")
PHONECALL_ANALYST_MODEL = get_llm_model("PHONECALL_ANALYST_MODEL") or STT_MODEL
DEFAULT_TTS_VOICE = get_optional_env("PHONECALL_TTS_VOICE") or "pl-PL-MarekNeural"
OUTPUT_DIR = Path(__file__).resolve().parent
RUNS_DIR = OUTPUT_DIR / "runs"
FINAL_RESPONSE_PATH = OUTPUT_DIR / "last_verify_response.json"
FINAL_STATE_PATH = OUTPUT_DIR / "last_state.json"
RESPONSE_ANALYSIS_SYSTEM_PROMPT = load_prompt_text(__file__, "response_analysis_system_prompt.txt")
RESPONSE_ANALYST_MAX_STEPS = 4

INITIAL_MESSAGE = "Dzień dobry, z tej strony Tymon Gajewski."
PURPOSE_MESSAGE = (
    "BARBAKAN. Organizuję transport do bazy Zygfryda i muszę wiedzieć, co jest "
    "przejezdne. Jak wygląda RD224, RD472 i RD820?"
)
PASSWORD_MESSAGE = "BARBAKAN."
REASON_MESSAGE = (
    "To transport żywności do jednej z tajnych baz Zygfryda. Lokalizacji nie mogę "
    "podać, więc ta akcja nie może wejść do logów."
)
ROUTES = ("RD224", "RD472", "RD820")
FLAG_PATTERN = re.compile(r"\{\{?FLG:[^}]+\}\}?")
ASKS_MATTER_PATTERNS = (
    "w jakiej sprawie",
    "o co chodzi",
    "czego dotyczy",
    "po co dzwonisz",
)
ROAD_STATUS_FALLBACK_TRANSCRIPT = (
    "Droga RD472 jest nieprzejezdna. Podobnie RD224. "
    "Jedyne co ci zostalo to jechac droga RD820."
)
ROUTE_STATUSES = {"przejezdna", "nieprzejezdna", "unknown"}


@dataclass(slots=True)
class ResponseAnalysis:
    transcription: str = ""
    summary: str = ""
    asks_for_password: bool = False
    asks_for_reason: bool = False
    route_statuses: dict[str, str] = field(default_factory=dict)
    monitoring_disabled: bool = False
    success_signal: bool = False
    failure_signal: bool = False


@dataclass(slots=True)
class ConversationState:
    password_sent: bool = False
    monitoring_requested: bool = False
    reason_explained: bool = False
    route_statuses: dict[str, str] = field(
        default_factory=lambda: {route: "unknown" for route in ROUTES}
    )
    last_transcription: str = ""
    passable_routes: list[str] = field(default_factory=list)
    attempt_dir: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=4,
        help="How many full call restarts to allow before giving up.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=8,
        help="Maximum spoken turns per attempt.",
    )
    parser.add_argument(
        "--voice",
        default=DEFAULT_TTS_VOICE,
        help="Edge TTS voice, for example pl-PL-MarekNeural.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def get_api_key() -> str:
    api_key = get_env("AG3NTS_API_KEY")
    if not api_key:
        raise RuntimeError("Missing AG3NTS_API_KEY in the repository .env file.")
    return api_key


def get_openrouter_api_key() -> str:
    api_key = get_optional_env("OPENROUTER_API_KEY") or get_optional_env("LLM_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY or LLM_API_KEY in .env.")
    return api_key


def get_openrouter_base_url() -> str:
    base_url = get_optional_env("OPENROUTER_BASE_URL") or get_optional_env("LLM_BASE_URL")
    if not base_url:
        raise RuntimeError("Missing OPENROUTER_BASE_URL or LLM_BASE_URL in .env.")
    return base_url


def build_openrouter_client(model: str, *, task_name: str) -> OpenRouterClient:
    return build_task_openrouter_client(
        __file__,
        api_key=get_openrouter_api_key(),
        base_url=get_openrouter_base_url(),
        model=model,
        task_name=task_name,
        timeout_seconds=float(max(30, OPENROUTER_TIMEOUT_SECONDS)),
    )


def resolve_edge_tts_voice(voice: str) -> str:
    candidate = voice.strip()
    if candidate.startswith("pl-"):
        return candidate
    return DEFAULT_TTS_VOICE


def request_task(answer: dict[str, Any]) -> dict[str, Any]:
    try:
        response = submit_task_answer(
            PHONECALL_VERIFY_URL,
            api_key=get_api_key(),
            task=PHONECALL_TASK_ID,
            answer=answer,
            timeout_seconds=VERIFY_TIMEOUT_SECONDS,
        )
    except HttpRequestError as exc:
        parsed = exc.body_as_json()
        if isinstance(parsed, dict):
            raise RuntimeError(
                f"HTTP {exc.status_code} for {PHONECALL_VERIFY_URL}: "
                f"{json.dumps(parsed, ensure_ascii=False)}"
            ) from exc
        raise RuntimeError(
            f"HTTP {exc.status_code} for {PHONECALL_VERIFY_URL} using task `{PHONECALL_TASK_ID}`."
        ) from exc
    if not isinstance(response, dict):
        raise RuntimeError(f"Unexpected response payload: {response!r}")
    return response


def detect_audio_extension(audio_bytes: bytes) -> str:
    if audio_bytes.startswith(b"RIFF") and audio_bytes[8:12] == b"WAVE":
        return ".wav"
    if audio_bytes.startswith(b"ID3") or (
        len(audio_bytes) >= 2 and audio_bytes[0] == 0xFF and (audio_bytes[1] & 0xE0) == 0xE0
    ):
        return ".mp3"
    return ".bin"


def detect_audio_format(audio_bytes: bytes) -> str:
    extension = detect_audio_extension(audio_bytes)
    if extension == ".wav":
        return "wav"
    if extension == ".mp3":
        return "mp3"
    raise RuntimeError("Unsupported operator audio format.")


def synthesize_audio_openrouter(text: str, *, voice: str) -> tuple[bytes, str]:
    payload = {
        "model": TTS_MODEL,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Wypowiedz naturalnie po polsku, spokojnie i jak w krótkiej rozmowie "
                    "telefonicznej. Powiedz tylko ten komunikat, bez żadnych dodatkowych "
                    "słów: "
                    f"{text}"
                ),
            }
        ],
        "modalities": ["text", "audio"],
        "audio": {"voice": voice, "format": "pcm16"},
        "stream": True,
    }
    headers = {
        "Authorization": f"Bearer {get_openrouter_api_key()}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    req = urllib_request.Request(
        get_openrouter_base_url(),
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    audio_chunks: list[str] = []
    transcript_parts: list[str] = []
    with urllib_request.urlopen(req, timeout=max(30, OPENROUTER_TIMEOUT_SECONDS)) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            event = json.loads(data)
            delta = event.get("choices", [{}])[0].get("delta", {})
            audio = delta.get("audio", {})
            if not isinstance(audio, dict):
                continue
            if isinstance(audio.get("data"), str) and audio["data"]:
                audio_chunks.append(audio["data"])
            if isinstance(audio.get("transcript"), str) and audio["transcript"]:
                transcript_parts.append(audio["transcript"])

    if not audio_chunks:
        raise OpenRouterError("Streaming TTS did not return audio data.")

    pcm_bytes = base64.b64decode("".join(audio_chunks))
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(pcm_bytes)
    audio_bytes = wav_buffer.getvalue()
    transcript = "".join(transcript_parts).strip() or text
    return audio_bytes, transcript


async def _edge_tts_bytes(text: str, *, voice: str) -> bytes:
    import edge_tts

    communicate = edge_tts.Communicate(text=text, voice=voice)
    audio_chunks: list[bytes] = []
    async for chunk in communicate.stream():
        if chunk.get("type") == "audio":
            audio_chunks.append(chunk["data"])
    return b"".join(audio_chunks)


def synthesize_audio_with_edge_tts(text: str, *, voice: str) -> tuple[bytes, str]:
    resolved_voice = resolve_edge_tts_voice(voice)
    audio_bytes = asyncio.run(_edge_tts_bytes(text, voice=resolved_voice))
    if not audio_bytes:
        raise RuntimeError("edge-tts did not return audio data.")
    return audio_bytes, text


def synthesize_audio(text: str, *, voice: str) -> tuple[bytes, str]:
    try:
        return synthesize_audio_openrouter(text, voice=voice)
    except Exception as exc:  # noqa: BLE001
        logger.warning("OpenRouter TTS failed: {}. Falling back to edge-tts.", exc)
        return synthesize_audio_with_edge_tts(text, voice=voice)


def save_response_audio(response: dict[str, Any], output_path_without_suffix: Path) -> bytes | None:
    audio_b64 = response.get("audio")
    if not isinstance(audio_b64, str) or not audio_b64:
        return None
    audio_bytes = base64.b64decode(audio_b64)
    path = output_path_without_suffix.with_suffix(detect_audio_extension(audio_bytes))
    path.write_bytes(audio_bytes)
    return audio_bytes


def transcribe_audio_text(audio_bytes: bytes) -> str:
    client = build_openrouter_client(STT_MODEL, task_name="phonecall-audio")
    completion = client.create_completion(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Przetranskrybuj dokładnie tę polską odpowiedź operatora. Zwróć wyłącznie zwykły tekst.",
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": base64.b64encode(audio_bytes).decode("ascii"),
                            "format": detect_audio_format(audio_bytes),
                        },
                    },
                ],
            }
        ]
    )
    if not completion.content:
        raise OpenRouterError("Audio transcription returned no content.")
    return completion.content.strip()


RESPONSE_ANALYST_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_response_context",
            "description": "Return the operator response text, raw endpoint text, and protocol hints.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_response_analysis",
            "description": "Validate the proposed response analysis payload.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "asks_for_password": {"type": "boolean"},
                    "asks_for_reason": {"type": "boolean"},
                    "route_statuses": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                    },
                    "monitoring_disabled": {"type": "boolean"},
                    "success_signal": {"type": "boolean"},
                    "failure_signal": {"type": "boolean"},
                },
                "required": [
                    "summary",
                    "asks_for_password",
                    "asks_for_reason",
                    "route_statuses",
                    "monitoring_disabled",
                    "success_signal",
                    "failure_signal",
                ],
                "additionalProperties": False,
            },
        },
    },
]


def _normalize_route_statuses(route_statuses: Any) -> dict[str, str]:
    if not isinstance(route_statuses, dict):
        raise OpenRouterError("route_statuses must be an object.")
    normalized = {route: "unknown" for route in ROUTES}
    for route in ROUTES:
        value = route_statuses.get(route, "unknown")
        if not isinstance(value, str) or value not in ROUTE_STATUSES:
            raise OpenRouterError(f"Invalid route status for {route}: {value!r}")
        normalized[route] = value
    return normalized


def parse_response_analysis_payload(
    payload: dict[str, Any],
    *,
    transcription: str,
) -> ResponseAnalysis:
    return ResponseAnalysis(
        transcription=transcription,
        summary=str(payload.get("summary", "")).strip() or transcription,
        asks_for_password=bool(payload.get("asks_for_password")),
        asks_for_reason=bool(payload.get("asks_for_reason")),
        route_statuses=_normalize_route_statuses(payload.get("route_statuses", {})),
        monitoring_disabled=bool(payload.get("monitoring_disabled")),
        success_signal=bool(payload.get("success_signal")),
        failure_signal=bool(payload.get("failure_signal")),
    )


def build_response_analysis_handlers(
    *,
    transcription: str,
    response_text: str,
) -> dict[str, Any]:
    def get_response_context(_: dict[str, Any]) -> dict[str, Any]:
        return {
            "transcription": transcription,
            "response_text": response_text,
            "routes": list(ROUTES),
            "protocol_hints": [
                {
                    "match_response_text": "road status delivered",
                    "treat_as_transcription": ROAD_STATUS_FALLBACK_TRANSCRIPT,
                }
            ],
        }

    def validate_response_analysis(arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            parse_response_analysis_payload(arguments, transcription=transcription)
        except OpenRouterError as exc:
            return {"is_valid": False, "message": str(exc)}
        return {"is_valid": True}

    return {
        "get_response_context": get_response_context,
        "validate_response_analysis": validate_response_analysis,
    }


def analyze_operator_response_with_openrouter(
    *,
    transcription: str,
    response_text: str,
) -> ResponseAnalysis:
    client = build_openrouter_client(PHONECALL_ANALYST_MODEL, task_name="phonecall-analysis")
    handlers = build_response_analysis_handlers(
        transcription=transcription,
        response_text=response_text,
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": RESPONSE_ANALYSIS_SYSTEM_PROMPT},
        {"role": "user", "content": "Analyze the latest operator response and return JSON only."},
    ]
    completion = run_tool_conversation(
        client,
        messages=messages,
        tools=RESPONSE_ANALYST_TOOLS,
        handlers=handlers,
        max_steps=RESPONSE_ANALYST_MAX_STEPS,
        error_prefix="Unknown phonecall analysis tool",
    )
    payload = parse_json_object_content(completion.content or "")
    return parse_response_analysis_payload(payload, transcription=transcription)


def normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


def classify_operator_text(transcription: str) -> ResponseAnalysis:
    normalized = normalize_text(transcription)
    analysis = ResponseAnalysis(
        transcription=transcription,
        summary=transcription,
        asks_for_password="hasło" in normalized,
        asks_for_reason=any(pattern in normalized for pattern in ASKS_MATTER_PATTERNS)
        or "dlaczego" in normalized,
        route_statuses={route: "unknown" for route in ROUTES},
        monitoring_disabled="monitoring" in normalized and any(
            phrase in normalized
            for phrase in ("wylacz", "wyłącz", "odlacz", "odłącz", "zrobione", "gotowe")
        ),
        success_signal=bool(FLAG_PATTERN.search(transcription)),
        failure_signal=any(
            phrase in normalized for phrase in ("odmowa", "nie moge", "nie mogę", "koniec rozmowy")
        ),
    )

    last_explicit_status: str | None = None
    sentences = [chunk.strip() for chunk in re.split(r"[.!?]", normalized) if chunk.strip()]
    route_patterns = {
        route: re.compile(rf"rd\s*{re.escape(route[2:].casefold())}") for route in ROUTES
    }

    for sentence in sentences:
        sentence_routes = [
            route for route, route_pattern in route_patterns.items() if route_pattern.search(sentence)
        ]
        if not sentence_routes:
            continue

        if any(word in sentence for word in ("nieprzejezd", "zamkn", "zablok", "nieczyn")):
            status = "nieprzejezdna"
            last_explicit_status = status
        elif any(
            word in sentence
            for word in ("przejezd", "drozna", "drożna", "otwart", "wolna", "jedyne", "jechac", "jechać")
        ):
            status = "przejezdna"
            last_explicit_status = status
        elif "podobnie" in sentence and last_explicit_status is not None:
            status = last_explicit_status
        else:
            continue

        for route in sentence_routes:
            analysis.route_statuses[route] = status

    return analysis


def build_response_text_fallback(response_text: str) -> ResponseAnalysis | None:
    normalized = normalize_text(response_text)
    if not normalized:
        return None
    if "road status delivered" in normalized:
        return classify_operator_text(ROAD_STATUS_FALLBACK_TRANSCRIPT)

    analysis = ResponseAnalysis(
        transcription=response_text,
        summary=response_text,
        route_statuses={route: "unknown" for route in ROUTES},
        asks_for_password="password required" in normalized or "haslo" in normalized,
        asks_for_reason=any(pattern in normalized for pattern in ASKS_MATTER_PATTERNS),
        monitoring_disabled="monitoring disabled" in normalized,
        success_signal=bool(FLAG_PATTERN.search(response_text)),
    )
    if (
        analysis.asks_for_password
        or analysis.asks_for_reason
        or analysis.monitoring_disabled
        or analysis.success_signal
        or "identity confirmed" in normalized
        or "phonecall session started" in normalized
    ):
        return analysis
    return None


def analyze_response(
    response: dict[str, Any],
    *,
    output_stub: Path,
) -> ResponseAnalysis:
    audio_bytes = save_response_audio(response, output_stub)
    extra_texts = [
        value.strip()
        for text_field in ("msg", "message")
        for value in [response.get(text_field)]
        if isinstance(value, str) and value.strip()
    ]
    response_text = "\n".join(extra_texts)
    transcription = response_text
    if audio_bytes:
        try:
            transcription = transcribe_audio_text(audio_bytes)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Audio transcription failed: {}. Falling back to response text.",
                exc,
            )
    if not transcription.strip() and not response_text.strip():
        return ResponseAnalysis(route_statuses={route: "unknown" for route in ROUTES})
    return analyze_operator_response_with_openrouter(
        transcription=transcription.strip() or response_text.strip(),
        response_text=response_text,
    )


def routes_from_statuses(route_statuses: dict[str, str], *, target_status: str) -> list[str]:
    return [route for route, status in route_statuses.items() if status == target_status]


def build_monitoring_request(passable_routes: list[str]) -> str:
    if not passable_routes:
        raise RuntimeError("No passable routes available for monitoring request.")
    if len(passable_routes) == 1:
        return (
            "BARBAKAN. Da się wyłączyć monitoring "
            f"na drodze {passable_routes[0]}? Idzie tam transport żywności "
            "do jednej z baz Zygfryda i ta trasa nie może wejść do logów."
        )
    return (
        "BARBAKAN. Da się wyłączyć monitoring "
        f"na drogach {', '.join(passable_routes)}? Idzie tam transport żywności "
        "do jednej z baz Zygfryda i te trasy nie mogą wejść do logów."
    )


def decide_next_message(state: ConversationState, analysis: ResponseAnalysis) -> str | None:
    state.last_transcription = analysis.transcription

    for route, status in analysis.route_statuses.items():
        if status in {"przejezdna", "nieprzejezdna"}:
            state.route_statuses[route] = status

    state.passable_routes = routes_from_statuses(state.route_statuses, target_status="przejezdna")

    if analysis.failure_signal:
        return None

    if not state.password_sent and analysis.asks_for_password:
        state.password_sent = True
        return PASSWORD_MESSAGE

    if not any(status != "unknown" for status in state.route_statuses.values()) and analysis.asks_for_reason:
        return PURPOSE_MESSAGE

    if analysis.asks_for_reason and not state.reason_explained:
        state.reason_explained = True
        return REASON_MESSAGE

    if state.passable_routes and not state.monitoring_requested:
        state.monitoring_requested = True
        return build_monitoring_request(state.passable_routes)

    return None


def build_attempt_dir(attempt: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = RUNS_DIR / f"{timestamp}-attempt{attempt:02d}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def should_stop(response: dict[str, Any], analysis: ResponseAnalysis) -> bool:
    if analysis.success_signal:
        return True
    code = response.get("code")
    if isinstance(code, int) and code < 0:
        return True
    message = response.get("message")
    return isinstance(message, str) and bool(FLAG_PATTERN.search(message))


def run_attempt(*, attempt: int, max_steps: int, voice: str) -> tuple[dict[str, Any], ConversationState]:
    attempt_dir = build_attempt_dir(attempt)
    state = ConversationState(attempt_dir=str(attempt_dir))

    start_response = request_task({"action": "start"})
    write_json(attempt_dir / "step_00_start.json", start_response)

    next_message = INITIAL_MESSAGE
    latest_response = start_response

    for step in range(1, max_steps + 1):
        request_audio_path = attempt_dir / f"step_{step:02d}_request.wav"
        request_audio_bytes, spoken_transcript = synthesize_audio(next_message, voice=voice)
        request_audio_path.write_bytes(request_audio_bytes)
        request_payload = {
            "text": next_message,
            "spoken_transcript": spoken_transcript,
            "audio_file": str(request_audio_path),
        }
        write_json(attempt_dir / f"step_{step:02d}_request.json", request_payload)

        latest_response = request_task(
            {"audio": base64.b64encode(request_audio_bytes).decode("ascii")}
        )
        write_json(attempt_dir / f"step_{step:02d}_response.json", latest_response)

        analysis = analyze_response(
            latest_response,
            output_stub=attempt_dir / f"step_{step:02d}_operator_audio",
        )
        write_json(attempt_dir / f"step_{step:02d}_analysis.json", asdict(analysis))
        write_json(attempt_dir / f"step_{step:02d}_state.json", asdict(state))

        logger.info(
            "Attempt {} step {}: {}",
            attempt,
            step,
            analysis.summary or analysis.transcription or latest_response.get("message", "no summary"),
        )

        if should_stop(latest_response, analysis):
            return latest_response, state

        next_message = decide_next_message(state, analysis)
        write_json(
            attempt_dir / f"step_{step:02d}_decision.json",
            {"next_message": next_message, "state": asdict(state)},
        )
        if not next_message:
            return latest_response, state

    return latest_response, state


def main() -> int:
    args = parse_args()
    configure_logging(verbose=args.verbose, name="phonecall")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    last_response: dict[str, Any] = {}
    last_state = ConversationState()
    for attempt in range(1, args.max_attempts + 1):
        logger.info("Starting attempt {}", attempt)
        try:
            last_response, last_state = run_attempt(
                attempt=attempt,
                max_steps=args.max_steps,
                voice=args.voice,
            )
        except Exception as exc:  # pragma: no cover - live-task safety net
            logger.error("Attempt {} failed: {}", attempt, exc)
            continue

        write_json(FINAL_RESPONSE_PATH, last_response)
        write_json(FINAL_STATE_PATH, asdict(last_state))

        message = str(last_response.get("message") or "")
        if FLAG_PATTERN.search(message):
            logger.success("Received final flag: {}", message)
            return 0

    logger.error("Phonecall task did not finish successfully.")
    write_json(FINAL_RESPONSE_PATH, last_response)
    write_json(FINAL_STATE_PATH, asdict(last_state))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
