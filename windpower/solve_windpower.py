"""Solve the AG3NTS windpower task by pipelining queued API calls."""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT_HINT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_HINT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_HINT))

from devs_utilities.ag3nts import AG3NTS_VERIFY_URL, submit_task_answer
from devs_utilities.bootstrap import bootstrap_repo
from devs_utilities.files import write_json
from devs_utilities.flags import extract_flag
from devs_utilities.http import HttpRequestError
from devs_utilities.logging import configure_logging, logger as shared_logger
from devs_utilities.repo_env import get_env, get_int_env


REPO_ROOT = bootstrap_repo(__file__)
logger = shared_logger.bind(component="windpower")

TASK_NAME = "windpower"
DEFAULT_TIMEOUT_SECONDS = get_int_env("AG3NTS_TIMEOUT_SECONDS", 30) or 30
RESULT_POLL_INTERVAL_SECONDS = 0.35
FINALIZATION_BUFFER_SECONDS = 1.0
OUTPUT_DIR = Path(__file__).resolve().parent
LAST_SESSION_PATH = OUTPUT_DIR / "last_session.json"
LAST_VERIFY_RESPONSE_PATH = OUTPUT_DIR / "last_verify_response.json"


class WindpowerError(RuntimeError):
    """Raised when the remote windpower workflow returns an invalid state."""


@dataclass(frozen=True, slots=True)
class AppConfig:
    api_key: str
    verify_url: str
    timeout_seconds: int


@dataclass(frozen=True, slots=True)
class WeatherPoint:
    timestamp: str
    wind_ms: float
    precipitation_mm: float
    temperature_c: float

    @property
    def date(self) -> str:
        return self.timestamp.split(" ", 1)[0]

    @property
    def hour(self) -> str:
        return self.timestamp.split(" ", 1)[1]


@dataclass(frozen=True, slots=True)
class ConfigPoint:
    timestamp: str
    wind_ms: float
    pitch_angle: int
    turbine_mode: str

    @property
    def date(self) -> str:
        return self.timestamp.split(" ", 1)[0]

    @property
    def hour(self) -> str:
        return self.timestamp.split(" ", 1)[1]


@dataclass(frozen=True, slots=True)
class TurbineDocumentation:
    rated_power_kw: float
    min_operational_wind_ms: float
    cutoff_wind_ms: float
    wind_yield_points: tuple[tuple[float, float], ...]
    pitch_yield_factors: dict[int, float]

    def estimate_wind_yield_percent(self, wind_ms: float) -> float:
        if wind_ms < self.min_operational_wind_ms or wind_ms > self.cutoff_wind_ms:
            return 0.0

        points = self.wind_yield_points
        if not points:
            raise WindpowerError("Missing wind yield points in documentation.")
        if wind_ms <= points[0][0]:
            return points[0][1]

        for index in range(1, len(points)):
            lower_wind, lower_yield = points[index - 1]
            upper_wind, upper_yield = points[index]
            if wind_ms <= upper_wind:
                spread = upper_wind - lower_wind
                if spread <= 0:
                    return upper_yield
                ratio = (wind_ms - lower_wind) / spread
                return lower_yield + (upper_yield - lower_yield) * ratio

        return points[-1][1]

    def estimate_power_kw(self, wind_ms: float, pitch_angle: int) -> float:
        pitch_factor = self.pitch_yield_factors.get(pitch_angle)
        if pitch_factor is None:
            raise WindpowerError(f"Unsupported pitch angle: {pitch_angle}")
        wind_yield = self.estimate_wind_yield_percent(wind_ms) / 100.0
        return self.rated_power_kw * wind_yield * pitch_factor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def build_config() -> AppConfig:
    api_key = (get_env("AG3NTS_API_KEY") or get_env("COURSE_API_KEY")).strip()
    if not api_key:
        raise SystemExit("Missing AG3NTS_API_KEY/COURSE_API_KEY in the repository .env file.")

    return AppConfig(
        api_key=api_key,
        verify_url=AG3NTS_VERIFY_URL,
        timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
    )


def parse_percent_midpoint(raw_value: Any) -> float:
    numbers = [float(token) for token in re.findall(r"\d+(?:\.\d+)?", str(raw_value))]
    if not numbers:
        raise WindpowerError(f"Cannot parse percentage value: {raw_value!r}")
    if len(numbers) == 1:
        return numbers[0]
    return sum(numbers) / len(numbers)


def parse_required_power_kw(raw_value: Any) -> float:
    numbers = [float(token) for token in re.findall(r"\d+(?:\.\d+)?", str(raw_value))]
    if not numbers:
        raise WindpowerError(f"Cannot parse power deficit from: {raw_value!r}")
    return max(numbers)


def parse_documentation(payload: dict[str, Any]) -> TurbineDocumentation:
    point_map: dict[float, float] = {}
    for item in payload.get("windPowerYieldPercent", []):
        if "windMs" in item:
            point_map[float(item["windMs"])] = parse_percent_midpoint(item["yieldPercent"])
            continue
        wind_range = str(item.get("windMsRange", "")).strip()
        bounds = [float(token) for token in re.findall(r"\d+(?:\.\d+)?", wind_range)]
        if len(bounds) == 2:
            midpoint = parse_percent_midpoint(item["yieldPercent"])
            point_map[bounds[0]] = midpoint
            point_map[bounds[1]] = midpoint

    pitch_factors: dict[int, float] = {}
    for item in payload.get("pitchAngleYieldPercent", []):
        pitch_factors[int(float(item["pitchAngleDeg"]))] = (
            parse_percent_midpoint(item["yieldPercent"]) / 100.0
        )

    safety = payload.get("safety") or {}
    return TurbineDocumentation(
        rated_power_kw=float(payload["ratedPowerKw"]),
        min_operational_wind_ms=float(safety["minOperationalWindMs"]),
        cutoff_wind_ms=float(safety["cutoffWindMs"]),
        wind_yield_points=tuple(sorted(point_map.items())),
        pitch_yield_factors=pitch_factors,
    )


def parse_weather_points(payload: dict[str, Any]) -> list[WeatherPoint]:
    forecast = payload.get("forecast")
    if not isinstance(forecast, list):
        raise WindpowerError("Weather payload does not contain a forecast list.")

    points: list[WeatherPoint] = []
    for item in forecast:
        points.append(
            WeatherPoint(
                timestamp=str(item["timestamp"]),
                wind_ms=float(item["windMs"]),
                precipitation_mm=float(item["precipitationMm"]),
                temperature_c=float(item["temperatureC"]),
            )
        )
    return points


def choose_production_point(
    weather_points: list[WeatherPoint],
    documentation: TurbineDocumentation,
    required_power_kw: float,
) -> ConfigPoint:
    safe_points = [
        point
        for point in weather_points
        if documentation.min_operational_wind_ms <= point.wind_ms <= documentation.cutoff_wind_ms
    ]
    if not safe_points:
        raise WindpowerError("No forecast point can safely produce electricity.")

    for point in safe_points:
        estimated_power = documentation.estimate_power_kw(point.wind_ms, pitch_angle=0)
        if estimated_power >= required_power_kw:
            return ConfigPoint(
                timestamp=point.timestamp,
                wind_ms=point.wind_ms,
                pitch_angle=0,
                turbine_mode="production",
            )

    fallback = max(
        safe_points,
        key=lambda point: documentation.estimate_power_kw(point.wind_ms, pitch_angle=0),
    )
    logger.warning(
        "No safe point reached the requested {:.2f} kW in the local estimate. "
        "Trying the strongest safe slot at {} ({:.2f} m/s).",
        required_power_kw,
        fallback.timestamp,
        fallback.wind_ms,
    )
    return ConfigPoint(
        timestamp=fallback.timestamp,
        wind_ms=fallback.wind_ms,
        pitch_angle=0,
        turbine_mode="production",
    )


def build_config_plan(
    weather_points: list[WeatherPoint],
    documentation: TurbineDocumentation,
    required_power_kw: float,
) -> list[ConfigPoint]:
    storm_points = [
        ConfigPoint(
            timestamp=point.timestamp,
            wind_ms=point.wind_ms,
            pitch_angle=90,
            turbine_mode="idle",
        )
        for point in weather_points
        if point.wind_ms > documentation.cutoff_wind_ms
    ]
    production_point = choose_production_point(
        weather_points,
        documentation,
        required_power_kw,
    )

    point_map: dict[str, ConfigPoint] = {point.timestamp: point for point in storm_points}
    point_map[production_point.timestamp] = production_point
    return [point_map[timestamp] for timestamp in sorted(point_map)]


def config_points_to_payload(
    config_points: list[ConfigPoint],
    unlock_codes: dict[str, str],
) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for point in config_points:
        unlock_code = unlock_codes.get(point.timestamp)
        if not unlock_code:
            raise WindpowerError(f"Missing unlockCode for {point.timestamp}.")
        payload[point.timestamp] = {
            "pitchAngle": point.pitch_angle,
            "turbineMode": point.turbine_mode,
            "unlockCode": unlock_code,
        }
    return payload


class WindpowerClient:
    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def call(self, answer: dict[str, Any]) -> Any:
        try:
            return submit_task_answer(
                self._config.verify_url,
                api_key=self._config.api_key,
                task=TASK_NAME,
                answer=answer,
                timeout_seconds=self._config.timeout_seconds,
            )
        except HttpRequestError as exc:
            body = exc.body_as_json()
            if body is not None:
                return body
            raise WindpowerError(str(exc)) from exc


def is_negative_response(payload: Any) -> bool:
    return isinstance(payload, dict) and int(payload.get("code", 0)) < 0


def require_success(payload: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise WindpowerError(f"{context} returned a non-dict response: {payload!r}")
    if is_negative_response(payload):
        raise WindpowerError(f"{context} failed: {payload}")
    return payload


def wait_for_results(
    client: WindpowerClient,
    *,
    deadline: float,
    documentation: TurbineDocumentation,
    session_log: dict[str, Any],
) -> tuple[list[ConfigPoint], dict[str, str]]:
    weather_result: dict[str, Any] | None = None
    powerplant_result: dict[str, Any] | None = None
    turbinecheck_result: dict[str, Any] | None = None
    config_points: list[ConfigPoint] | None = None
    unlock_codes: dict[str, str] = {}
    unlock_requests_sent = False

    while time.monotonic() < deadline:
        result = require_success(client.call({"action": "getResult"}), context="getResult")
        source = str(result.get("sourceFunction", "")).strip()

        if source == "weather":
            weather_result = result
            session_log["weather"] = result
        elif source == "powerplantcheck":
            powerplant_result = result
            session_log["powerplantcheck"] = result
        elif source == "turbinecheck":
            turbinecheck_result = result
            session_log["turbinecheck"] = result
        elif source == "unlockCodeGenerator":
            signed = result.get("signedParams") or {}
            timestamp = f"{signed.get('startDate')} {signed.get('startHour')}"
            unlock_codes[timestamp] = str(result["unlockCode"])
        elif int(result.get("code", 0)) != 11:
            logger.debug("Ignoring unexpected queued response: {}", result)

        if (
            not unlock_requests_sent
            and weather_result is not None
            and powerplant_result is not None
        ):
            weather_points = parse_weather_points(weather_result)
            required_power_kw = parse_required_power_kw(powerplant_result["powerDeficitKw"])
            config_points = build_config_plan(weather_points, documentation, required_power_kw)
            session_log["plan"] = {
                "requiredPowerKw": required_power_kw,
                "configPoints": [
                    {
                        "timestamp": point.timestamp,
                        "windMs": point.wind_ms,
                        "pitchAngle": point.pitch_angle,
                        "turbineMode": point.turbine_mode,
                        "estimatedPowerKw": round(
                            documentation.estimate_power_kw(point.wind_ms, point.pitch_angle),
                            4,
                        ),
                    }
                    for point in config_points
                ],
            }
            for point in config_points:
                require_success(
                    client.call(
                        {
                            "action": "unlockCodeGenerator",
                            "startDate": point.date,
                            "startHour": point.hour,
                            "windMs": point.wind_ms,
                            "pitchAngle": point.pitch_angle,
                        }
                    ),
                    context=f"unlockCodeGenerator {point.timestamp}",
                )
            unlock_requests_sent = True

        if (
            config_points is not None
            and turbinecheck_result is not None
            and len(unlock_codes) == len(config_points)
        ):
            return config_points, unlock_codes

        time.sleep(RESULT_POLL_INTERVAL_SECONDS)

    raise WindpowerError("Timed out waiting for queued windpower results.")


def solve_task(config: AppConfig) -> tuple[str, dict[str, Any]]:
    client = WindpowerClient(config)
    session_log: dict[str, Any] = {}

    start_response = require_success(client.call({"action": "start"}), context="start")
    session_log["start"] = start_response
    session_timeout = float(start_response.get("sessionTimeout", 40))
    deadline = time.monotonic() + session_timeout - FINALIZATION_BUFFER_SECONDS

    documentation_response = require_success(
        client.call({"action": "get", "param": "documentation"}),
        context="get documentation",
    )
    session_log["documentation"] = documentation_response
    documentation = parse_documentation(documentation_response)

    for param in ("weather", "powerplantcheck", "turbinecheck"):
        require_success(client.call({"action": "get", "param": param}), context=f"queue {param}")

    config_points, unlock_codes = wait_for_results(
        client,
        deadline=deadline,
        documentation=documentation,
        session_log=session_log,
    )

    config_payload = config_points_to_payload(config_points, unlock_codes)
    config_response = require_success(
        client.call({"action": "config", "configs": config_payload}),
        context="config",
    )
    session_log["config"] = {
        "request": config_payload,
        "response": config_response,
    }

    final_response = require_success(client.call({"action": "done"}), context="done")
    session_log["done"] = final_response
    write_json(LAST_VERIFY_RESPONSE_PATH, final_response)
    write_json(LAST_SESSION_PATH, session_log)

    flag = extract_flag(final_response)
    if not flag:
        raise WindpowerError(f"Final response does not contain a flag: {final_response}")
    return flag, session_log


def main() -> int:
    args = parse_args()
    configure_logging(name="windpower", verbose=args.verbose)
    config = build_config()

    try:
        started_at = time.monotonic()
        flag, session_log = solve_task(config)
        elapsed = time.monotonic() - started_at
        stored_points = session_log.get("config", {}).get("response", {}).get("storedPoints")
        logger.success(
            "Flag: {} (storedPoints={}, elapsed={:.2f}s)",
            flag,
            stored_points,
            elapsed,
        )
    except WindpowerError as exc:
        logger.error("Windpower solve failed: {}", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
