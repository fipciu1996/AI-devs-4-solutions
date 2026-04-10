"""Runtime configuration for the negotiations API server."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from devs_utilities.ag3nts import build_ag3nts_public_data_url


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    api_host: str
    api_port: int
    api_tool_path: str
    data_base_url: str
    data_cache_dir: Path
    request_timeout_seconds: float
    match_limit: int


def load_settings() -> Settings:
    """Load application settings from the environment."""

    default_data_base_url = build_ag3nts_public_data_url("s03e04_csv")
    data_base_url = (
        os.getenv("DATA_BASE_URL", "").strip()
        or os.getenv("NEGOTIATIONS_DATA_BASE_URL", "").strip()
        or default_data_base_url
    )
    api_tool_path = os.getenv("API_TOOL_PATH", "/api/find-cities").strip() or "/api/find-cities"
    if not api_tool_path.startswith("/"):
        api_tool_path = f"/{api_tool_path}"
    return Settings(
        api_host=os.getenv("API_HOST", "0.0.0.0").strip() or "0.0.0.0",
        api_port=int(os.getenv("API_PORT", "18081")),
        api_tool_path=api_tool_path.rstrip("/") or "/api/find-cities",
        data_base_url=data_base_url.rstrip("/"),
        data_cache_dir=Path(os.getenv("DATA_CACHE_DIR", "data_cache")),
        request_timeout_seconds=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "30")),
        match_limit=max(1, int(os.getenv("MATCH_LIMIT", "5"))),
    )
