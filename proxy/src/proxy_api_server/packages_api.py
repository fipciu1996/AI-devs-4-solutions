"""Packages API client utilities used by the proxy API server."""

from __future__ import annotations

import json
from typing import Any
from urllib import error, request

from proxy_api_server.config import Settings


class PackagesApiError(RuntimeError):
    """Raised when the packages API request fails."""


class PackagesApiClient:
    """Thin client for the external packages API."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def check_package(self, packageid: str) -> dict[str, Any]:
        return self._post(
            {
                "apikey": self._settings.packages_api_key,
                "action": "check",
                "packageid": packageid,
            }
        )

    def redirect_package(
        self,
        packageid: str,
        destination: str,
        code: str,
    ) -> dict[str, Any]:
        return self._post(
            {
                "apikey": self._settings.packages_api_key,
                "action": "redirect",
                "packageid": packageid,
                "destination": destination,
                "code": code,
            }
        )

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        req = request.Request(
            self._settings.packages_api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self._settings.packages_timeout_seconds) as response:
                raw_response = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            raise PackagesApiError(
                f"Packages API returned HTTP {exc.code}: {detail or exc.reason}"
            ) from exc
        except error.URLError as exc:
            raise PackagesApiError(f"Packages API request failed: {exc.reason}") from exc
        except TimeoutError as exc:
            raise PackagesApiError("Packages API request timed out.") from exc

        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise PackagesApiError("Packages API returned invalid JSON.") from exc

        if not isinstance(parsed, dict):
            raise PackagesApiError("Packages API returned a non-object JSON response.")
        return parsed
