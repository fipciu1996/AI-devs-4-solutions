"""Packages API client utilities used by the proxy API server."""

from __future__ import annotations

from typing import Any

from devs_utilities.http import HttpRequestError, JsonResponseError, post_json

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
        try:
            parsed = post_json(
                self._settings.packages_api_url,
                payload,
                timeout_seconds=self._settings.packages_timeout_seconds,
            )
        except (HttpRequestError, JsonResponseError) as exc:
            raise PackagesApiError(str(exc)) from exc

        if not isinstance(parsed, dict):
            raise PackagesApiError("Packages API returned a non-object JSON response.")
        return parsed
