"""Shared HTTP and JSON utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping
from urllib import error, request


RawFallback = str
RAW_TEXT = "raw_text"
RAW_DICT = "raw_dict"
RAISE = "raise"


@dataclass(frozen=True, slots=True)
class HttpRequestError(RuntimeError):
    """Raised when an HTTP request fails before a valid response is returned."""

    url: str
    message: str
    status_code: int | None = None
    body: str | None = None
    headers: Mapping[str, str] | None = None

    def __str__(self) -> str:
        return self.message

    def body_as_json(self) -> Any | None:
        if self.body in (None, ""):
            return None
        try:
            return json.loads(self.body)
        except json.JSONDecodeError:
            return None

    def to_response_dict(self) -> dict[str, Any]:
        payload = self.body_as_json()
        if isinstance(payload, dict):
            result = dict(payload)
        elif self.body is not None:
            result = {"raw": self.body}
        else:
            result = {"error": self.message}
        if self.status_code is not None:
            result["http_status"] = self.status_code
        return result


@dataclass(frozen=True, slots=True)
class JsonResponseError(RuntimeError):
    """Raised when a response cannot be decoded as JSON."""

    url: str
    raw_response: str

    def __str__(self) -> str:
        return f"Response from {self.url} is not valid JSON."


def _decode_json(
    url: str,
    raw_response: str,
    *,
    on_decode_error: str,
) -> Any:
    try:
        return json.loads(raw_response)
    except json.JSONDecodeError:
        if on_decode_error == RAW_TEXT:
            return raw_response
        if on_decode_error == RAW_DICT:
            return {"raw": raw_response}
        raise JsonResponseError(url=url, raw_response=raw_response) from None


def request_bytes(
    url: str,
    *,
    method: str = "GET",
    data: bytes | None = None,
    headers: Mapping[str, str] | None = None,
    timeout_seconds: float = 30.0,
) -> bytes:
    """Perform an HTTP request and return response bytes."""

    http_request = request.Request(
        url=url,
        data=data,
        headers=dict(headers or {}),
        method=method,
    )
    try:
        with request.urlopen(http_request, timeout=timeout_seconds) as response:
            return response.read()
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise HttpRequestError(
            url=url,
            message=f"HTTP {exc.code} for {url}: {detail or exc.reason}",
            status_code=exc.code,
            body=detail,
            headers=dict(exc.headers.items()),
        ) from exc
    except error.URLError as exc:
        raise HttpRequestError(
            url=url,
            message=f"Network error for {url}: {exc.reason}",
        ) from exc
    except TimeoutError as exc:
        raise HttpRequestError(
            url=url,
            message=f"Request to {url} timed out.",
        ) from exc


def get_bytes(
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    timeout_seconds: float = 30.0,
) -> bytes:
    """Fetch raw bytes with a GET request."""

    return request_bytes(
        url,
        method="GET",
        headers=headers,
        timeout_seconds=timeout_seconds,
    )


def get_text(
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    timeout_seconds: float = 30.0,
    encoding: str = "utf-8",
    errors: str = "replace",
) -> str:
    """Fetch response text with a GET request."""

    raw_bytes = get_bytes(url, headers=headers, timeout_seconds=timeout_seconds)
    return raw_bytes.decode(encoding, errors=errors)


def get_json(
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    timeout_seconds: float = 30.0,
    on_decode_error: str = RAISE,
) -> Any:
    """Fetch JSON with a GET request."""

    raw_response = get_text(
        url,
        headers=headers,
        timeout_seconds=timeout_seconds,
        errors="replace",
    )
    return _decode_json(
        url,
        raw_response,
        on_decode_error=on_decode_error,
    )


def post_json(
    url: str,
    payload: Any,
    *,
    headers: Mapping[str, str] | None = None,
    timeout_seconds: float = 30.0,
    on_decode_error: str = RAISE,
) -> Any:
    """Send JSON and decode the response."""

    request_headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if headers:
        request_headers.update(dict(headers))
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    raw_bytes = request_bytes(
        url,
        method="POST",
        data=body,
        headers=request_headers,
        timeout_seconds=timeout_seconds,
    )
    raw_response = raw_bytes.decode("utf-8", errors="replace")
    return _decode_json(
        url,
        raw_response,
        on_decode_error=on_decode_error,
    )
