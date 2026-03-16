from __future__ import annotations

import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parent))

from route_agent import RailwayApiClient, RailwayError, require_route, require_status


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class ValidationTests(unittest.TestCase):
    def test_require_route_normalizes_case(self) -> None:
        self.assertEqual(require_route("A-12"), "a-12")

    def test_require_route_rejects_invalid_value(self) -> None:
        with self.assertRaises(RailwayError):
            require_route("12-a")

    def test_require_status_normalizes_case(self) -> None:
        self.assertEqual(require_status("rtclose"), "RTCLOSE")

    def test_require_status_rejects_invalid_value(self) -> None:
        with self.assertRaises(RailwayError):
            require_status("paused")


class RailwayClientTests(unittest.TestCase):
    @patch("route_agent.request.urlopen")
    def test_call_builds_expected_payload(self, mock_urlopen) -> None:
        mock_urlopen.return_value = FakeResponse({"ok": True, "status": "RTOPEN"})
        client = RailwayApiClient("https://***MASKED***/api", "secret")

        result = client.call(action="setstatus", route="B-4", value="rtopen")

        self.assertEqual(result["status"], "RTOPEN")
        http_request = mock_urlopen.call_args.args[0]
        sent_payload = json.loads(http_request.data.decode("utf-8"))
        self.assertEqual(
            sent_payload,
            {
                "apikey": "secret",
                "task": "railway",
                "answer": {
                    "action": "setstatus",
                    "route": "b-4",
                    "value": "RTOPEN",
                },
            },
        )

    @patch("route_agent.request.urlopen")
    def test_call_raises_for_api_error_response(self, mock_urlopen) -> None:
        mock_urlopen.return_value = FakeResponse({"ok": False, "error": "boom"})
        client = RailwayApiClient("https://***MASKED***/api", "secret")

        with self.assertRaises(RailwayError):
            client.call(action="help")

    @patch("route_agent.time.sleep")
    @patch("route_agent.time.monotonic")
    @patch("route_agent.request.urlopen")
    def test_call_applies_polite_delay_between_requests(
        self,
        mock_urlopen,
        mock_monotonic,
        mock_sleep,
    ) -> None:
        mock_urlopen.side_effect = [
            FakeResponse({"ok": True, "status": "RTOPEN"}),
            FakeResponse({"ok": True, "status": "RTCLOSE"}),
        ]
        mock_monotonic.side_effect = [10.0, 10.05]
        client = RailwayApiClient("https://***MASKED***/api", "secret")

        client.call(action="getstatus", route="a-1")
        client.call(action="getstatus", route="a-1")

        mock_sleep.assert_called_once_with(0.05)

    @patch("route_agent.time.sleep")
    @patch("route_agent.request.urlopen")
    def test_call_retries_after_rate_limit(self, mock_urlopen, mock_sleep) -> None:
        rate_limit_payload = {
            "code": -985,
            "message": "API rate limit exceeded.",
            "retry_after": 16,
            "penalty_seconds": 5,
        }
        rate_limit_error = HTTPError(
            url="https://***MASKED***/api",
            code=429,
            msg="Too Many Requests",
            hdrs={},
            fp=io.BytesIO(json.dumps(rate_limit_payload).encode("utf-8")),
        )
        mock_urlopen.side_effect = [
            rate_limit_error,
            FakeResponse({"ok": True, "status": "RTOPEN"}),
        ]
        client = RailwayApiClient("https://***MASKED***/api", "secret")

        result = client.call(action="getstatus", route="a-1")

        self.assertEqual(result["status"], "RTOPEN")
        mock_sleep.assert_called_once_with(16)

    @patch("route_agent.time.sleep")
    @patch("route_agent.request.urlopen")
    def test_call_retries_after_service_unavailable(self, mock_urlopen, mock_sleep) -> None:
        service_unavailable_payload = {
            "message": "Service temporarily unavailable.",
            "retry_after": 7,
        }
        service_unavailable_error = HTTPError(
            url="https://***MASKED***/api",
            code=503,
            msg="Service Unavailable",
            hdrs={},
            fp=io.BytesIO(json.dumps(service_unavailable_payload).encode("utf-8")),
        )
        mock_urlopen.side_effect = [
            service_unavailable_error,
            FakeResponse({"ok": True, "status": "RTCLOSE"}),
        ]
        client = RailwayApiClient("https://***MASKED***/api", "secret")

        result = client.call(action="getstatus", route="a-1")

        self.assertEqual(result["status"], "RTCLOSE")
        mock_sleep.assert_called_once_with(7)

    @patch("route_agent.time.sleep")
    @patch("route_agent.request.urlopen")
    def test_call_retries_after_service_unavailable_without_retry_after(
        self,
        mock_urlopen,
        mock_sleep,
    ) -> None:
        service_unavailable_payload = {
            "code": -925,
            "message": "Temporary server outage. Please retry in a moment.",
        }
        service_unavailable_error = HTTPError(
            url="https://***MASKED***/api",
            code=503,
            msg="Service Unavailable",
            hdrs={},
            fp=io.BytesIO(json.dumps(service_unavailable_payload).encode("utf-8")),
        )
        mock_urlopen.side_effect = [
            service_unavailable_error,
            FakeResponse({"ok": True, "status": "RTOPEN"}),
        ]
        client = RailwayApiClient("https://***MASKED***/api", "secret")

        result = client.call(action="getstatus", route="a-1")

        self.assertEqual(result["status"], "RTOPEN")
        mock_sleep.assert_called_once_with(3)


if __name__ == "__main__":
    unittest.main()
