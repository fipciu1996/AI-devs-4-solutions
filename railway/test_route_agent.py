from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
RAILWAY_DIR = Path(__file__).resolve().parent
for candidate in (str(REPO_ROOT), str(RAILWAY_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from devs_utilities.http import HttpRequestError
from route_agent import RailwayApiClient, RailwayError, require_route, require_status


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
    @patch("route_agent.post_json")
    def test_call_builds_expected_payload(self, mock_post_json) -> None:
        mock_post_json.return_value = {"ok": True, "status": "RTOPEN"}
        client = RailwayApiClient("https://***MASKED***/api", "secret")

        result = client.call(action="setstatus", route="B-4", value="rtopen")

        self.assertEqual(result["status"], "RTOPEN")
        self.assertEqual(
            mock_post_json.call_args.args,
            (
                "https://***MASKED***/api",
                {
                    "apikey": "secret",
                    "task": "railway",
                    "answer": {
                        "action": "setstatus",
                        "route": "b-4",
                        "value": "RTOPEN",
                    },
                },
            ),
        )
        self.assertEqual(
            mock_post_json.call_args.kwargs,
            {
                "headers": {"Content-Type": "application/json"},
                "timeout_seconds": 60,
            },
        )

    @patch("route_agent.post_json")
    def test_call_raises_for_api_error_response(self, mock_post_json) -> None:
        mock_post_json.return_value = {"ok": False, "error": "boom"}
        client = RailwayApiClient("https://***MASKED***/api", "secret")

        with self.assertRaises(RailwayError):
            client.call(action="help")

    @patch("route_agent.time.sleep")
    @patch("route_agent.post_json")
    def test_call_retries_after_rate_limit(self, mock_post_json, mock_sleep) -> None:
        mock_post_json.side_effect = [
            HttpRequestError(
                url="https://***MASKED***/api",
                message="HTTP 429",
                status_code=429,
                body=json.dumps(
                    {
                        "code": -985,
                        "message": "API rate limit exceeded.",
                        "retry_after": 16,
                        "penalty_seconds": 5,
                    }
                ),
            ),
            {"ok": True, "status": "RTOPEN"},
        ]
        client = RailwayApiClient("https://***MASKED***/api", "secret")

        result = client.call(action="getstatus", route="a-1")

        self.assertEqual(result["status"], "RTOPEN")
        mock_sleep.assert_called_once_with(16)

    @patch("route_agent.time.sleep")
    @patch("route_agent.post_json")
    def test_call_retries_after_service_unavailable(self, mock_post_json, mock_sleep) -> None:
        mock_post_json.side_effect = [
            HttpRequestError(
                url="https://***MASKED***/api",
                message="HTTP 503",
                status_code=503,
                body=json.dumps(
                    {
                        "message": "Service temporarily unavailable.",
                        "retry_after": 7,
                    }
                ),
            ),
            {"ok": True, "status": "RTCLOSE"},
        ]
        client = RailwayApiClient("https://***MASKED***/api", "secret")

        result = client.call(action="getstatus", route="a-1")

        self.assertEqual(result["status"], "RTCLOSE")
        mock_sleep.assert_called_once_with(7)

    @patch("route_agent.time.sleep")
    @patch("route_agent.post_json")
    def test_call_retries_after_service_unavailable_without_retry_after(
        self,
        mock_post_json,
        mock_sleep,
    ) -> None:
        mock_post_json.side_effect = [
            HttpRequestError(
                url="https://***MASKED***/api",
                message="HTTP 503",
                status_code=503,
                body=json.dumps(
                    {
                        "code": -925,
                        "message": "Temporary server outage. Please retry in a moment.",
                    }
                ),
            ),
            {"ok": True, "status": "RTOPEN"},
        ]
        client = RailwayApiClient("https://***MASKED***/api", "secret")

        result = client.call(action="getstatus", route="a-1")

        self.assertEqual(result["status"], "RTOPEN")
        mock_sleep.assert_called_once_with(30)


if __name__ == "__main__":
    unittest.main()
