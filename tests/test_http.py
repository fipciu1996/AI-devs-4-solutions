from __future__ import annotations

import io
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from devs_utilities.http import HttpRequestError, RAW_TEXT, get_json, post_json


TEST_BASE_URL = "".join(("https", "://", "example", ".com"))


class FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class HttpUtilitiesTests(unittest.TestCase):
    @patch("devs_utilities.http.request.urlopen")
    def test_get_json_decodes_json(self, mock_urlopen) -> None:
        mock_urlopen.return_value = FakeResponse(json.dumps({"ok": True}).encode("utf-8"))

        result = get_json(f"{TEST_BASE_URL}/data")

        self.assertEqual(result, {"ok": True})

    @patch("devs_utilities.http.request.urlopen")
    def test_get_json_returns_raw_text_when_requested(self, mock_urlopen) -> None:
        mock_urlopen.return_value = FakeResponse(b"plain text body")

        result = get_json(f"{TEST_BASE_URL}/text", on_decode_error=RAW_TEXT)

        self.assertEqual(result, "plain text body")

    @patch("devs_utilities.http.request.urlopen")
    def test_post_json_raises_http_request_error_for_http_error(self, mock_urlopen) -> None:
        mock_urlopen.side_effect = HTTPError(
            url=f"{TEST_BASE_URL}/fail",
            code=503,
            msg="Service Unavailable",
            hdrs={},
            fp=io.BytesIO(b'{"message":"down"}'),
        )

        with self.assertRaises(HttpRequestError) as context:
            post_json(f"{TEST_BASE_URL}/fail", {"hello": "world"})

        self.assertEqual(context.exception.status_code, 503)
        self.assertEqual(context.exception.body_as_json(), {"message": "down"})

    @patch("devs_utilities.http.request.urlopen")
    def test_get_json_raises_http_request_error_for_url_error(self, mock_urlopen) -> None:
        mock_urlopen.side_effect = URLError("no route to host")

        with self.assertRaises(HttpRequestError) as context:
            get_json(f"{TEST_BASE_URL}/data")

        self.assertIn("no route to host", str(context.exception))

    @patch("devs_utilities.http.request.urlopen")
    def test_get_json_raises_http_request_error_for_timeout(self, mock_urlopen) -> None:
        mock_urlopen.side_effect = TimeoutError()

        with self.assertRaises(HttpRequestError) as context:
            get_json(f"{TEST_BASE_URL}/slow")

        self.assertIn("timed out", str(context.exception))


if __name__ == "__main__":
    unittest.main()
