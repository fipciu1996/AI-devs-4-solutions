from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NEGOTIATIONS_DIR = REPO_ROOT / "negotiations"
NEGOTIATIONS_SRC_DIR = NEGOTIATIONS_DIR / "src"
for candidate in (
    str(REPO_ROOT),
    str(NEGOTIATIONS_DIR),
    str(NEGOTIATIONS_SRC_DIR),
):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from submit_tools import (
    build_answer,
    build_ngrok_forward_kwargs,
    extract_listener_url,
    is_pending_check_response,
)


class _ListenerWithMethod:
    def url(self) -> str:
        return "https://example.ngrok.app"


class _ListenerWithAttribute:
    url = "https://example-attr.ngrok.app"


class SubmitToolsTests(unittest.TestCase):
    def test_build_answer_uses_custom_tool_path(self) -> None:
        payload = build_answer("https://public.example", tool_path="/api/custom-tool")

        tool = payload["tools"][0]
        self.assertEqual(tool["URL"], "https://public.example/api/custom-tool")

    def test_build_ngrok_forward_kwargs_omits_empty_values(self) -> None:
        kwargs = build_ngrok_forward_kwargs(
            18081,
            authtoken=None,
            domain=None,
        )

        self.assertEqual(kwargs, {"addr": 18081})

    def test_extract_listener_url_supports_method_and_attribute(self) -> None:
        self.assertEqual(
            extract_listener_url(_ListenerWithMethod()),
            "https://example.ngrok.app",
        )
        self.assertEqual(
            extract_listener_url(_ListenerWithAttribute()),
            "https://example-attr.ngrok.app",
        )

    def test_is_pending_check_response_detects_pending_markers(self) -> None:
        self.assertTrue(
            is_pending_check_response({"message": "Processing, check again later."})
        )
        self.assertFalse(
            is_pending_check_response({"message": "FLG:SUCCESS"})
        )


if __name__ == "__main__":
    unittest.main()
