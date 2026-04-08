from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

from negotiations import submit_tools


class NegotiationsSubmitToolsArgTests(unittest.TestCase):
    def test_parse_args_reads_ngrok_auth_token_from_new_env_name(self) -> None:
        with patch.dict("os.environ", {"NGROK_AUTH_TOKEN": "new-token"}, clear=True):
            with patch.object(sys, "argv", ["submit_tools.py", "--use-ngrok"]):
                args = submit_tools.parse_args()

        self.assertEqual(args.ngrok_auth_token, "new-token")

    def test_parse_args_falls_back_to_legacy_ngrok_env_name(self) -> None:
        with patch.dict("os.environ", {"NGROK_AUTHTOKEN": "legacy-token"}, clear=True):
            with patch.object(sys, "argv", ["submit_tools.py", "--use-ngrok"]):
                args = submit_tools.parse_args()

        self.assertEqual(args.ngrok_auth_token, "legacy-token")

    def test_build_ngrok_forward_kwargs_includes_application_name_metadata(self) -> None:
        kwargs = submit_tools.build_ngrok_forward_kwargs(
            8080,
            authtoken="token",
            domain="example.ngrok.app",
        )

        self.assertEqual(kwargs["metadata"], "AI Devs 4 - negotiations")
        self.assertEqual(kwargs["session_metadata"], "AI Devs 4 - negotiations")
        self.assertEqual(kwargs["authtoken"], "token")
        self.assertEqual(kwargs["domain"], "example.ngrok.app")


if __name__ == "__main__":
    unittest.main()
