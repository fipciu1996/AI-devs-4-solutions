from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FAILURE_DIR = REPO_ROOT / "failure"
for candidate in (str(REPO_ROOT), str(FAILURE_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from devs_utilities.openrouter import OpenRouterError
from solve_failure import validate_model_log_lines


class FailureModelLogTests(unittest.TestCase):
    def test_validate_model_log_lines_accepts_prefixed_lines(self) -> None:
        lines = validate_model_log_lines(
            [
                "[2026-04-01 10:15] [ERROR] ECCS8 cooling reserve falling; shutdown path armed.",
                "[2026-04-01 10:16] [WARN] WTRPMP cavitation detected; pressure unstable.",
            ]
        )

        self.assertEqual(len(lines), 2)

    def test_validate_model_log_lines_rejects_missing_prefix(self) -> None:
        with self.assertRaises(OpenRouterError):
            validate_model_log_lines(["ECCS8 cooling reserve falling"])


if __name__ == "__main__":
    unittest.main()
