from __future__ import annotations

import sys
import unittest
from pathlib import Path

ELECTRICITY_DIR = Path(__file__).resolve().parent
REPO_ROOT = ELECTRICITY_DIR.parent
for candidate in (str(REPO_ROOT), str(ELECTRICITY_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from devs_utilities.openrouter import OpenRouterError
from solve_electricity import normalize_rotation_token, validate_rotation_sequence


class ElectricityModelPlanTests(unittest.TestCase):
    def test_normalize_rotation_token_returns_canonical_form(self) -> None:
        self.assertEqual(normalize_rotation_token("2x3"), "2x3")

    def test_validate_rotation_sequence_rejects_out_of_bounds_move(self) -> None:
        with self.assertRaises(OpenRouterError):
            validate_rotation_sequence(["4x1"])


if __name__ == "__main__":
    unittest.main()
