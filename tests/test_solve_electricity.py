from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
ELECTRICITY_DIR = REPO_ROOT / "electricity"
for candidate in (str(REPO_ROOT), str(ELECTRICITY_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from devs_utilities.openrouter import OpenRouterError
from solve_electricity import execute_rotation_sequence, normalize_rotation_token, validate_rotation_sequence


class ElectricityModelPlanTests(unittest.TestCase):
    def test_normalize_rotation_token_returns_canonical_form(self) -> None:
        self.assertEqual(normalize_rotation_token("2x3"), "2x3")

    def test_validate_rotation_sequence_rejects_out_of_bounds_move(self) -> None:
        with self.assertRaises(OpenRouterError):
            validate_rotation_sequence(["4x1"])

    @patch("solve_electricity.rotate_once")
    @patch("solve_electricity.extract_flag")
    def test_execute_rotation_sequence_returns_first_flag(self, mock_extract_flag, mock_rotate_once) -> None:
        mock_rotate_once.side_effect = [{"code": 1}, {"message": "{FLG:ROTATEIT}"}]
        mock_extract_flag.side_effect = [None, "{FLG:ROTATEIT}"]

        flag = execute_rotation_sequence("apikey", ["1x2", "2x3"])

        self.assertEqual(flag, "{FLG:ROTATEIT}")


if __name__ == "__main__":
    unittest.main()
