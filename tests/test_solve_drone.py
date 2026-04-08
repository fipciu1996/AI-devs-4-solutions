from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DRONE_DIR = REPO_ROOT / "drone"
for candidate in (str(REPO_ROOT), str(DRONE_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from devs_utilities.openrouter import OpenRouterError
from solve_drone import escape_argparse_help, validate_target_sector


class DroneModelTargetTests(unittest.TestCase):
    def test_escape_argparse_help_escapes_percent_signs(self) -> None:
        self.assertEqual(escape_argparse_help("100%"), "100%%")

    def test_validate_target_sector_accepts_valid_coordinates(self) -> None:
        self.assertEqual(
            validate_target_sector({"sector_x": 2, "sector_y": 4, "reason": "dam at the bottom"}),
            (2, 4, "dam at the bottom"),
        )

    def test_validate_target_sector_rejects_out_of_grid_result(self) -> None:
        with self.assertRaises(OpenRouterError):
            validate_target_sector({"sector_x": 5, "sector_y": 1})


if __name__ == "__main__":
    unittest.main()
