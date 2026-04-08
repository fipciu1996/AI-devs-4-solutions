from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
FINDHIM_DIR = REPO_ROOT / "findhim"
for candidate in (str(REPO_ROOT), str(FINDHIM_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from solve_findhim import DEFAULT_SUSPECTS_PATH, parse_args, resolve_findhim_config_path


class SolveFindhimTests(unittest.TestCase):
    def test_parse_args_defaults_point_to_split_directories(self) -> None:
        with patch.object(sys, "argv", ["solve_findhim.py"]):
            args = parse_args()

        self.assertEqual(args.config, "findhim_config.json")
        self.assertEqual(args.csv, "../people/people.csv")
        self.assertEqual(args.suspects, DEFAULT_SUSPECTS_PATH)

    def test_resolve_findhim_config_path_falls_back_to_legacy_people_config(self) -> None:
        temp_root = REPO_ROOT / "tests" / "_tmp_findhim_config"
        findhim_dir = temp_root / "findhim"
        legacy_config = temp_root / "people" / "people_config.json"
        legacy_config.parent.mkdir(parents=True, exist_ok=True)
        findhim_dir.mkdir(parents=True, exist_ok=True)
        legacy_config.write_text("{}", encoding="utf-8")

        try:
            with patch("solve_findhim.REPO_ROOT", temp_root):
                resolved = resolve_findhim_config_path(findhim_dir, "findhim_config.json")

            self.assertEqual(resolved, legacy_config)
        finally:
            for path in sorted(temp_root.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink(missing_ok=True)
                else:
                    path.rmdir()
            temp_root.rmdir()


if __name__ == "__main__":
    unittest.main()
