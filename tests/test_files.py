from __future__ import annotations

import os
import shutil
import unittest
import uuid
from pathlib import Path

from devs_utilities.files import resolve_path


class FileUtilitiesTests(unittest.TestCase):
    def test_resolve_path_prefers_existing_cwd_relative_path(self) -> None:
        original_cwd = Path.cwd()
        temp_path = Path(__file__).resolve().parent / f"_test_tmp_{uuid.uuid4().hex}"
        temp_path.mkdir()
        try:
            nested_dir = temp_path / "people"
            nested_dir.mkdir()
            existing_file = nested_dir / "people.csv"
            existing_file.write_text("name\nAlice\n", encoding="utf-8")

            os.chdir(temp_path)
            resolved = resolve_path(Path("people") / "people.csv", temp_path / "fallback")
        finally:
            os.chdir(original_cwd)
            shutil.rmtree(temp_path, ignore_errors=True)

        self.assertEqual(resolved, existing_file)


if __name__ == "__main__":
    unittest.main()
