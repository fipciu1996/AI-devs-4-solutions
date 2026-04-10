from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
SENDIT_DIR = REPO_ROOT / "sendit"
for candidate in (str(REPO_ROOT), str(SENDIT_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from generate_legal_declaration import main


class GenerateLegalDeclarationTests(unittest.TestCase):
    def test_main_returns_zero_for_controlled_rejection_with_report(self) -> None:
        temp_root = REPO_ROOT / "tests" / "_tmp_sendit_legal"
        shipment_path = temp_root / "shipment.json"
        output_dir = temp_root / "out"
        temp_root.mkdir(parents=True, exist_ok=True)
        shipment_path.write_text(
            json.dumps(
                {
                    "sender_id": "sender-1",
                    "origin": "Gdansk",
                    "destination": "Zarnowiec",
                    "declared_mass_kg": 2800,
                    "budget_pp": 0,
                    "contents": "niejednoznaczny ladunek",
                    "special_notes": "",
                    "confirmed_legal_basis": False,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        try:
            with patch.object(
                sys,
                "argv",
                [
                    "generate_legal_declaration.py",
                    "--shipment-file",
                    str(shipment_path),
                    "--output-dir",
                    str(output_dir),
                ],
            ):
                result = main()

            self.assertEqual(result, 0)
            self.assertTrue((output_dir / "validation_report.json").exists())
        finally:
            for path in sorted(temp_root.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink(missing_ok=True)
                else:
                    path.rmdir()
            temp_root.rmdir()


if __name__ == "__main__":
    unittest.main()
