"""Unit tests for the negotiations catalog matcher."""

from __future__ import annotations

import unittest
from pathlib import Path

from negotiations_api.catalog import format_match_output, load_catalog_from_csv


class CatalogMatchingTests(unittest.TestCase):
    """Validate matching against a tiny synthetic dataset."""

    def setUp(self) -> None:
        base = Path(__file__).resolve().parent / "tests_data"
        self.cities_path = base / "cities.csv"
        self.items_path = base / "items.csv"
        self.connections_path = base / "connections.csv"

        self.catalog = load_catalog_from_csv(
            self.cities_path,
            self.items_path,
            self.connections_path,
        )

    def test_matches_inverter_synonym_query(self) -> None:
        match = self.catalog.find_best_match("Potrzebuję przetwornicy 48V 3000W")
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.item.code, "A94MAZ")
        self.assertEqual(match.cities, ("Bydgoszcz", "Domatowo", "Skolwin"))

    def test_matches_battery_query(self) -> None:
        match = self.catalog.find_best_match("Szukam akumulatora AGM 48V 150Ah")
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.item.code, "06OTEA")

    def test_matches_inflected_turbine_query(self) -> None:
        match = self.catalog.find_best_match(
            "Potrzebuję turbiny wiatrowej 400W 48V do zasilania"
        )
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.item.code, "WITR48")

    def test_formats_short_response(self) -> None:
        match = self.catalog.find_best_match("turbina wiatrowa 400W 48V")
        self.assertIsNotNone(match)
        assert match is not None
        output = format_match_output(match)
        self.assertIn("WITR48", output)
        self.assertLessEqual(len(output.encode("utf-8")), 500)


if __name__ == "__main__":
    unittest.main()
