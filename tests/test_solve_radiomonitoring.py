from __future__ import annotations

import unittest

from radiomonitoring.solve_radiomonitoring import (
    TextClue,
    TradeEntry,
    format_area,
    normalize_phone_number,
    resolve_city_name,
    resolve_warehouses_count,
)


class RadiomonitoringSolverTest(unittest.TestCase):
    def test_format_area_rounds_half_up(self) -> None:
        self.assertEqual(format_area("10.7284"), "10.73")
        self.assertEqual(format_area("12.345"), "12.35")

    def test_normalize_phone_number(self) -> None:
        self.assertEqual(normalize_phone_number("644-122-092"), "644122092")

    def test_resolve_city_name_matches_skarszewy(self) -> None:
        trades = [
            TradeEntry(city="Syjon", action="szuka", goods="kilof", quantity=None, note="bydło"),
            TradeEntry(city="Syjon", action="sprzedaje", goods="bydło", quantity=None, note="kilof"),
            TradeEntry(
                city="Skarszewy",
                action="szuka",
                goods="kilof",
                quantity=None,
                note="wołowina",
            ),
            TradeEntry(
                city="Skarszewy",
                action="sprzedaje",
                goods="wołowina",
                quantity=1,
                note="kilof",
            ),
            TradeEntry(
                city="Karlikowo",
                action="szuka",
                goods="kilof",
                quantity=247,
                note="więcej na priv",
            ),
        ]
        clues = [
            TextClue(
                capture_index=1,
                text=(
                    "Ze Skarszewami to zawsze był problem. Miasto ocalałych. "
                    "Niektórzy mówią, że to taki prawie biblijny raj."
                ),
                mentioned_words=("Skarszewami",),
                phone_numbers=(),
                decoded_morse=None,
            )
        ]

        self.assertEqual(resolve_city_name(trades, clues), "Skarszewy")

    def test_resolve_warehouses_count_uses_planned_count_minus_one(self) -> None:
        audio_analysis = {
            "transcription": "Planujemy na wiosnę wybudować dwunasty magazyn.",
            "planned_warehouses_count": 12,
        }

        self.assertEqual(resolve_warehouses_count(audio_analysis), 11)


if __name__ == "__main__":
    unittest.main()
