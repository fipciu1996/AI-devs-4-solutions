"""Unit tests for the `savethem` route planner."""

from __future__ import annotations

import unittest

from savethem.solve_savethem import (
    Position,
    PreviewState,
    VehicleSpec,
    find_best_route,
)


class SaveThemRouteTests(unittest.TestCase):
    def test_find_best_route_prefers_rocket_then_dismount(self) -> None:
        terrain = (
            (".", ".", ".", ".", ".", ".", ".", ".", "W", "W"),
            (".", ".", ".", ".", ".", ".", ".", "W", "W", "."),
            (".", "T", ".", ".", ".", ".", "W", "W", ".", "."),
            (".", ".", ".", ".", ".", ".", "W", ".", ".", "."),
            (".", ".", "T", ".", ".", ".", "W", ".", "G", "."),
            (".", ".", ".", ".", "R", ".", "W", ".", ".", "."),
            (".", ".", ".", "R", "R", ".", "W", "W", ".", "."),
            ("S", "R", ".", ".", ".", ".", ".", "W", ".", "."),
            (".", ".", ".", ".", ".", ".", "W", "W", ".", "."),
            (".", ".", ".", ".", ".", "W", "W", ".", ".", "."),
        )
        preview = PreviewState(
            terrain=terrain,
            start=Position(row=7, col=0),
            goal=Position(row=4, col=8),
        )
        specs = {
            "walk": VehicleSpec(name="walk", fuel_per_move=0.0, food_per_move=2.5),
            "horse": VehicleSpec(name="horse", fuel_per_move=0.0, food_per_move=1.6),
            "car": VehicleSpec(name="car", fuel_per_move=0.7, food_per_move=1.0),
            "rocket": VehicleSpec(name="rocket", fuel_per_move=1.0, food_per_move=0.1),
        }

        answer = find_best_route(preview, specs)

        self.assertEqual(
            answer,
            [
                "rocket",
                "up",
                "up",
                "right",
                "right",
                "right",
                "up",
                "right",
                "right",
                "dismount",
                "right",
                "right",
                "right",
            ],
        )


if __name__ == "__main__":
    unittest.main()
