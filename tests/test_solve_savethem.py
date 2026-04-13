"""Unit tests for the `savethem` route planner."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from devs_utilities.ag3nts import build_ag3nts_api_url
from devs_utilities.http import HttpRequestError
from savethem.solve_savethem import (
    Position,
    PreviewState,
    VehicleSpec,
    choose_route_with_openrouter,
    find_best_route,
    with_ag3nts_retry,
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

    def test_with_ag3nts_retry_retries_transient_http_error(self) -> None:
        attempts = {"count": 0}

        def flaky_call():
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise HttpRequestError(
                    url=build_ag3nts_api_url("wehicles"),
                    message="HTTP 429",
                    status_code=429,
                    body='{"code":-9999,"message":"Za często"}',
                )
            return {"ok": True}

        with patch("savethem.solve_savethem.time.sleep"):
            result = with_ag3nts_retry("vehicle lookup", flaky_call)

        self.assertEqual(result, {"ok": True})
        self.assertEqual(attempts["count"], 2)

    def test_choose_route_with_openrouter_accepts_list_payload(self) -> None:
        preview = PreviewState(
            terrain=((".", "."), (".", "G")),
            start=Position(row=0, col=0),
            goal=Position(row=1, col=1),
        )
        specs = {
            "walk": VehicleSpec(name="walk", fuel_per_move=0.0, food_per_move=1.0),
        }

        class FakeClient:
            def create_completion(self, _messages, tools=None):  # type: ignore[no-untyped-def]
                del tools
                return type("Completion", (), {"content": '["walk", "right", "down"]', "tool_calls": []})()

        with (
            patch("savethem.solve_savethem.build_openrouter_client", return_value=FakeClient()),
            patch("savethem.solve_savethem.write_json"),
        ):
            answer = choose_route_with_openrouter(preview, specs, ["walk", "down", "right"])

        self.assertEqual(answer, ["walk", "right", "down"])


if __name__ == "__main__":
    unittest.main()
