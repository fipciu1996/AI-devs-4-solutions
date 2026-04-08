from __future__ import annotations

import unittest

from foodwarehouse.solve_foodwarehouse import (
    CityDemand,
    Creator,
    DeliveryRequirement,
    build_planned_orders,
    parse_city_demands,
    parse_missing_requirements,
    requirement_index,
)


class FoodwarehouseSolverTests(unittest.TestCase):
    def test_parse_city_demands_normalizes_city_names(self) -> None:
        payload = {
            "opalino": {"chleb": 45, "woda": 120, "mlotek": 6},
            "domatowo": {"makaron": 60, "woda": 150, "lopata": 8},
        }

        demands = parse_city_demands(payload)

        self.assertEqual(
            [
                CityDemand(city="Opalino", items={"chleb": 45, "woda": 120, "mlotek": 6}),
                CityDemand(city="Domatowo", items={"makaron": 60, "woda": 150, "lopata": 8}),
            ],
            demands,
        )

    def test_build_planned_orders_attaches_destinations_and_creator(self) -> None:
        creator = Creator(user_id=2, login="tgajewski", birthday="1991-04-06")
        demands = [
            CityDemand(city="Opalino", items={"chleb": 45, "woda": 120, "mlotek": 6}),
            CityDemand(city="Domatowo", items={"makaron": 60, "woda": 150, "lopata": 8}),
        ]

        planned_orders = build_planned_orders(
            demands,
            {"Opalino": 991828, "Domatowo": 761834},
            creator,
        )

        self.assertEqual("Dostawa dla Opalino", planned_orders[0].title)
        self.assertEqual(991828, planned_orders[0].destination)
        self.assertEqual(creator, planned_orders[0].creator)
        self.assertEqual(761834, planned_orders[1].destination)

    def test_requirement_index_matches_backend_missing_payload(self) -> None:
        expected = [
            DeliveryRequirement(
                city="Opalino",
                destination=991828,
                items={"chleb": 45, "mlotek": 6, "woda": 120},
            ),
            DeliveryRequirement(
                city="Domatowo",
                destination=761834,
                items={"lopata": 8, "makaron": 60, "woda": 150},
            ),
        ]
        backend_rows = [
            {
                "city": "Domatowo",
                "destination": 761834,
                "items": {"woda": 150, "makaron": 60, "lopata": 8},
            },
            {
                "city": "Opalino",
                "destination": 991828,
                "items": {"woda": 120, "mlotek": 6, "chleb": 45},
            },
        ]

        self.assertEqual(
            requirement_index(expected),
            requirement_index(parse_missing_requirements(backend_rows)),
        )


if __name__ == "__main__":
    unittest.main()
