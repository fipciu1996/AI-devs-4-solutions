from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
FINDHIM_DIR = REPO_ROOT / "findhim"
for candidate in (str(REPO_ROOT), str(FINDHIM_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from solve_findhim import (
    DEFAULT_SUSPECTS_PATH,
    CandidateMatch,
    Coordinate,
    build_power_plants_from_coordinates,
    build_match_from_known_answer,
    extract_geocoded_coordinates,
    load_city_coordinate_overrides,
    load_known_verified_answer,
    normalize_name,
    PowerPlant,
    parse_args,
    parse_json_content,
    resolve_findhim_config_path,
    Suspect,
)
from devs_utilities.openrouter import parse_json_object_content


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

    def test_parse_json_object_content_handles_fenced_payload(self) -> None:
        parsed = parse_json_object_content(
            """```json
{"plants":[{"city":"Gdansk","latitude":54.35,"longitude":18.65}]}
```"""
        )

        self.assertEqual(parsed["plants"][0]["city"], "Gdansk")

    def test_parse_json_content_accepts_top_level_list_payload(self) -> None:
        parsed = parse_json_content(
            """```json
[{"city":"Gdansk","latitude":54.35,"longitude":18.65}]
```"""
        )

        self.assertEqual(parsed[0]["city"], "Gdansk")

    def test_extract_geocoded_coordinates_accepts_city_mapping_shape(self) -> None:
        coordinates = extract_geocoded_coordinates(
            {
                "Gdansk": {"latitude": 54.35, "longitude": 18.65},
                "Gdynia": [54.52, 18.53],
            }
        )

        self.assertEqual(coordinates["gdansk"], Coordinate(latitude=54.35, longitude=18.65))
        self.assertEqual(coordinates["gdynia"], Coordinate(latitude=54.52, longitude=18.53))

    def test_local_city_coordinate_overrides_cover_known_power_plants(self) -> None:
        overrides = load_city_coordinate_overrides(FINDHIM_DIR / "city_coordinates.json")
        plants = build_power_plants_from_coordinates(
            {
                "power_plants": {
                    "Grudziądz": {"code": "PWR7264PL"},
                    "Żarnowiec": {"code": "PWR6132PL"},
                }
            },
            overrides,
        )

        self.assertEqual([plant.city for plant in plants], ["Grudziądz", "Żarnowiec"])
        self.assertEqual(plants[0].code, "PWR7264PL")

    def test_normalize_name_matches_chelmno_with_and_without_diacritics(self) -> None:
        self.assertEqual(normalize_name("Chełmno"), normalize_name("Chelmno"))

    def test_load_known_verified_answer_reads_cached_answer(self) -> None:
        path = REPO_ROOT / "tests" / "_tmp_findhim_known_answer.json"
        path.write_text(
            json.dumps(
                {
                    "answer": {
                        "name": "Wojciech",
                        "surname": "Bielik",
                        "accessLevel": 7,
                        "powerPlant": "PWR2758PL",
                    }
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        try:
            answer = load_known_verified_answer(path)
        finally:
            path.unlink(missing_ok=True)

        self.assertEqual(answer["powerPlant"], "PWR2758PL")

    def test_build_match_from_known_answer_uses_cached_payload(self) -> None:
        match = build_match_from_known_answer(
            {
                "name": "Wojciech",
                "surname": "Bielik",
                "accessLevel": 7,
                "powerPlant": "PWR2758PL",
            },
            suspects=[
                Suspect(name="Wojciech", surname="Bielik", birth_year=1986),
            ],
            plants=[
                PowerPlant(
                    city="Chełmno",
                    code="PWR2758PL",
                    coordinate=Coordinate(latitude=53.3485, longitude=18.4251),
                )
            ],
        )

        self.assertIsInstance(match, CandidateMatch)
        self.assertEqual(match.access_level, 7)
        self.assertEqual(match.power_plant.code, "PWR2758PL")


if __name__ == "__main__":
    unittest.main()
