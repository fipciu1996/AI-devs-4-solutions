from __future__ import annotations

import unittest

from domatowo.solve_domatowo import (
    MapState,
    build_clusters,
    build_side_quest_plan,
    classify_log_heuristically,
    parse_church_clue,
    run_with_retries,
    resolve_side_secret_answer,
    SIDE_SECRET_HASH,
    group_adjacent_positions,
)


class DomatowoSolverTests(unittest.TestCase):
    CHURCH_CLUE_MESSAGE = (
        "FLG? 4a,61,6b,20,6d,69,61,c5,82,20,6e,61,20,69,6d,69,65,20,67,6f,c5,9b,c4,87,"
        "20,6f,64,20,56,69,67,65,6e,c3,a8,72,65,3f,0a,2d,2d,2d,0a,50,20,75,61,73,61,71,"
        "20,71,7a,6a,6d,c5,ba,76,64,6a,70,20,70,77,20,73,72,68,74,65,74,6b,6f,76,20,78,"
        "c3,b3,77,71,20,53,64,62,6b,65,74,3f,0a,6d,6b,68,6e,66,3a,2f,2f,63,7a,73,2e,6f,"
        "65,33,61,6f,78,2e,66,66,65,2f,71,76,73,76,2f,6f,78,6e,75,6a,63,5f,67,63,70,6d,"
        "6a,6b,2e,61,6e,34"
    )

    def setUp(self) -> None:
        symbol_by_tile = {
            "road": "UL",
            "tree": "DR",
            "empty": "  ",
            "block3": "B3",
            "block2": "B2",
            "church": "KS",
        }
        grid = (
            ("tree", "road", "road", "road", "empty", "block3", "block3", "tree", "empty", "empty", "empty"),
            ("tree", "tree", "empty", "road", "road", "block3", "block3", "tree", "road", "empty", "empty"),
            ("empty",) * 11,
            ("empty",) * 11,
            ("empty",) * 11,
            ("road",) * 11,
            ("block2", "block2", "empty", "road", "empty", "church", "church", "church", "empty", "tree", "empty"),
            ("block2", "block2", "empty", "road", "empty", "church", "church", "church", "empty", "tree", "empty"),
            ("empty", "road", "road", "road", "road", "road", "road", "road", "road", "road", "empty"),
            ("block3", "block3", "block3", "empty", "tree", "empty", "empty", "block3", "block3", "tree", "empty"),
            ("block3", "block3", "block3", "empty", "tree", "empty", "empty", "block3", "block3", "tree", "empty"),
        )
        self.map_state = MapState(size=11, grid=grid, symbol_by_tile=symbol_by_tile)

    def test_group_adjacent_positions_splits_three_clusters(self) -> None:
        groups = group_adjacent_positions(("F1", "G1", "F2", "G2", "A10", "B10", "A11", "H10", "I10"))
        self.assertEqual(3, len(groups))
        self.assertIn(("F1", "G1", "F2", "G2"), groups)

    def test_build_clusters_prefers_safe_anchor_tiles(self) -> None:
        block_positions = (
            "F1",
            "G1",
            "F2",
            "G2",
            "A10",
            "B10",
            "C10",
            "A11",
            "B11",
            "C11",
            "H10",
            "I10",
            "H11",
            "I11",
        )
        clusters = build_clusters(self.map_state, block_positions)
        anchors = {cluster.region: cluster.anchor for cluster in clusters}
        self.assertEqual("E2", anchors["north"])
        self.assertEqual("C9", anchors["southwest"])
        self.assertEqual("I9", anchors["southeast"])

    def test_build_side_quest_plan_points_to_church_secret(self) -> None:
        plan = build_side_quest_plan(self.map_state)
        self.assertEqual("F9", plan.anchor)
        self.assertEqual("F8", plan.predicted_dismount)
        self.assertEqual("G8", plan.secret_field)

    def test_log_heuristics_detect_positive_and_negative_messages(self) -> None:
        found = classify_log_heuristically(
            "Mamy osob\u0119. M\u0119\u017cczyzna mniej wi\u0119cej 30-letni, schowa\u0142 si\u0119 pod przewr\u00f3conym sto\u0142em."
        )
        empty = classify_log_heuristically(
            "Nie odnotowano cz\u0142owieka. Jedynie przewr\u00f3cona szafa i druty."
        )
        self.assertEqual("found", found.result)
        self.assertEqual("empty", empty.result)

    def test_parse_church_clue_recovers_hidden_media_url(self) -> None:
        clue = parse_church_clue(self.CHURCH_CLUE_MESSAGE)
        self.assertIn("Jak mia", clue["decoded_message"])
        self.assertEqual("Jak mia\u0142 na imie go\u015b\u0107 od Vigen\u00e8re?", clue["question"])
        self.assertEqual("https://example.invalid/dane/azazel_secret.mp4", clue["media_url"])

    def test_resolve_side_secret_answer_matches_corrected_hash(self) -> None:
        self.assertEqual("plane", resolve_side_secret_answer(SIDE_SECRET_HASH))

    def test_run_with_retries_returns_first_successful_attempt(self) -> None:
        attempts: list[int] = []

        def runner(attempt: int) -> str:
            attempts.append(attempt)
            if attempt < 3:
                raise RuntimeError("retry me")
            return "ok"

        self.assertEqual("ok", run_with_retries(10, runner))
        self.assertEqual([1, 2, 3], attempts)

    def test_run_with_retries_raises_after_limit(self) -> None:
        attempts: list[int] = []

        def runner(attempt: int) -> str:
            attempts.append(attempt)
            raise RuntimeError(f"boom-{attempt}")

        with self.assertRaises(RuntimeError):
            run_with_retries(3, runner)
        self.assertEqual([1, 2, 3], attempts)


if __name__ == "__main__":
    unittest.main()
