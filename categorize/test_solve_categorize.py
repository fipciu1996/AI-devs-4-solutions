from __future__ import annotations

import sys
import unittest
from pathlib import Path

CATEGORIZE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CATEGORIZE_DIR.parent
for candidate in (str(REPO_ROOT), str(CATEGORIZE_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from devs_utilities.openrouter import OpenRouterError
from solve_categorize import Item, build_prompt, validate_model_prompt_prefix


class CategorizePromptTests(unittest.TestCase):
    def test_validate_model_prompt_prefix_accepts_compact_prefix(self) -> None:
        items = [
            Item("A", "reactor cooling rod"),
            Item("B", "crate of rifle ammo"),
        ]

        prefix = validate_model_prompt_prefix(
            "Reply DNG or NEU. Weapons/ammo/explosives/poison/radioactive/drugs => DNG. Reactor and industrial parts => NEU.",
            items,
        )

        self.assertIn("DNG", prefix)
        self.assertLessEqual(len(build_prompt(items[0], prefix)), 300)

    def test_validate_model_prompt_prefix_rejects_over_budget_prompt(self) -> None:
        items = [Item("A", "very long description " * 40)]

        with self.assertRaises(OpenRouterError):
            validate_model_prompt_prefix(
                "Reply DNG or NEU. Reactor parts are neutral and dangerous items are dangerous.",
                items,
            )


if __name__ == "__main__":
    unittest.main()
