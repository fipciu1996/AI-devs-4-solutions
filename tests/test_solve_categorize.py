from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CATEGORIZE_DIR = REPO_ROOT / "categorize"
for candidate in (str(REPO_ROOT), str(CATEGORIZE_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from devs_utilities.openrouter import OpenRouterError
from solve_categorize import (
    Item,
    build_prompt,
    should_use_model_prompt_optimizer,
    validate_model_prompt_prefix,
)


class CategorizePromptTests(unittest.TestCase):
    def test_prompt_optimizer_is_disabled_by_default(self) -> None:
        self.assertFalse(should_use_model_prompt_optimizer([]))

    def test_prompt_optimizer_can_be_enabled_explicitly(self) -> None:
        self.assertTrue(should_use_model_prompt_optimizer(["--use-model"]))

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
