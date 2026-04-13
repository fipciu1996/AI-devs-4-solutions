"""Regression tests for task-specific system prompt files."""

from __future__ import annotations

import unittest
from pathlib import Path

from devs_utilities.prompts import load_prompt_text
from firmware import solve_firmware


REPO_ROOT = Path(__file__).resolve().parents[1]

PROMPT_FILES = (
    Path("categorize/system_prompt.txt"),
    Path("domatowo/planner_system_prompt.txt"),
    Path("domatowo/log_classifier_system_prompt.txt"),
    Path("drone/vision_system_prompt.txt"),
    Path("electricity/vision_system_prompt.txt"),
    Path("failure/system_prompt.txt"),
    Path("filesystem/system_prompt.txt"),
    Path("firmware/system_prompt.txt"),
    Path("mailbox/system_prompt.txt"),
    Path("people/filter_system_prompt.txt"),
    Path("phonecall/response_analysis_system_prompt.txt"),
    Path("findhim/geocode_legacy_system_prompt.txt"),
    Path("findhim/geocode_system_prompt.txt"),
    Path("railway/system_prompt.txt"),
    Path("reactor/system_prompt.txt"),
    Path("savethem/system_prompt.txt"),
    Path("sendit/analyze_system_prompt.txt"),
    Path("sendit/analyze_tool_system_prompt.txt"),
    Path("sendit/draft_tool_system_prompt.txt"),
    Path("sensors/system_prompt.txt"),
    Path("shellaccess/system_prompt.txt"),
)

MODULE_PROMPT_REFERENCES = {
    Path("categorize/solve_categorize.py"): ("system_prompt.txt",),
    Path("domatowo/solve_domatowo.py"): (
        "planner_system_prompt.txt",
        "log_classifier_system_prompt.txt",
    ),
    Path("drone/solve_drone.py"): ("vision_system_prompt.txt",),
    Path("electricity/solve_electricity.py"): ("vision_system_prompt.txt",),
    Path("failure/solve_failure.py"): ("system_prompt.txt",),
    Path("filesystem/solve_filesystem.py"): ("system_prompt.txt",),
    Path("firmware/solve_firmware.py"): ("system_prompt.txt",),
    Path("mailbox/solve_mailbox.py"): ("system_prompt.txt",),
    Path("people/filter_people.py"): ("filter_system_prompt.txt",),
    Path("phonecall/solve_phonecall.py"): ("response_analysis_system_prompt.txt",),
    Path("findhim/solve_findhim.py"): (
        "geocode_legacy_system_prompt.txt",
        "geocode_system_prompt.txt",
    ),
    Path("railway/route_agent.py"): ("system_prompt.txt",),
    Path("reactor/solve_reactor.py"): ("system_prompt.txt",),
    Path("savethem/solve_savethem.py"): ("system_prompt.txt",),
    Path("sendit/analyze_attachments_openrouter.py"): (
        "analyze_system_prompt.txt",
        "analyze_tool_system_prompt.txt",
    ),
    Path("sendit/build_declaration_draft.py"): ("draft_tool_system_prompt.txt",),
    Path("sensors/solve_sensors.py"): ("system_prompt.txt",),
    Path("shellaccess/solve_shellaccess.py"): ("system_prompt.txt",),
}


class PromptFileTests(unittest.TestCase):
    def test_prompt_files_exist_and_are_non_empty(self) -> None:
        for relative_path in PROMPT_FILES:
            prompt_path = REPO_ROOT / relative_path
            with self.subTest(prompt_path=str(relative_path)):
                self.assertTrue(prompt_path.exists(), f"Missing prompt file: {relative_path}")
                self.assertTrue(
                    prompt_path.read_text(encoding="utf-8").strip(),
                    f"Prompt file is empty: {relative_path}",
                )

    def test_modules_reference_prompt_files(self) -> None:
        for relative_path, prompt_files in MODULE_PROMPT_REFERENCES.items():
            source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
            with self.subTest(module=str(relative_path)):
                self.assertIn("load_prompt_text", source)
                for prompt_file in prompt_files:
                    self.assertIn(prompt_file, source)

    def test_prompt_loader_matches_repository_file_contents(self) -> None:
        source_path = REPO_ROOT / "categorize" / "solve_categorize.py"
        prompt_path = REPO_ROOT / "categorize" / "system_prompt.txt"
        expected = prompt_path.read_text(encoding="utf-8").strip()
        self.assertEqual(load_prompt_text(source_path, "system_prompt.txt"), expected)

    def test_firmware_prompt_template_renders_dynamic_sections(self) -> None:
        rendered = solve_firmware.build_system_prompt(
            ["help - show commands", "cat - print file"],
            ["*.tmp", "cache/"],
            verify_enabled=True,
            password_hint="sprinkler",
        )
        self.assertIn("Stop only after submit_confirmation returns a flag.", rendered)
        self.assertIn("- help - show commands", rendered)
        self.assertIn("- *.tmp", rendered)
        self.assertIn("User supplied password hint: sprinkler", rendered)
        self.assertNotIn("{verification_line}", rendered)


if __name__ == "__main__":
    unittest.main()
