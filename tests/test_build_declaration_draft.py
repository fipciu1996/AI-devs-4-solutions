from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SENDIT_DIR = REPO_ROOT / "sendit"
for candidate in (str(REPO_ROOT), str(SENDIT_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from devs_utilities.openrouter import ChatCompletionResult
from build_declaration_draft import build_draft_with_tool_calling, resolve_system_prompt_path


class ResolveSystemPromptPathTests(unittest.TestCase):
    def test_falls_back_to_sendit_directory_for_legacy_bare_filename(self) -> None:
        resolved = resolve_system_prompt_path(Path("openrouter_system_prompt.txt"), REPO_ROOT)

        self.assertEqual(resolved, SENDIT_DIR / "openrouter_system_prompt.txt")

    def test_build_draft_reprompts_after_invalid_json(self) -> None:
        client = type(
            "StubClient",
            (),
            {
                "__init__": lambda self: setattr(
                    self,
                    "responses",
                    [
                        ChatCompletionResult(content="not json", tool_calls=[]),
                        ChatCompletionResult(
                            content='{"status":"READY","declaration_text":"","review_notes":[],"evidence":{},"cheapest_legal_option_summary":"ok"}',
                            tool_calls=[],
                        ),
                    ],
                ),
                "create_completion": lambda self, messages, **_kwargs: self.responses.pop(0),
            },
        )()

        result = build_draft_with_tool_calling(
            client,
            system_prompt="system",
            documentation_bundle="docs",
            shipment={"sender_id": "A1"},
        )

        self.assertEqual(result["status"], "READY")


if __name__ == "__main__":
    unittest.main()
