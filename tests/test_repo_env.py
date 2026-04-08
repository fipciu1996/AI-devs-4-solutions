from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import devs_utilities.repo_env as repo_env


class RepoEnvAliasTests(unittest.TestCase):
    def test_get_course_api_key_falls_back_to_ag3nts_alias(self) -> None:
        with patch.dict(os.environ, {"AG3NTS_API_KEY": "course-token"}, clear=True):
            self.assertEqual(repo_env.get_course_api_key(), "course-token")

    def test_get_llm_api_key_falls_back_to_openrouter_alias(self) -> None:
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "llm-token"}, clear=True):
            self.assertEqual(repo_env.get_llm_api_key(), "llm-token")

    def test_get_llm_base_url_falls_back_to_openrouter_alias(self) -> None:
        with patch.dict(os.environ, {"OPENROUTER_BASE_URL": "https://openrouter.example"}, clear=True):
            self.assertEqual(repo_env.get_llm_base_url(), "https://openrouter.example")

    def test_get_llm_model_falls_back_to_openrouter_model(self) -> None:
        with patch.dict(os.environ, {"OPENROUTER_MODEL": "shared-model"}, clear=True):
            self.assertEqual(repo_env.get_llm_model(), "shared-model")

    def test_get_llm_model_prefers_task_specific_alias(self) -> None:
        with patch.dict(
            os.environ,
            {
                "FIRMWARE_MODEL": "firmware-model",
                "OPENROUTER_MODEL": "shared-model",
            },
            clear=True,
        ):
            self.assertEqual(repo_env.get_llm_model("FIRMWARE_MODEL"), "firmware-model")

    def test_legacy_names_still_win_when_present(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LLM_API_KEY": "legacy-llm-token",
                "OPENROUTER_API_KEY": "new-llm-token",
            },
            clear=True,
        ):
            self.assertEqual(repo_env.get_llm_api_key(), "legacy-llm-token")


if __name__ == "__main__":
    unittest.main()
