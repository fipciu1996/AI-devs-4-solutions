from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
PHONECALL_DIR = REPO_ROOT / "phonecall"
for candidate in (str(REPO_ROOT), str(PHONECALL_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from devs_utilities.openrouter import OpenRouterError
from solve_phonecall import (
    ResponseAnalysis,
    analyze_response,
    parse_response_analysis_payload,
    resolve_edge_tts_voice,
    synthesize_audio,
)


class PhonecallTests(unittest.TestCase):
    def test_resolve_edge_tts_voice_falls_back_to_polish_voice(self) -> None:
        self.assertEqual(resolve_edge_tts_voice("ash"), "pl-PL-MarekNeural")
        self.assertEqual(resolve_edge_tts_voice("pl-PL-ZofiaNeural"), "pl-PL-ZofiaNeural")

    def test_synthesize_audio_uses_edge_tts_when_openrouter_tts_fails(self) -> None:
        with (
            patch("solve_phonecall.synthesize_audio_openrouter", side_effect=RuntimeError("404")),
            patch(
                "solve_phonecall.synthesize_audio_with_edge_tts",
                return_value=(b"mp3-bytes", "Dzien dobry"),
            ) as edge_tts_mock,
        ):
            audio_bytes, transcript = synthesize_audio("Dzien dobry", voice="ash")

        self.assertEqual(audio_bytes, b"mp3-bytes")
        self.assertEqual(transcript, "Dzien dobry")
        edge_tts_mock.assert_called_once()

    def test_parse_response_analysis_payload_reads_route_statuses(self) -> None:
        analysis = parse_response_analysis_payload(
            {
                "summary": "Only RD820 is passable.",
                "asks_for_password": False,
                "asks_for_reason": False,
                "route_statuses": {
                    "RD224": "nieprzejezdna",
                    "RD472": "nieprzejezdna",
                    "RD820": "przejezdna",
                },
                "monitoring_disabled": False,
                "success_signal": False,
                "failure_signal": False,
            },
            transcription="Road status delivered.",
        )

        self.assertEqual(analysis.route_statuses["RD224"], "nieprzejezdna")
        self.assertEqual(analysis.route_statuses["RD472"], "nieprzejezdna")
        self.assertEqual(analysis.route_statuses["RD820"], "przejezdna")

    def test_analyze_response_uses_response_text_when_audio_transcription_fails(self) -> None:
        with (
            patch("solve_phonecall.save_response_audio", return_value=b"fake-wav"),
            patch(
                "solve_phonecall.transcribe_audio_text",
                side_effect=OpenRouterError("No endpoints found that support input audio"),
            ),
            patch(
                "solve_phonecall.analyze_operator_response_with_openrouter",
                return_value=ResponseAnalysis(
                    transcription="Password required.",
                    summary="Operator asks for the password.",
                    asks_for_password=True,
                    route_statuses={route: "unknown" for route in ("RD224", "RD472", "RD820")},
                ),
            ) as analysis_mock,
        ):
            analysis = analyze_response(
                {"message": "Password required."},
                output_stub=Path("phonecall-test"),
            )

        self.assertTrue(analysis.asks_for_password)
        self.assertIn("Password required.", analysis.transcription)
        analysis_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
