from __future__ import annotations

from unittest.mock import patch
import unittest

from timetravel.backend_api import (
    TimetravelApiClient,
    extract_flag_from_ui_text,
    extract_stabilization_value,
)


class TimetravelBackendApiTests(unittest.TestCase):
    def test_extract_stabilization_value_supports_structured_payload(self) -> None:
        payload = {"guidance": {"stabilization": 321}}

        self.assertEqual(extract_stabilization_value(payload), 321)

    def test_extract_stabilization_value_supports_text_payload(self) -> None:
        payload = {"message": "Set stabilization to 654 before activating the core."}

        self.assertEqual(extract_stabilization_value(payload), 654)

    def test_extract_stabilization_value_prefers_needconfig_over_config_value(self) -> None:
        payload = {
            "config": {"stabilization": 0},
            "needConfig": (
                "Moduł historyczny zakończył analizę warunków podróży. "
                "Dla tego rodzaju skoku podręczniki operatora sugerują zwykle "
                "dziewięćset jednostek. Mimo to zalecane jest obniżenie poziomu "
                "o siedemset jedenaście."
            ),
        }

        self.assertEqual(extract_stabilization_value(payload), 189)

    def test_extract_stabilization_value_handles_polish_addition_guidance(self) -> None:
        payload = {
            "needConfig": (
                "Po porównaniu bieżących odczytów z archiwami epoki stwierdzono "
                "następującą rekomendację. W dokumentacji serwisowej dla podobnych "
                "warunków najczęściej pojawia się poziom sześćset. Jednak z uwagi "
                "na wzmożoną aktywność Słońca warto zwiększyć tę nastawę o 395 punktów."
            )
        }

        self.assertEqual(extract_stabilization_value(payload), 995)

    def test_extract_stabilization_value_handles_polish_subtraction_guidance(self) -> None:
        payload = {
            "needConfig": (
                "Po zestawieniu raportów z ostatnich misji wybrano następujący "
                "wariant konfiguracji. Najbardziej typowy poziom dla podobnej podróży "
                "to wartość bazowa wynosząca siedemset. Ze względu na poprawioną "
                "wydajność nowych stabilizatorów można od tej wartości odjąć "
                "sto dwanaście jednostek."
            )
        }

        self.assertEqual(extract_stabilization_value(payload), 588)

    def test_extract_stabilization_value_returns_none_for_unrelated_text(self) -> None:
        self.assertIsNone(extract_stabilization_value({"message": "No guidance available."}))

    def test_extract_flag_from_ui_text_rejects_non_flag_text(self) -> None:
        self.assertIsNone(extract_flag_from_ui_text("copied"))
        self.assertEqual(extract_flag_from_ui_text("{FLG:test}"), "{FLG:test}")

    @patch("timetravel.backend_api.submit_task_answer")
    def test_client_configure_uses_expected_payload(self, mocked_submit) -> None:
        mocked_submit.return_value = {"code": 0}
        client = TimetravelApiClient(api_key="secret-key")

        result = client.configure("year", 2024)

        self.assertEqual(result, {"code": 0})
        mocked_submit.assert_called_once()
        _, kwargs = mocked_submit.call_args
        self.assertEqual(kwargs["api_key"], "secret-key")
        self.assertEqual(kwargs["task"], "timetravel")
        self.assertEqual(
            kwargs["answer"],
            {
                "action": "configure",
                "param": "year",
                "value": 2024,
            },
        )

    @patch("timetravel.backend_api.submit_task_answer")
    def test_client_get_config_normalizes_payload(self, mocked_submit) -> None:
        mocked_submit.return_value = {
            "config": {
                "currentDate": "2026-04-10",
                "day": 12,
                "month": 11,
                "year": 2024,
                "syncRatio": 0.54,
                "stabilization": 123,
                "condition": "stable",
                "fluxDensity": 100,
                "batteryStatus": "2/3",
                "PTA": True,
                "PTB": True,
                "PWR": 19,
                "mode": "active",
                "internalMode": 2,
            }
        }
        client = TimetravelApiClient(api_key="secret-key")

        config = client.get_config()

        self.assertEqual(config.current_date, "2026-04-10")
        self.assertEqual(config.configured_date, "2024-11-12")
        self.assertEqual(config.battery_level, 2)


if __name__ == "__main__":
    unittest.main()
