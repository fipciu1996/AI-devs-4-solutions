from __future__ import annotations

import unittest

from devs_utilities.flags import extract_flag


class FlagExtractionTests(unittest.TestCase):
    def test_extract_flag_from_string(self) -> None:
        self.assertEqual(extract_flag("{FLG:test}"), "{FLG:test}")

    def test_extract_flag_from_dict(self) -> None:
        payload = {"status": "ok", "result": {"flag": "{FLG:nested}"}}
        self.assertEqual(extract_flag(payload), "{FLG:nested}")

    def test_extract_flag_from_list(self) -> None:
        payload = ["noise", {"message": "still no"}, "{FLG:list}"]
        self.assertEqual(extract_flag(payload), "{FLG:list}")

    def test_extract_flag_returns_none_when_missing(self) -> None:
        payload = {"status": "ok", "result": ["noise", {"another": "value"}]}
        self.assertIsNone(extract_flag(payload))


if __name__ == "__main__":
    unittest.main()
