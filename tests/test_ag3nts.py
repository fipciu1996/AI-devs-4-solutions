from __future__ import annotations

import unittest

from devs_utilities.ag3nts import (
    AG3NTS_LOCATION_URL,
    AG3NTS_RAILWAY_URL,
    AG3NTS_VERIFY_URL,
    AG3NTS_ZMAIL_URL,
)


class Ag3ntsUrlTests(unittest.TestCase):
    def test_railway_url_uses_verify_endpoint(self) -> None:
        self.assertEqual(AG3NTS_RAILWAY_URL, AG3NTS_VERIFY_URL)

    def test_location_url_does_not_repeat_slashes(self) -> None:
        self.assertEqual(AG3NTS_LOCATION_URL, "https://example.invalid/location")

    def test_zmail_url_does_not_repeat_slashes(self) -> None:
        self.assertEqual(AG3NTS_ZMAIL_URL, "https://example.invalid/api/zmail")


if __name__ == "__main__":
    unittest.main()
