from __future__ import annotations

import unittest

from devs_utilities.ag3nts import (
    AG3NTS_API_BASE_URL,
    AG3NTS_LOCATION_URL,
    AG3NTS_RAILWAY_URL,
    AG3NTS_TIMETRAVEL_PREVIEW_URL,
    AG3NTS_VERIFY_URL,
    AG3NTS_ZMAIL_URL,
    build_ag3nts_api_url,
    build_ag3nts_url,
)


class Ag3ntsUrlTests(unittest.TestCase):
    def test_railway_url_uses_verify_endpoint(self) -> None:
        self.assertEqual(AG3NTS_RAILWAY_URL, AG3NTS_VERIFY_URL)

    def test_build_ag3nts_url_normalizes_leading_slashes(self) -> None:
        self.assertEqual(AG3NTS_LOCATION_URL, build_ag3nts_url("/location"))

    def test_build_ag3nts_api_url_reuses_shared_api_base(self) -> None:
        self.assertEqual(AG3NTS_ZMAIL_URL, build_ag3nts_api_url("/zmail"))
        self.assertEqual(AG3NTS_API_BASE_URL, build_ag3nts_url("api"))

    def test_timetravel_preview_uses_shared_base(self) -> None:
        self.assertEqual(
            AG3NTS_TIMETRAVEL_PREVIEW_URL,
            build_ag3nts_url("timetravel_preview"),
        )


if __name__ == "__main__":
    unittest.main()
