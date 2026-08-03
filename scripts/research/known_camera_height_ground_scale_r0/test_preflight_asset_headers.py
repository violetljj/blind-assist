import unittest

from preflight_asset_headers import ASSET_FILES, cohort_header_requests


class PreflightAssetHeadersTest(unittest.TestCase):
    def test_requests_exactly_five_validation_assets_per_locked_parent(self) -> None:
        lock = {
            "selected_parent_count": 4,
            "fresh_evaluation": [
                {"visit_id": str(index), "video_id": str(100 + index)}
                for index in range(4)
            ],
        }
        requests = cohort_header_requests(lock)
        self.assertEqual(20, len(requests))
        self.assertEqual(set(ASSET_FILES), {row["asset"] for row in requests})
        self.assertTrue(all("/Validation/" in row["url"] for row in requests))


if __name__ == "__main__":
    unittest.main()
