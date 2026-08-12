from __future__ import annotations

import unittest

from scripts.research.taro_o1r_r8_clear_runtime import truth_owned_fallback as fallback


class TruthOwnedFallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plane = {"evaluable": True, "normal_camera_xyz": [0.0, -1.0, 0.0], "camera_height_m": 1.5}
        self.queries = fallback.build_fallback_queries("frame", self.plane)

    def test_source_query_is_retained_exactly(self) -> None:
        receipt = {"query_id": "source"}
        selected, owner = fallback.select_query_frame({"query_receipt": receipt}, self.queries[0])
        self.assertIs(selected, receipt)
        self.assertEqual(owner, fallback.SOURCE_QUERY_OWNER)

    def test_missing_source_query_uses_truth_owned_frame(self) -> None:
        query = self.queries[4]
        selected, owner = fallback.select_query_frame({"query_receipt": None, "grid_index": 4, "query_id": query["query_id"]}, query)
        self.assertIs(selected, query)
        self.assertEqual(owner, fallback.FALLBACK_QUERY_OWNER)
        self.assertEqual(selected["virtual_query_frame"]["kind"], "FARO_TRUTH_OWNED_SUPPORT_PLANE_V1")

    def test_fallback_identity_drift_fails_closed(self) -> None:
        with self.assertRaises(fallback.TruthOwnedFallbackError):
            fallback.select_query_frame({"query_receipt": None, "grid_index": 3, "query_id": self.queries[4]["query_id"]}, self.queries[4])


if __name__ == "__main__":
    unittest.main()
