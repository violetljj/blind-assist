from __future__ import annotations

import unittest
from pathlib import Path

from scripts.research.taro_o1r_r8_clear_runtime import pool_cohort


class R8PoolCohortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pool = pool_cohort.build_pool(Path(__file__).resolve().parents[3])

    def test_exact_metadata_only_pool(self) -> None:
        observed = [(row["visit_id"], row["video_id"], row["pool_rank_sha256"]) for row in self.pool["pool"]]
        self.assertEqual(observed, pool_cohort.EXPECTED_POOL)
        self.assertEqual(len({row["visit_id"] for row in self.pool["pool"]}), 24)

    def test_request_plan_has_no_authority(self) -> None:
        self.assertEqual(self.pool["request_plan"]["request_count"], 72)
        self.assertEqual(self.pool["request_plan"]["method"], "HEAD")
        self.assertFalse(self.pool["network_authorized"])
        self.assertFalse(self.pool["source_payload_read"])
        self.assertFalse(self.pool["faro_payload_read"])


if __name__ == "__main__":
    unittest.main()
