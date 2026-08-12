from __future__ import annotations

import copy
import unittest
from pathlib import Path

from scripts.research.taro_o1r_r10_clear_runtime import fresh_pool as r10_pool
from scripts.research.taro_o1r_r11_abstention_runtime import fresh_pool


ROOT = Path(__file__).resolve().parents[3]


class FreshPoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pool = fresh_pool.build_pool(ROOT)

    def test_exact_48_parent_pool_is_recomputable_and_r10_disjoint(self) -> None:
        self.assertEqual(fresh_pool.validate_pool(self.pool, repo_root=ROOT), self.pool)
        self.assertEqual(len(self.pool["pool"]), 48)
        self.assertEqual(len({row["visit_id"] for row in self.pool["pool"]}), 48)
        old = {value for visit, video, _rank in r10_pool.EXPECTED_POOL for value in (visit, video)}
        new = {value for row in self.pool["pool"] for value in (row["visit_id"], row["video_id"])}
        self.assertFalse(old & new)

    def test_exact_144_url_plan_has_no_execution_authority(self) -> None:
        request = self.pool["request_plan"]
        self.assertEqual(request["request_count"], 144)
        self.assertEqual(len({row["url"] for row in request["requests"]}), 144)
        self.assertEqual(request["response_body_bytes_allowed"], 0)
        self.assertFalse(self.pool["network_authorized"])
        self.assertFalse(self.pool["data_use_authorized"])

    def test_roster_or_authority_mutation_fails_closed(self) -> None:
        mutated = copy.deepcopy(self.pool)
        mutated["pool"][0]["video_id"] = "00000000"
        with self.assertRaises(fresh_pool.FreshPoolError):
            fresh_pool.validate_pool(mutated, repo_root=ROOT, recompute=False)
        mutated = copy.deepcopy(self.pool)
        mutated["network_authorized"] = True
        with self.assertRaises(fresh_pool.FreshPoolError):
            fresh_pool.validate_pool(mutated, repo_root=ROOT, recompute=False)


if __name__ == "__main__":
    unittest.main()
