from __future__ import annotations

import unittest
from pathlib import Path

from scripts.research.taro_o1r_r10_clear_runtime import fresh_pool


class FreshPoolTests(unittest.TestCase):
    def test_pool_is_recomputable_and_disjoint(self) -> None:
        root = Path(__file__).resolve().parents[3]
        pool = fresh_pool.build_pool(root)
        self.assertEqual(pool["pool_parent_count"], 32)
        self.assertEqual(pool["request_plan"]["request_count"], 96)
        self.assertEqual(len({row["visit_id"] for row in pool["pool"]}), 32)
        self.assertFalse(pool["network_authorized"])


if __name__ == "__main__":
    unittest.main()
