from __future__ import annotations

import unittest
from pathlib import Path

from scripts.research.taro_o1r_r10_clear_runtime import fresh_pool
from scripts.research.taro_o1r_r10_clear_runtime import run_pool_download as runner


class PoolDownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = fresh_pool.build_pool(Path(__file__).resolve().parents[3])

    def test_expanded_download_plan_is_exact_and_unique(self) -> None:
        rows = runner.expanded_download_plan(self.plan)
        self.assertEqual(len(rows), 96)
        self.assertEqual(len({row["relative_path"] for row in rows}), 96)
        self.assertTrue(all(row["relative_path"].startswith(("upsampling/Training/", "raw/Training/")) for row in rows))

    def test_no_decode_or_faro_authority(self) -> None:
        authority = runner.EXPECTED_AUTHORITY
        self.assertTrue(authority["source_download"])
        self.assertFalse(authority["archive_decode"])
        self.assertFalse(authority["source_frame_decode"])
        self.assertFalse(authority["faro_read"])
        self.assertFalse(authority["model_execution"])


if __name__ == "__main__":
    unittest.main()
