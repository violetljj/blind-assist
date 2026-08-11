from __future__ import annotations

import copy
import unittest

from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o1r_r8_clear_runtime import pool_cohort
from scripts.research.taro_o1r_r8_clear_runtime import run_pool_inventory as runner


def _synthetic_inventory() -> dict:
    parents = []
    for visit, video, rank in pool_cohort.EXPECTED_POOL:
        parents.append({"visit_id": visit, "video_id": video, "pool_rank_sha256": rank, "frame_plan": {"exact_pose_bounded_frame_count": 1, "exact_timestamp_tokens": ["1.0"]}, "materialized_bytes": 1})
    value = {"schema": "blindassist.taro.o1r.r8_clear_pool_inventory.v1", "parent_count": 24, "exact_pose_bounded_frame_count": 24, "materialized_bytes": 24, "parents": parents, "zip_crc_validated": True, "pixel_arrays_decoded": False, "faro_values_interpreted": False, "truth_values_interpreted": False, "model_outputs_read": False, "training": False}
    value["content_sha256"] = adapter.canonical_sha256(value)
    return value


class PoolInventoryTests(unittest.TestCase):
    def test_exact_roster_and_frozen_counts_validate(self) -> None:
        result = runner.validate_inventory(_synthetic_inventory(), expected_frame_counts=[1] * 24, expected_materialized_bytes=24)
        self.assertEqual(result["parent_count"], 24)
        self.assertFalse(result["pixel_arrays_decoded"])
        self.assertFalse(result["faro_values_interpreted"])

    def test_roster_tamper_is_rejected(self) -> None:
        value = _synthetic_inventory()
        value["parents"][0]["visit_id"] = "tampered"
        value.pop("content_sha256")
        value["content_sha256"] = adapter.canonical_sha256(value)
        with self.assertRaises(runner.PoolInventoryError):
            runner.validate_inventory(value)


if __name__ == "__main__":
    unittest.main()
