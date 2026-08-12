from __future__ import annotations

import unittest
from pathlib import Path

from scripts.research.taro_o1r_r7_canary_runtime import run_fresh_phase_a as shared_r7
from scripts.research.taro_o1r_r10_clear_runtime import run_pool_phase_a as runner


class PoolPhaseATests(unittest.TestCase):
    def test_exact_r10_constants_do_not_mutate_shared_r7_runtime(self) -> None:
        self.assertEqual(runner.PARENT_COUNT, 32)
        self.assertEqual(runner.FRAME_COUNT, 710)
        self.assertEqual(runner.QUERY_COUNT, 6390)
        self.assertEqual(5 * runner.FRAME_COUNT + 4, 3554)
        self.assertEqual(shared_r7.PARENT_COUNT, 8)
        self.assertEqual(shared_r7.FRAME_COUNT, 170)

    def test_exact_inventory_loads_710_unique_frames_without_faro(self) -> None:
        frames = runner._load_frames(Path(runner.REPO_ROOT) / runner.INVENTORY_PATH)
        self.assertEqual(len(frames), 710)
        self.assertEqual(len({frame.physical_frame_id for frame in frames}), 710)
        self.assertFalse(any("faro" in str(member).lower() for frame in frames for member in frame.members))

    def test_r10_schema_is_sealed_and_legacy_schema_is_rejected(self) -> None:
        schema = "blindassist.taro.o1r.r10_fresh_pool_candidate_input.v1"
        record = runner._seal({"schema": schema, "value": 1})
        self.assertEqual(runner._validate_seal(record, schema), record)
        with self.assertRaises(runner.FreshPhaseAError):
            runner._validate_seal(record, "blindassist.taro.o1r.r7_fresh_candidate_input.v1")


if __name__ == "__main__":
    unittest.main()
