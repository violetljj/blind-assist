from __future__ import annotations

import unittest
from pathlib import Path

from scripts.research.taro_o1r_r7_canary_runtime import run_fresh_phase_a as shared
from scripts.research.taro_o1r_r8_clear_runtime import run_pool_phase_a as runner


class PoolPhaseATests(unittest.TestCase):
    def test_shared_runtime_is_configured_for_exact_r8_cohort(self) -> None:
        self.assertEqual(shared.PARENT_COUNT, 24)
        self.assertEqual(shared.FRAME_COUNT, 402)
        self.assertEqual(shared.QUERY_COUNT, 3618)
        self.assertEqual(shared.OUTPUT_ROOT, runner.OUTPUT_ROOT)

    def test_exact_inventory_loads_402_unique_frames_without_faro(self) -> None:
        frames = runner.load_frames(Path(runner.REPO_ROOT) / runner.INVENTORY_PATH)
        self.assertEqual(len(frames), 402)
        self.assertEqual(len({frame.physical_frame_id for frame in frames}), 402)
        self.assertFalse(any("faro" in str(member).lower() for frame in frames for member in frame.members))


if __name__ == "__main__":
    unittest.main()
