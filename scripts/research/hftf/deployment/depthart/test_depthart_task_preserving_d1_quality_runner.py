import hashlib
import json
import unittest
from pathlib import Path

from scripts.research.hftf.deployment.depthart.finalize_depthart_task_preserving_d1_quality import (
    chunk_schedule as finalizer_schedule,
)
from scripts.research.hftf.deployment.depthart.run_depthart_task_preserving_d1_quality_chunk import (
    chunk_schedule as runner_schedule,
)


REPO_ROOT = Path(__file__).resolve().parents[5]
PROTOCOL_PATH = REPO_ROOT / "docs/research/hftf/DEPTHART_TASK_PRESERVING_D1_QUALITY_SCREEN_PROTOCOL_2026-08-10.json"
REPAIR_PATH = REPO_ROOT / "docs/research/hftf/DEPTHART_TASK_PRESERVING_D1_QUALITY_SCREEN_RUNNER_REPAIR_2026-08-10.json"
RUNNER_PATH = REPO_ROOT / "scripts/research/hftf/deployment/depthart/run_depthart_task_preserving_d1_quality_chunk.py"


class D1QualityRunnerTest(unittest.TestCase):
    def test_schedule_is_exact_and_shared(self):
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        schedule = runner_schedule(protocol)
        self.assertEqual(schedule, finalizer_schedule(protocol))
        self.assertEqual(len(schedule), 48)
        self.assertEqual(schedule[0]["frame_start"], 0)
        self.assertEqual(schedule[-1]["frame_stop"], 300)
        self.assertTrue(all(row["frame_stop"] - row["frame_start"] == 50 for row in schedule))

    def test_repair_binds_current_runner_and_only_separator_change(self):
        repair = json.loads(REPAIR_PATH.read_text(encoding="utf-8"))
        digest = hashlib.sha256(RUNNER_PATH.read_bytes()).hexdigest().upper()
        self.assertEqual(repair["repaired_runner_sha256"], digest)
        self.assertFalse(repair["candidate_data_policy_or_gate_changed"])
        source = RUNNER_PATH.read_text(encoding="utf-8")
        self.assertIn("ADSP_LIBRARY_PATH='", source)
        self.assertIn(";/system/lib/rfsa/adsp;", source)


if __name__ == "__main__":
    unittest.main()
