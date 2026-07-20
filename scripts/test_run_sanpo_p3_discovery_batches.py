from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parent
TEST_ARTIFACTS = SCRIPTS.parent / "artifacts.local" / "tests" / "sanpo_p3_discovery_batches"
SPEC = importlib.util.spec_from_file_location("batched_discovery", SCRIPTS / "run_sanpo_p3_discovery_batches.py")
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


class BatchedDiscoveryTest(unittest.TestCase):
    def args(self, root: Path, resume: bool = False) -> argparse.Namespace:
        return argparse.Namespace(
            split="train", start_session_index=0, max_sessions=4, batch_size=2, sample_count=6, minimum_hits=2,
            camera="auto", profiles=["center_obstacle"], labels=["pedestrian"], local_lateral_frame_count=16,
            local_lateral_min_target_frames=8, local_lateral_min_target_run=8, local_lateral_min_path_frames=13,
            target_fps=10.0, retries=3, resume=resume, aggregate_output=root / "aggregate.json",
        )

    @staticmethod
    def write_batch(command: list[str]) -> int:
        output = Path(command[command.index("--output") + 1])
        start = int(command[command.index("--start-session-index") + 1])
        count = int(command[command.index("--max-sessions") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"scan_coverage": [{"requested_start_session_index": start, "selected_session_count": count, "attempted_session_count": count, "network_or_data_failure_count": 0}], "candidates": [], "local_lateral_prefilter_rejections": []}), encoding="utf-8")
        return 0

    def test_resume_binds_contract_and_skips_completed_batch(self) -> None:
        TEST_ARTIFACTS.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEST_ARTIFACTS) as temp:
            root = Path(temp)
            checkpoint = root / "checkpoint.json"
            calls: list[list[str]] = []
            def first(command: list[str]) -> int:
                calls.append(command)
                if len(calls) == 2:
                    return 1
                return self.write_batch(command)
            with self.assertRaisesRegex(RuntimeError, "exited 1"):
                runner.run_batches(self.args(root), ["a", "b", "c", "d"], checkpoint, root / "batches", first)
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
            self.assertEqual(["0"], sorted(saved["completed_batches"]))
            resumed_calls: list[list[str]] = []
            result = runner.run_batches(self.args(root, resume=True), ["a", "b", "c", "d"], checkpoint, root / "batches", lambda command: (resumed_calls.append(command), self.write_batch(command))[1])
            self.assertEqual(1, len(resumed_calls))
            self.assertEqual(2, len(result["batches"]))
            self.assertTrue(self.args(root).aggregate_output.is_file())

    def test_contract_change_refuses_resume_before_invocation(self) -> None:
        TEST_ARTIFACTS.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEST_ARTIFACTS) as temp:
            root = Path(temp)
            checkpoint = root / "checkpoint.json"
            args = self.args(root)
            runner.atomic_write_json(checkpoint, runner.new_checkpoint(runner.contract_from_args(args, ["a", "b", "c", "d"])))
            changed = self.args(root, resume=True)
            changed.sample_count = 7
            with self.assertRaisesRegex(ValueError, "contract"):
                runner.run_batches(changed, ["a", "b", "c", "d"], checkpoint, root / "batches", lambda _: self.fail("must not invoke"))

    def test_clean_batch_limit_leaves_resumable_checkpoint(self) -> None:
        TEST_ARTIFACTS.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEST_ARTIFACTS) as temp:
            root = Path(temp)
            checkpoint = root / "checkpoint.json"
            calls: list[list[str]] = []
            result = runner.run_batches(
                self.args(root), ["a", "b", "c", "d"], checkpoint, root / "batches",
                lambda command: (calls.append(command), self.write_batch(command))[1],
                max_batches_per_invocation=1,
            )
            self.assertIsNone(result)
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
            self.assertFalse(saved["complete"])
            self.assertEqual(["0"], sorted(saved["completed_batches"]))
            self.assertEqual(1, len(calls))


if __name__ == "__main__":
    unittest.main()
