import json
import tempfile
import time
import unittest
from pathlib import Path

from scripts.research.assistive_geometry.train_b1_a0_formal import (
    PROGRESS_FIELDS,
    atomic_write_json,
    effective_batches_to_microbatches,
    write_progress,
)


class FormalA0TrainingTest(unittest.TestCase):
    def test_effective_batch_expands_to_four_ordered_microbatches(self) -> None:
        microbatches, orientations = effective_batches_to_microbatches([
            ("portrait", list(range(16))),
            ("landscape", list(range(16, 32))),
        ])
        self.assertEqual(8, len(microbatches))
        self.assertEqual(list(range(32)), [item for batch in microbatches for item in batch])
        self.assertEqual(["portrait"] * 4 + ["landscape"] * 4, orientations)

    def test_progress_contract_contains_guarded_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.json"
            payload = write_progress(
                path,
                phase="pilot",
                completed=1,
                total=2,
                started_at=time.perf_counter() - 1.0,
                status="running",
            )
            self.assertTrue(set(PROGRESS_FIELDS).issubset(payload))
            self.assertEqual(payload, json.loads(path.read_text(encoding="utf-8")))
            self.assertGreater(payload["throughput"], 0.0)
            self.assertGreater(payload["eta_seconds"], 0.0)

    def test_exclusive_json_rejects_existing_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            atomic_write_json(path, {"status": "first"}, exclusive=True)
            with self.assertRaises(ValueError):
                atomic_write_json(path, {"status": "second"}, exclusive=True)


if __name__ == "__main__":
    unittest.main()
