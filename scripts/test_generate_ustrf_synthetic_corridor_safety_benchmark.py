import importlib.util
import csv
import csv
import tempfile
from pathlib import Path
import unittest


SCRIPT = Path(__file__).with_name("generate_ustrf_synthetic_corridor_safety_benchmark.py")
SPEC = importlib.util.spec_from_file_location("corridor_benchmark", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class CorridorBenchmarkTest(unittest.TestCase):
    def test_gpu_generated_truth_has_explicit_body_ground_and_event_labels(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA required by synthetic corridor benchmark")
        with tempfile.TemporaryDirectory() as temporary:
            report = MODULE.generate(Path(temporary) / "benchmark")
            self.assertEqual(256, report["scene_count"])
            self.assertGreater(report["critical_scene_count"], 0)
            self.assertEqual(0, report["expected_clear_stop_count"])
            self.assertTrue(report["body_frame_ground_truth"])
            self.assertTrue(report["local_ground_truth"])
            self.assertTrue(report["dynamic_event_truth"])
            with (Path(temporary) / "benchmark" / "kotlin_corridor_safety_replay.tsv").open(encoding="utf-8", newline="") as source:
                rows = list(csv.DictReader(source, delimiter="\t"))
            self.assertTrue(all(row["expected_selected_offset"] == "-2" for row in rows if row["family"] == "central_unknown"))
            self.assertEqual("-1", next(row["expected_selected_offset"] for row in rows if row["hazards"] == "O:1:3"))
            with (Path(temporary) / "benchmark" / "kotlin_corridor_safety_replay.tsv").open(encoding="utf-8", newline="") as source:
                rows = list(csv.DictReader(source, delimiter="\t"))
            self.assertTrue(all(row["expected_selected_offset"] == "-2" for row in rows if row["family"] == "central_unknown"))
            self.assertEqual("-1", next(row["expected_selected_offset"] for row in rows if row["hazards"] == "O:1:3"))


if __name__ == "__main__":
    unittest.main()
