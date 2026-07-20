import importlib.util
import tempfile
from pathlib import Path
import unittest


SCRIPT = Path(__file__).with_name("generate_ustrf_synthetic_dynamic_ttc_benchmark.py")
SPEC = importlib.util.spec_from_file_location("dynamic_ttc_benchmark", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class DynamicTtcBenchmarkTest(unittest.TestCase):
    def test_generated_manifest_and_cuda_audit_are_self_consistent(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA required by this benchmark audit")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dynamic-ttc"
            specification = MODULE.generate(root)
            report = MODULE.audit(root, require_cuda=True)
            self.assertEqual(9, specification["sequence_count"])
            self.assertEqual(7, report["admitted_sequence_count"])
            self.assertEqual(2, report["rejected_sequence_count"])
            self.assertEqual(6, report["moving_admitted_count"])
            self.assertEqual(0.0, report["max_velocity_error_mps"])
            self.assertLessEqual(report["max_ttc_error_ms"], 1.0)
            self.assertEqual(1.0, report["collision_label_accuracy"])


if __name__ == "__main__":
    unittest.main()
