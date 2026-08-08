import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.research.hftf.deployment.depthart.evaluate_depthart_g4d_repair import evaluate


class EvaluateDepthArtG4dRepairTest(unittest.TestCase):
    def test_three_way_gate_preserves_negative_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arrays = {
                "pytorch": np.asarray([1.0, 2.0], np.float32),
                "canonical_onnx_output": np.asarray([1.0, 2.00001], np.float32),
                "htp_direct": np.asarray([1.0, 2.1], np.float32),
                "htp_context": np.asarray([1.0, 2.1], np.float32),
                "frontier_ort": np.asarray([1.0, 2.0], np.float32),
                "frontier_htp": np.asarray([1.0, 2.01], np.float32),
            }
            paths = {}
            for name, value in arrays.items():
                paths[name] = root / f"{name}.raw"
                value.tofile(paths[name])
            asset_names = (
                "frozen_input", "deployment_onnx", "dlc", "context",
                "op_package_arm64", "op_package_dsp",
            )
            for name in asset_names:
                paths[name] = root / name
                paths[name].write_bytes(name.encode())
            args = argparse.Namespace(
                **paths,
                frontier_name="/patch/Conv2",
                serial="SERIAL",
            )
            receipt = evaluate(args)
            self.assertEqual(receipt["status"], "G4_D_FAIL")
            self.assertEqual(receipt["gates"]["pytorch_vs_canonical_onnx"], "PASS")
            self.assertEqual(receipt["gates"]["canonical_onnx_vs_sm8650_htp"], "FAIL")
            self.assertEqual(receipt["gates"]["dlc_direct_vs_saved_context"], "PASS")
            self.assertIn("STRICT_G4D_NOT_SUPPORTED", receipt["terminal"])
            self.assertEqual(receipt["downstream"]["G4_E_partition_purity"], "NOT_EVALUATED")


if __name__ == "__main__":
    unittest.main()
