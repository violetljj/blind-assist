import importlib.util
import json
import tempfile
from pathlib import Path
import unittest

import numpy as np


SCRIPT = Path(__file__).with_name("audit_carla_ped_rgbd_slice.py")
SPEC = importlib.util.spec_from_file_location("carla_rgbd", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class CarlaRgbdSliceAuditTest(unittest.TestCase):
    def test_synced_depth_camera_and_metadata_receipts_are_admitted_source_native(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA required by CARLA RGB-D audit")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for frame_id, timestamp, x in [(0, 0.0, 0.0), (1, .04, .1)]:
                base = f"scene_{frame_id:03d}"
                matrix = np.eye(4); matrix[0, 3] = x
                camera = {"intrinsic": {"K": [[4.0, 0.0, 2.0], [0.0, 4.0, 1.0], [0.0, 0.0, 1.0]]}, "extrinsic": {"c2w": matrix.tolist(), "w2c": np.linalg.inv(matrix).tolist()}, "frame_id": frame_id, "timestamp": timestamp}
                metadata = {"frame_id": frame_id, "config": {"capture": {"fps": 50, "frame_skip": 2}}}
                (root / f"{base}.camera.json").write_text(json.dumps(camera), encoding="utf-8")
                (root / f"{base}.metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
                np.save(root / f"{base}.depth.npy", np.full((2, 4), 3.0, dtype=np.float32))
                (root / f"{base}.rgb.png").write_bytes(b"fixture")
            report = MODULE.audit(root)
            self.assertTrue(report["ok"])
            self.assertEqual(2, report["frame_count"])
            self.assertAlmostEqual(.04, report["timestamp"]["median_interval_seconds"])
            self.assertTrue(report["source_rgbd_pose_sequence_admitted"])
            self.assertFalse(report["ustrf_metric_geometry_input_admitted"])


if __name__ == "__main__":
    unittest.main()
