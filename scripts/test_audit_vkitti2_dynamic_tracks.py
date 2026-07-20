import importlib.util
import tempfile
from pathlib import Path
import unittest


SCRIPT = Path(__file__).with_name("audit_vkitti2_dynamic_tracks.py")
SPEC = importlib.util.spec_from_file_location("vkitti_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class VkittiDynamicTrackAuditTest(unittest.TestCase):
    def test_static_and_moving_track_labels_are_separated_after_ego_compensation(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA required by VKITTI audit")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "pose.txt").write_text(
                "frame cameraID trackID alpha width height length world_space_X world_space_Y world_space_Z rotation_world_space_y rotation_world_space_x rotation_world_space_z camera_space_X camera_space_Y camera_space_Z rotation_camera_space_y rotation_camera_space_x rotation_camera_space_z\n"
                "0 0 1 0 1 1 1 0 0 0 0 0 0 0 0 5 0 0 0\n"
                "0 0 2 0 1 1 1 0 0 0 0 0 0 1 0 5 0 0 0\n"
                "1 0 1 0 1 1 1 0 0 0 0 0 0 1 0 5 0 0 0\n"
                "1 0 2 0 1 1 1 0 0 0 0 0 0 2.5 0 5 0 0 0\n", encoding="utf-8")
            (root / "extrinsic.txt").write_text(
                "frame cameraID r1,1 r1,2 r1,3 t1 r2,1 r2,2 r2,3 t2 r3,1 r3,2 r3,3 t3 0 0 0 1\n"
                "0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1\n"
                "1 0 1 0 0 1 0 1 0 0 0 0 1 0 0 0 0 1\n", encoding="utf-8")
            (root / "bbox.txt").write_text(
                "frame cameraID trackID left right top bottom number_pixels truncation_ratio occupancy_ratio isMoving\n"
                "0 0 1 0 1 0 1 1 0 0 False\n0 0 2 0 1 0 1 1 0 0 True\n"
                "1 0 1 0 1 0 1 1 0 0 False\n1 0 2 0 1 0 1 1 0 0 True\n", encoding="utf-8")
            report = MODULE.audit(root)
            self.assertEqual(2, report["consecutive_track_pair_count"])
            self.assertEqual(1, report["source_moving_pair_count"])
            self.assertEqual(1.0, report["classification"]["precision"])
            self.assertEqual(1.0, report["classification"]["recall"])
            self.assertFalse(report["physical_ttc_seconds_admitted"])


if __name__ == "__main__":
    unittest.main()
