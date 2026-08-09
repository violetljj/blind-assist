import unittest

from scripts.research.assistive_geometry.audit_b1_orientation_geometry import (
    audit,
    tensor_hw_for_orientation,
)


class AuditB1OrientationGeometryTest(unittest.TestCase):
    def test_tensor_shape_preserves_orientation_family(self) -> None:
        self.assertEqual((448, 608), tensor_hw_for_orientation(0))
        self.assertEqual((608, 448), tensor_hw_for_orientation(1))
        self.assertEqual((448, 608), tensor_hw_for_orientation(2))
        self.assertEqual((608, 448), tensor_hw_for_orientation(3))

    def test_firewall_fails_before_trajectory_access(self) -> None:
        manifest = {
            "schema": "blindassist_assistive_geometry_b0_arkitscenes_pose_covered_media_manifest_v1",
            "task_outcome_opened": True,
            "model_outputs_read": False,
            "videos": [],
        }
        with self.assertRaisesRegex(ValueError, "task outcome"):
            audit(manifest)


if __name__ == "__main__":
    unittest.main()
