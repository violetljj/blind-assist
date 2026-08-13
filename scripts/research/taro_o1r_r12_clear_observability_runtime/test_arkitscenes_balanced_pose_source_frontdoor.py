import unittest

import numpy as np

from scripts.research.taro_o1r_r12_clear_observability_runtime import arkitscenes_balanced_pose_source_frontdoor as subject


class ARKitScenesBalancedPoseSourceFrontdoorTest(unittest.TestCase):
    def test_landscape_pose_is_canonicalized_without_resampling_shape(self) -> None:
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
        result = subject.canonical_landscape_pose(pose)
        self.assertIsNotNone(result)
        self.assertIn(result[1], subject.ALLOWED_ORIENTATION_INDICES)

    def test_portrait_pose_is_rejected_before_payload_read(self) -> None:
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = np.asarray([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
        result = subject.canonical_landscape_pose(pose)
        self.assertIsNone(result)

    def test_unknown_confidence_never_becomes_static_support(self) -> None:
        depth = np.full((3, 3), 1.0, dtype=np.float64)
        confidence = np.zeros((3, 3), dtype=np.uint8)
        intrinsics = np.asarray([[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]])
        truth, _points, static = subject.observation_geometry(depth, confidence, intrinsics)
        self.assertFalse(np.any(truth))
        self.assertFalse(np.any(static))


if __name__ == "__main__":
    unittest.main()
