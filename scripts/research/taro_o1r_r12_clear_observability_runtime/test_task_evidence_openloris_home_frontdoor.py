from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_openloris_home_frontdoor as subject


class OpenLorisHomeFrontdoorTest(unittest.TestCase):
    def test_nearest_pose_accepts_xyzw_and_enforces_delta(self) -> None:
        rows = [
            ["10.0", "1", "2", "3", "0", "0", "0", "1"],
            ["10.2", "4", "5", "6", "0", "0", "0", "1"],
        ]
        accepted = subject._nearest_base_pose(rows, [10.0, 10.2], 10.04)
        self.assertIsNotNone(accepted)
        pose, delta = accepted or (None, None)
        np.testing.assert_allclose(pose[:3, 3], [1.0, 2.0, 3.0])
        self.assertAlmostEqual(delta, 0.04)
        self.assertIsNone(subject._nearest_base_pose(rows, [10.0, 10.2], 10.31))

    def test_calibration_crop_preserves_focal_length_and_shifts_principal_point(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sensors = cv2.FileStorage(str(root / "sensors.yaml"), cv2.FILE_STORAGE_WRITE)
            sensors.startWriteStruct("d400_color_optical_frame", cv2.FileNode_MAP)
            sensors.write("intrinsics", np.asarray([[600.0, 424.0, 601.0, 240.0]], dtype=np.float64))
            sensors.endWriteStruct()
            sensors.release()
            transforms = cv2.FileStorage(str(root / "trans_matrix.yaml"), cv2.FILE_STORAGE_WRITE)
            transforms.startWriteStruct("trans_matrix", cv2.FileNode_SEQ)
            transforms.startWriteStruct("", cv2.FileNode_MAP)
            transforms.write("parent_frame", "base_link")
            transforms.write("child_frame", "d400_color_optical_frame")
            transforms.write("matrix", np.eye(4, dtype=np.float64))
            transforms.endWriteStruct()
            transforms.endWriteStruct()
            transforms.release()
            intrinsics, base_to_color, _receipt = subject._opencv_calibration(root)
        np.testing.assert_allclose(intrinsics, [[600.0, 0.0, 320.0], [0.0, 601.0, 240.0], [0.0, 0.0, 1.0]])
        np.testing.assert_allclose(base_to_color, np.eye(4))

    def test_safe_child_rejects_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            self.assertEqual(subject._safe_child(root, "rgb/1.png"), root / "rgb" / "1.png")
            with self.assertRaises(subject.OpenLorisFrontdoorError):
                subject._safe_child(root, "../escape.png")

    def test_groundtruth_deduplicates_bounded_same_timestamp_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "groundtruth.txt"
            path.write_text(
                "1.0 0 0 0 0 0 0 1\n"
                "1.0 0 0 0 0 0 0 1\n"
                "2.0 1 0 0 0 0 0 1\n",
                encoding="utf-8",
            )
            rows, timestamps, duplicates = subject._parse_groundtruth(path)
            self.assertEqual([1.0, 2.0], timestamps)
            self.assertEqual(2, len(rows))
            self.assertEqual(1, duplicates["identical_extra_row_count"])
            self.assertEqual(0, duplicates["near_identical_extra_row_count"])
            path.write_text(
                "1.0 0 0 0 0 0 0 1\n"
                "1.0 0.0005 0 0 0 0 0 1\n"
                "2.0 1 0 0 0 0 0 1\n",
                encoding="utf-8",
            )
            rows, timestamps, duplicates = subject._parse_groundtruth(path)
            self.assertEqual([1.0, 2.0], timestamps)
            self.assertEqual("0", rows[0][1])
            self.assertEqual(1, duplicates["near_identical_extra_row_count"])
            self.assertAlmostEqual(0.0005, duplicates["maximum_duplicate_translation_m"])
            path.write_text(
                "1.0 0 0 0 0 0 0 1\n"
                "1.0 1 0 0 0 0 0 1\n",
                encoding="utf-8",
            )
            with self.assertRaises(subject.OpenLorisFrontdoorError):
                subject._parse_groundtruth(path)

    def test_candidate_identity_is_deterministic(self) -> None:
        pose = np.eye(4, dtype=np.float64)
        reference = subject.bonn.Frame("p", 1.0, Path("r"), Path("r"), pose)
        neighbor = subject.bonn.Frame("p", 0.5, Path("n"), Path("n"), pose.copy())
        neighbor.camera_to_world[0, 3] = 0.06
        pair = subject.bonn._pair(reference, neighbor)
        support = subject.bonn.ReferenceSupport(reference, (pair,), (pair,))
        _first, first_sha = subject._candidate_identity([support])
        _second, second_sha = subject._candidate_identity([support])
        self.assertEqual(first_sha, second_sha)

    def test_candidate_identity_rejects_reference_payload_overlap(self) -> None:
        pose = np.eye(4, dtype=np.float64)
        older = subject.bonn.Frame("p", 0.5, Path("a"), Path("a"), pose.copy())
        newer_pose = pose.copy()
        newer_pose[0, 3] = 0.06
        newer = subject.bonn.Frame("p", 1.0, Path("b"), Path("b"), newer_pose)
        later_pose = newer_pose.copy()
        later_pose[0, 3] = 0.12
        later = subject.bonn.Frame("p", 1.5, Path("c"), Path("c"), later_pose)
        first = subject.bonn._pair(newer, older)
        second = subject.bonn._pair(later, newer)
        supports = [
            subject.bonn.ReferenceSupport(newer, (first,), (first,)),
            subject.bonn.ReferenceSupport(later, (second,), (second,)),
        ]
        with self.assertRaises(subject.OpenLorisFrontdoorError):
            subject._candidate_identity(supports)


if __name__ == "__main__":
    unittest.main()
