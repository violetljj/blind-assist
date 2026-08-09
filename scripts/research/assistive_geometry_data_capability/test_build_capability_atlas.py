from __future__ import annotations

import unittest

import numpy as np

from scripts.research.assistive_geometry_data_capability.build_capability_atlas import (
    CAPABILITY_NAMES,
    EXPECTED_TARGET_KEYS,
    aggregate_capabilities,
    evaluate_hypothesis,
    temporal_pair_flags,
    validate_target_contract,
)


class AssistiveGeometryDataCapabilityTest(unittest.TestCase):
    def test_target_contract_rejects_extra_keys_after_exact_schema_pass(self) -> None:
        class Loaded(dict[str, np.ndarray]):
            @property
            def files(self) -> list[str]:
                return list(self)

        source_hw = (192, 256)
        payload = Loaded(
            {
                "band_confidence_valid": np.ones(3, dtype=np.bool_),
                "camera_height_m": np.asarray(1.5, dtype=np.float32),
                "clearance_m": np.zeros(3, dtype=np.float32),
                "clearance_valid": np.ones(3, dtype=np.bool_),
                "depth_m_source": np.ones(source_hw, dtype=np.float32),
                "depth_valid_source": np.ones(source_hw, dtype=np.bool_),
                "ground_label_valid_source": np.ones(source_hw, dtype=np.bool_),
                "ground_plane_valid": np.asarray(True, dtype=np.bool_),
                "ground_probability_source": np.ones(source_hw, dtype=np.float32),
                "intrinsics_source": np.eye(3, dtype=np.float32),
                "intrinsics_tensor": np.eye(3, dtype=np.float32),
                "occupancy": np.zeros((3, 3), dtype=np.float32),
                "occupancy_valid": np.ones((3, 3), dtype=np.bool_),
                "orientation_index": np.asarray(0, dtype=np.int8),
                "target_hw": np.asarray([448, 608], dtype=np.int32),
                "up_camera": np.asarray([0.0, -1.0, 0.0], dtype=np.float32),
            }
        )
        self.assertEqual(set(EXPECTED_TARGET_KEYS), set(payload))
        frame = {
            "source_hw": list(source_hw),
            "target_hw": [448, 608],
            "orientation_index": 0,
            "ground_plane_valid": True,
        }
        validate_target_contract(payload, frame, "p/f")
        payload["unexpected"] = np.asarray(0, dtype=np.int8)
        with self.assertRaisesRegex(ValueError, "target keyset drift"):
            validate_target_contract(payload, frame, "p/f")

    def test_temporal_pair_requires_same_parent_order_and_small_positive_gap(self) -> None:
        frames = [
            {"video_id": "p0", "frame_index": 0, "frame_stem": "p0_1.000"},
            {"video_id": "p0", "frame_index": 1, "frame_stem": "p0_1.016"},
            {"video_id": "p0", "frame_index": 2, "frame_stem": "p0_1.500"},
        ]
        self.assertEqual(
            {"p0/p0_1.000": False, "p0/p0_1.016": True, "p0/p0_1.500": False},
            temporal_pair_flags(frames, 0.1),
        )

    def test_aggregate_is_parent_and_orientation_explicit(self) -> None:
        blank = {name: False for name in CAPABILITY_NAMES}
        rows = [
            {"video_id": "p0", "orientation_family": "portrait", "capabilities": {**blank, "right_censor": True}},
            {"video_id": "p0", "orientation_family": "portrait", "capabilities": {**blank, "right_censor": False}},
            {"video_id": "p1", "orientation_family": "landscape", "capabilities": {**blank, "right_censor": True}},
        ]
        result = aggregate_capabilities(rows, ["p0", "p1"])["right_censor"]
        self.assertEqual(2, result["frame_count"])
        self.assertEqual({"p0": 1, "p1": 1}, result["parent_frame_counts"])
        self.assertEqual({"portrait": 1, "landscape": 1}, result["orientation_frame_counts"])

    def test_hypothesis_fails_when_support_is_not_parent_disjoint(self) -> None:
        atlas = {
            "parent_order": ["p0", "p1", "p2", "p3"],
            "capabilities": {
                "event": {
                    "frame_count": 40,
                    "parent_frame_counts": {"p0": 10, "p1": 10, "p2": 10, "p3": 10},
                    "orientation_frame_counts": {"portrait": 20, "landscape": 20},
                },
                "censor": {
                    "frame_count": 10,
                    "parent_frame_counts": {"p0": 10, "p1": 0, "p2": 0, "p3": 0},
                    "orientation_frame_counts": {"portrait": 10, "landscape": 0},
                },
            },
        }
        contract = {
            "capabilities": {
                "event": {"minimum_total_frames": 1, "minimum_frames_per_parent": 1, "minimum_parents": 4},
                "censor": {"minimum_total_frames": 1, "minimum_frames_per_parent": 1, "minimum_parents": 4},
            },
            "joint_parent_gate": {"minimum_joint_parents": 4, "minimum_fit_parents": 2, "minimum_eval_parents": 2},
            "authority_requirements": {},
        }
        result = evaluate_hypothesis(atlas, contract, {})
        self.assertEqual("NOT_SUPPORTED_DATA", result["terminal"])
        self.assertEqual(["p0"], result["joint_eligible_parents"])

    def test_authority_failure_is_separate_from_data_support(self) -> None:
        atlas = {
            "parent_order": ["p0", "p1"],
            "capabilities": {
                "factor": {
                    "frame_count": 20,
                    "parent_frame_counts": {"p0": 10, "p1": 10},
                    "orientation_frame_counts": {"portrait": 10, "landscape": 10},
                }
            },
        }
        contract = {
            "capabilities": {
                "factor": {"minimum_total_frames": 10, "minimum_frames_per_parent": 5, "minimum_parents": 2}
            },
            "joint_parent_gate": {"minimum_joint_parents": 2, "minimum_fit_parents": 1, "minimum_eval_parents": 1},
            "authority_requirements": {"fresh_outcome": True},
        }
        result = evaluate_hypothesis(atlas, contract, {"fresh_outcome": False})
        self.assertTrue(result["data_pass"])
        self.assertFalse(result["authority_pass"])
        self.assertEqual("NOT_SUPPORTED_AUTHORITY", result["terminal"])


if __name__ == "__main__":
    unittest.main()
