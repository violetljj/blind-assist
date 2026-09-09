from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
import unittest

import cv2
import numpy as np

from scripts.research.egomotion_compensated_looming.rcle_synthetic_stress_r1.builder import (
    DEFAULT_SPEC,
    _check_budget,
    _index_base_algorithm_inputs,
    _transform_rgb,
    _validate_frozen_parent_chain,
    _yaw_rotation,
)
from scripts.research.egomotion_compensated_looming.rcle_low_reference_false_trigger_r1.temporal_confirmation import (
    REQUIRED_CONSECUTIVE_PAIRS,
    THRESHOLD,
)


class StressContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = json.loads(DEFAULT_SPEC.read_text(encoding="utf-8"))
        self.conditions = {
            row["condition_id"]: row for row in self.spec["conditions"]
        }
        self.image = np.arange(32 * 48 * 3, dtype=np.uint8).reshape(
            32, 48, 3
        )

    def test_frozen_inventory_and_denominators(self) -> None:
        allocation = self.spec["allocation"]
        self.assertEqual(len(allocation["scene_families"]), 2)
        self.assertEqual(len(allocation["motion_ids"]), 4)
        self.assertEqual(len(self.spec["conditions"]), 7)
        self.assertEqual(allocation["expected_stress_sequences"], 56)
        self.assertEqual(allocation["expected_algorithm_frames"], 5656)
        self.assertEqual(allocation["expected_algorithm_pairs"], 5600)
        self.assertTrue(
            self.spec["parent_r0"][
                "all_development_gates_reused_without_change"
            ]
        )

    def test_control_depth_and_pose_rgb_are_identical(self) -> None:
        for condition_id in (
            "matched_clean_control",
            "depth_noise_gaussian_2pct",
            "pose_yaw_error",
        ):
            transformed = _transform_rgb(
                self.image, self.conditions[condition_id], 17
            )
            self.assertTrue(np.array_equal(transformed, self.image))
            self.assertIsNot(transformed, self.image)

    def test_rgb_stress_transforms_are_deterministic_and_bounded(self) -> None:
        for condition_id in (
            "lighting_low_exposure",
            "low_texture_contrast",
            "motion_blur_horizontal_9",
            "partial_occlusion_35x45",
        ):
            first = _transform_rgb(
                self.image, self.conditions[condition_id], 37
            )
            second = _transform_rgb(
                self.image, self.conditions[condition_id], 37
            )
            self.assertEqual(first.dtype, np.uint8)
            self.assertEqual(first.shape, self.image.shape)
            self.assertTrue(np.array_equal(first, second))
            self.assertFalse(np.array_equal(first, self.image))

    def test_low_texture_reduces_gray_standard_deviation(self) -> None:
        transformed = _transform_rgb(
            self.image, self.conditions["low_texture_contrast"], 0
        )
        before = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        after = cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY)
        self.assertLess(float(np.std(after)), float(np.std(before)) * 0.2)

    def test_yaw_rotation_is_proper_and_nonidentity(self) -> None:
        rotation = _yaw_rotation(0.5)
        self.assertTrue(
            np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        )
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=12)
        self.assertFalse(np.allclose(rotation, np.eye(3)))

    def test_parent_chain_and_implementation_are_current(self) -> None:
        result = _validate_frozen_parent_chain(self.spec)
        self.assertEqual(
            result["scientific_outcome"],
            "SYNTHETIC_MECHANISM_SEALED_REVISE",
        )
        self.assertEqual(
            result["terminal_sha256"],
            self.spec["frozen_parent_chain"]["terminal"]["sha256"],
        )

    def test_parent_chain_hash_tamper_is_fail_closed(self) -> None:
        tampered = json.loads(json.dumps(self.spec))
        tampered["frozen_parent_chain"]["terminal"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(
            ValueError, "PARENT_CHAIN_HASH_MISMATCH:terminal"
        ):
            _validate_frozen_parent_chain(tampered)

    def test_threshold_and_confirmation_constants_are_frozen(self) -> None:
        self.assertEqual(
            self.spec["parent_r0"]["algorithm_threshold_per_s"], THRESHOLD
        )
        self.assertEqual(
            self.spec["parent_r0"]["confirmation_consecutive_pairs"],
            REQUIRED_CONSECUTIVE_PAIRS,
        )

    def test_resource_budget_helpers_fail_closed(self) -> None:
        budget = {
            "runtime_ceiling_seconds": 1.0,
            "disk_ceiling_bytes": 100,
        }
        _check_budget(time.perf_counter(), 100, budget)
        with self.assertRaisesRegex(RuntimeError, "DISK_CEILING"):
            _check_budget(time.perf_counter(), 101, budget)
        with self.assertRaisesRegex(TimeoutError, "RUNTIME_CEILING"):
            _check_budget(time.perf_counter() - 2.0, 0, budget)

    def test_qa_inventory_exceeds_twenty_percent(self) -> None:
        allocation = self.spec["allocation"]
        sampled_per_sequence = len(
            range(0, allocation["frame_count_per_sequence"], 5)
        )
        sampled = (
            allocation["expected_stress_sequences"] * sampled_per_sequence
        )
        self.assertEqual(sampled, 1176)
        self.assertGreaterEqual(
            sampled / allocation["expected_algorithm_frames"], 0.2
        )

    def test_rotation_is_joined_from_strict_base_algorithm_input(self) -> None:
        frame = {"sequence_id": "s", "frame_index": 0}
        algorithm = {
            "sequence_id": "s",
            "frame_index": 0,
            "rotation_current_from_previous": np.eye(3).tolist(),
        }
        indexed = _index_base_algorithm_inputs([frame], [algorithm])
        self.assertEqual(
            indexed[("s", 0)]["rotation_current_from_previous"],
            np.eye(3).tolist(),
        )
        with self.assertRaisesRegex(
            ValueError, "BASE_ALGORITHM_INPUT_IDENTITY"
        ):
            _index_base_algorithm_inputs([frame], [])


if __name__ == "__main__":
    unittest.main()
