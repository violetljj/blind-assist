from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.research.satom_r0.core import (
    ArmConfig,
    PolarEvidenceMemory,
    TofConfig,
    _metric_rows_for_parent,
    default_arms,
    evaluate_frames,
    make_synthetic_frames,
)
from scripts.research.satom_r0.bonn import estimate_camera_height_m
from scripts.research.satom_r0.run_satom_r0 import load_manifest


class SatomR0Test(unittest.TestCase):
    def test_all_required_comparators_and_controls_exist(self) -> None:
        names = {arm.name for arm in default_arms()}
        self.assertTrue(
            {
                "single_frame_depthart",
                "uniform_multiframe_fusion",
                "tof_only_round_robin",
                "satom_center_only",
                "satom_random",
                "satom_round_robin",
                "satom_max_entropy",
                "satom_task_weighted_information_gain",
                "satom_task_weighted_information_gain_shuffled_timestamp",
                "satom_task_weighted_information_gain_wrong_extrinsic",
                "satom_task_weighted_information_gain_wrong_roi",
            }.issubset(names)
        )

    def test_prefix_is_invariant_to_future_truth(self) -> None:
        frames = make_synthetic_frames(parent_count=1, frames_per_parent=10)
        arm = ArmConfig("candidate", "task_weighted_information_gain", True, True, True)
        original = _metric_rows_for_parent(frames, arm, TofConfig(missing_probability=0.0))
        changed = list(frames)
        tail = changed[-1]
        changed[-1] = type(tail)(
            **{
                **tail.__dict__,
                "truth_depth_m": np.full_like(tail.truth_depth_m, 0.35),
            }
        )
        perturbed = _metric_rows_for_parent(changed, arm, TofConfig(missing_probability=0.0))
        prefix_original = [row for row in original if row["frame_index"] < 9]
        prefix_perturbed = [row for row in perturbed if row["frame_index"] < 9]
        self.assertEqual(prefix_original, prefix_perturbed)

    def test_pose_warp_moves_evidence_forward(self) -> None:
        memory = PolarEvidenceMemory(range_step_m=0.2, max_range_m=4.0, decay=1.0)
        pose0 = np.eye(4)
        memory.begin_frame(pose0)
        memory.update_ray(1, 1.2, free_weight=0.2, occupied_weight=1.0)
        before = max(
            index for index in range(memory.range_bins) if memory.cells[1, index].occupied > 1.5
        )
        pose1 = np.eye(4)
        pose1[2, 3] = 0.2
        memory.begin_frame(pose1)
        after = max(
            index for index in range(memory.range_bins) if memory.cells[1, index].occupied > 1.5
        )
        self.assertLess(after, before)

    def test_evaluator_reports_pooled_macro_worst_and_negative_controls(self) -> None:
        result = evaluate_frames(
            make_synthetic_frames(parent_count=3, frames_per_parent=12),
            evidence_role="SYNTHETIC_MECHANICS_CANARY",
            prior_provenance={
                "family": "SYNTHETIC_DEPTHART_LIKE_PRIOR",
                "frozen": True,
                "truth_derived": True,
            },
        )
        self.assertFalse(result["causality"]["policy_truth_access"])
        self.assertFalse(result["causality"]["complete_parent_future_distribution_access"])
        for arm in result["arms"].values():
            self.assertEqual(set(arm).intersection({"pooled", "parent_macro", "worst_parent"}), {"pooled", "parent_macro", "worst_parent"})
            self.assertIn("false_clear", arm["parent_macro"])
            self.assertIn("false_block", arm["parent_macro"])
            self.assertIn("coverage", arm["parent_macro"])
            self.assertIn("clearance_mae_m", arm["parent_macro"])
            self.assertIn("calibration_error", arm["parent_macro"])
            self.assertEqual(arm["matched_coverage"]["targets"], [0.5, 0.6, 0.7, 0.8, 0.9])
            self.assertIn("0.70", arm["matched_coverage"]["across_parents"])
        self.assertEqual(
            result["pareto_diagnostic"]["metrics"],
            ["false_clear:min", "false_block:min", "coverage:max"],
        )
        self.assertIn("satom_round_robin", result["pareto_diagnostic"]["parent_macro"])

    def test_non_unit_gravity_is_rejected(self) -> None:
        frame = make_synthetic_frames(parent_count=1, frames_per_parent=1)[0]
        invalid = type(frame)(**{**frame.__dict__, "gravity_down_camera": np.array([0.0, 2.0, 0.0])})
        with self.assertRaisesRegex(ValueError, "unit length"):
            invalid.validate()

    def test_real_e0_rejects_confirmation_role(self) -> None:
        manifest = {
            "schema": "blindassist.satom_r0.dataset_manifest.v1",
            "evidence_role": "FRESH_CONFIRMATION",
            "prior_provenance": {
                "family": "DepthART",
                "frozen": True,
                "truth_derived": False,
            },
            "parents": [{"parent_id": "not-opened", "bundle": "missing.npz", "sha256": "00"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Development role"):
                load_manifest(path)

    def test_camera_height_estimation_is_source_specific(self) -> None:
        prior = np.full((32, 48), 1.2, dtype=np.float32)
        truth = np.full((32, 48), 1.6, dtype=np.float32)
        intrinsics = np.asarray([[40.0, 0.0, 23.5], [0.0, 40.0, 15.5], [0.0, 0.0, 1.0]])
        gravity = np.asarray([0.0, 0.0, 1.0])
        policy = (0.5, 2.5, 0.04, 0.08, 20, 0.02)
        prior_height = estimate_camera_height_m(prior, intrinsics, gravity, *policy)
        truth_height = estimate_camera_height_m(truth, intrinsics, gravity, *policy)
        self.assertAlmostEqual(prior_height, 1.2, places=5)
        self.assertAlmostEqual(truth_height, 1.6, places=5)


if __name__ == "__main__":
    unittest.main()
