from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_ag_st_stage0a import (  # noqa: E402
    compute_selective_metrics,
    estimate_observed_anchor_scale,
    make_withheld_pattern,
    resolve_trajectory_path,
    select_source_videos,
    select_train_videos,
    split_observed_and_hidden_depth,
)


class AgStStage0ATest(unittest.TestCase):
    def test_hidden_depth_is_zero_in_teacher_input(self) -> None:
        depth = np.arange(1, 65, dtype=np.float32).reshape(8, 8) / 10.0
        valid = np.ones_like(depth, dtype=np.bool_)
        pattern = make_withheld_pattern(depth.shape, block_size=2, modulus=4, residue=1)
        observed, observed_valid, hidden = split_observed_and_hidden_depth(depth, valid, pattern)
        self.assertGreater(int(hidden.sum()), 0)
        self.assertFalse(np.any(observed[hidden] > 0))
        self.assertFalse(np.any(observed_valid & hidden))
        np.testing.assert_array_equal(observed[observed_valid], depth[observed_valid])

    def test_mask_is_value_independent_and_contiguous(self) -> None:
        first = make_withheld_pattern((8, 8), block_size=2, modulus=4, residue=0)
        second = make_withheld_pattern((8, 8), block_size=2, modulus=4, residue=0)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all(first[:2, :2]))
        self.assertFalse(np.any(first[:2, 2:4]))

    def test_only_requested_train_parents_are_selected(self) -> None:
        manifest = {
            "videos": [
                {"video_id": "a", "role": "TRAIN"},
                {"video_id": "b", "role": "TRAIN"},
                {"video_id": "c", "role": "DEVELOPMENT"},
            ]
        }
        self.assertEqual(["b", "a"], [row["video_id"] for row in select_train_videos(manifest, ["b", "a"])])
        with self.assertRaisesRegex(ValueError, "manifest role TRAIN"):
            select_train_videos(manifest, ["c"])

    def test_source_role_token_preserves_external_train_role(self) -> None:
        manifest = {
            "videos": [
                {"video_id": "a", "role": "train"},
                {"video_id": "b", "role": "validation"},
            ]
        }
        selected = select_source_videos(
            manifest,
            ["a"],
            role_token="train",
        )
        self.assertEqual(["a"], [row["video_id"] for row in selected])
        with self.assertRaisesRegex(ValueError, "manifest role train"):
            select_source_videos(manifest, ["b"], role_token="train")

    def test_trajectory_resolves_from_scoped_media_asset_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trajectory = root / "lowres_wide.traj"
            payload = b"0 0 0 0 0 0 0\n"
            trajectory.write_bytes(payload)
            video = {
                "extracted": {
                    "lowres_wide": [
                        {"path": str(root / "lowres_wide" / "frame.png")}
                    ]
                },
                "source_assets": [
                    {
                        "asset": "lowres_wide.traj",
                        "content_length_bytes": len(payload),
                        "archive_sha256": hashlib.sha256(payload).hexdigest().upper(),
                    }
                ],
            }
            self.assertEqual(trajectory, resolve_trajectory_path(video))

    def test_observed_anchor_scale_corrects_metric_bias(self) -> None:
        observed = np.asarray([[1.0, 2.0], [0.0, 4.0]], dtype=np.float32)
        raw_prediction = np.asarray([[1.1, 2.2], [9.0, 4.4]], dtype=np.float32)
        scale, support = estimate_observed_anchor_scale(
            observed, raw_prediction, minimum_support=3
        )
        self.assertEqual(3, support)
        self.assertAlmostEqual(1.0 / 1.1, scale, places=6)

    def test_confidence_curve_exposes_lower_risk_at_high_confidence(self) -> None:
        truth = np.ones((2, 4), dtype=np.float32)
        prediction = np.asarray([[1.01, 1.02, 1.20, 1.30], [1.01, 1.02, 1.20, 1.30]], dtype=np.float32)
        confidence = np.asarray([[10.0, 9.0, 2.0, 1.0], [10.0, 9.0, 2.0, 1.0]], dtype=np.float32)
        records = []
        for parent in ("p0", "p1"):
            records.append(
                {
                    "parent_id": parent,
                    "truth_depth_m": truth,
                    "prediction_depth_m": prediction,
                    "confidence": confidence,
                    "hidden_mask": np.ones_like(truth, dtype=np.bool_),
                    "model_mask": np.ones_like(truth, dtype=np.bool_),
                    "baseline_depth_m": np.full_like(truth, 1.05),
                }
            )
        result = compute_selective_metrics(records)
        curve = result["teacher_confidence_risk_coverage"]
        self.assertEqual(16, result["hidden_pixel_count"])
        self.assertGreater(curve[0]["overall"]["mae_m"], curve[-1]["overall"]["mae_m"])
        self.assertGreater(curve[0]["coverage_of_hidden"], curve[-1]["coverage_of_hidden"])
        self.assertTrue(all(row["parent_macro_evaluable"] for row in curve))


if __name__ == "__main__":
    unittest.main()
