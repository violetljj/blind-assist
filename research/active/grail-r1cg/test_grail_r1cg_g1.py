from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from collect_grail_r1cg_g1 import _side_position
from evaluate_grail_r1cg_g1 import evaluate, sha256_file


class G1GeometryTest(unittest.TestCase):
    def test_side_positions_are_anchor_frame_lateral(self) -> None:
        anchor = {"x": 0.0, "y": 0.0, "z": 0.0}
        center = {"x": 0.0, "z": 2.0}
        reachable = [
            anchor,
            {"x": -0.25, "y": 0.0, "z": 0.0},
            {"x": 0.25, "y": 0.0, "z": 0.0},
            {"x": 0.0, "y": 0.0, "z": 0.25},
        ]
        left = _side_position(anchor, reachable, center, -1, "group")
        right = _side_position(anchor, reachable, center, 1, "group")
        self.assertIsNotNone(left)
        self.assertIsNotNone(right)
        assert left is not None and right is not None
        self.assertAlmostEqual(left[1], -0.25)
        self.assertAlmostEqual(right[1], 0.25)
        self.assertAlmostEqual(left[2], 0.0)
        self.assertAlmostEqual(right[2], 0.0)
        self.assertNotEqual(left[0], right[0])


class G1GateTest(unittest.TestCase):
    def test_complete_gate_passes_only_with_all_conditions(self) -> None:
        manifest = {
            "architecture": {"seeds": [1, 2]},
            "advance_if": {
                "balanced_accuracy_uplift_percentage_points_minimum_each_seed": 8.0,
                "preserve_accuracy_drop_percentage_points_maximum_each_seed": 5.0,
            },
        }
        validation = {
            "houses": 2,
            "samples": [
                {"valid_slot_modes": ["PRESERVE"], "object_type": "Drawer"},
                {"valid_slot_modes": ["FLIP"], "object_type": "Drawer"},
                {"valid_slot_modes": ["PRESERVE"], "object_type": "Doorway"},
                {"valid_slot_modes": ["FLIP"], "object_type": "Doorway"},
                {"valid_slot_modes": ["PRESERVE", "FLIP"], "object_type": "Drawer"},
            ],
        }
        baseline_metrics = {
            "balanced_accuracy": 0.70,
            "by_mode": {"PRESERVE": {"accuracy": 0.90}, "FLIP": {"accuracy": 0.50}},
            "by_type": {
                "Drawer": {"balanced_accuracy": 0.65},
                "Doorway": {"balanced_accuracy": 0.65},
            },
        }
        challenger_metrics = {
            "balanced_accuracy": 0.82,
            "by_mode": {"PRESERVE": {"accuracy": 0.88}, "FLIP": {"accuracy": 0.76}},
            "by_type": {
                "Drawer": {"balanced_accuracy": 0.80},
                "Doorway": {"balanced_accuracy": 0.78},
            },
        }
        baseline_predictions = [
            {"sample_id": "a", "truth": "PRESERVE", "prediction": "PRESERVE"},
            {"sample_id": "b", "truth": "FLIP", "prediction": "PRESERVE"},
            {"sample_id": "c", "truth": "PRESERVE", "prediction": "PRESERVE"},
            {"sample_id": "d", "truth": "FLIP", "prediction": "PRESERVE"},
        ]
        challenger_predictions = [
            {"sample_id": "a", "truth": "PRESERVE", "prediction": "PRESERVE"},
            {"sample_id": "b", "truth": "FLIP", "prediction": "FLIP"},
            {"sample_id": "c", "truth": "PRESERVE", "prediction": "PRESERVE"},
            {"sample_id": "d", "truth": "FLIP", "prediction": "FLIP"},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for seed in (1, 2):
                for arm, metrics, predictions in (
                    ("b1_single", baseline_metrics, baseline_predictions),
                    ("g1_triplet", challenger_metrics, challenger_predictions),
                ):
                    run = root / arm / f"seed-{seed}"
                    run.mkdir(parents=True)
                    prediction_path = run / "predictions.json"
                    prediction_path.write_text(json.dumps(predictions), encoding="utf-8")
                    (run / "result.json").write_text(json.dumps({
                        "arm": arm, "seed": seed, "best_epoch": 1,
                        "checkpoint_sha256": f"checkpoint-{arm}-{seed}",
                        "predictions_sha256": sha256_file(prediction_path),
                        "validation": metrics,
                    }), encoding="utf-8")
            result = evaluate(manifest, validation, root)
        self.assertTrue(result["decision"]["passed"])
        self.assertEqual(result["state"], "ADVANCE_G1_ACTIVE_MULTIVIEW_APPEARANCE")
        self.assertTrue(all(row["rescue"] == 2 and row["collateral"] == 0 for row in result["seeds"]))


if __name__ == "__main__":
    unittest.main()
