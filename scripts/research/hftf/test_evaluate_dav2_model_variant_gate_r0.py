from __future__ import annotations

import unittest

import numpy as np
from evaluate_dav2_model_variant_gate_r0 import compare_geometry, depth_metrics


class DepthMetricsTest(unittest.TestCase):
    def test_identity_has_zero_error_and_unit_scale(self) -> None:
        truth = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result = depth_metrics(truth.copy(), truth)
        self.assertEqual(result["paired_coverage"], 1.0)
        self.assertEqual(result["metric_abs_rel_median"], 0.0)
        self.assertEqual(result["scale_aligned_abs_rel_median"], 0.0)
        self.assertEqual(result["median_scale"], 1.0)

    def test_scale_aligned_error_removes_only_one_global_scale(self) -> None:
        truth = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        candidate = truth * 2.0
        result = depth_metrics(candidate, truth)
        self.assertEqual(result["metric_abs_rel_median"], 1.0)
        self.assertEqual(result["scale_aligned_abs_rel_median"], 0.0)
        self.assertEqual(result["median_scale"], 0.5)


def valid_field(clear: bool = True) -> dict:
    occupied = not clear
    return {
        "status": "VALID",
        "bands": {
            band: {
                "clearance_m": 2.5 if clear else 0.5,
                "occupied_by_horizon": {
                    "1.0": occupied,
                    "1.5": occupied,
                    "2.0": occupied,
                },
            }
            for band in ("left", "center", "right")
        },
    }


class GeometryComparisonTest(unittest.TestCase):
    def test_identity_is_exact(self) -> None:
        field = valid_field()
        rows = [
            {
                "sequence_id": "s",
                "baseline": field,
                "candidate": field,
            },
            {
                "sequence_id": "s",
                "baseline": field,
                "candidate": field,
            },
        ]
        result = compare_geometry(rows)
        self.assertEqual(result["status_exact_agreement"], 1.0)
        self.assertEqual(result["geometry_state_exact_agreement"], 1.0)
        self.assertEqual(result["transition_change_agreement"], 1.0)

    def test_state_and_transition_drift_are_counted(self) -> None:
        clear = valid_field(True)
        occupied = valid_field(False)
        rows = [
            {
                "sequence_id": "s",
                "baseline": clear,
                "candidate": clear,
            },
            {
                "sequence_id": "s",
                "baseline": clear,
                "candidate": occupied,
            },
        ]
        result = compare_geometry(rows)
        self.assertEqual(result["geometry_state_change_frames"], 1)
        self.assertEqual(result["geometry_state_exact_agreement"], 0.5)
        self.assertEqual(result["transition_change_agreement"], 0.0)


if __name__ == "__main__":
    unittest.main()
