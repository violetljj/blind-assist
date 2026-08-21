from __future__ import annotations

import unittest

import cv2
import numpy as np

import run_p1_a1_local_validity as a1
from run_rgb_observer import flow_bbox


class FlowHealthTest(unittest.TestCase):
    def test_instrumented_forward_flow_preserves_frozen_bbox(self):
        previous = np.zeros((96, 96), dtype=np.uint8)
        current = np.zeros((96, 96), dtype=np.uint8)
        cv2.rectangle(previous, (20, 25), (55, 60), 255, -1)
        cv2.line(previous, (20, 25), (55, 60), 0, 2)
        cv2.rectangle(current, (23, 27), (58, 62), 255, -1)
        cv2.line(current, (23, 27), (58, 62), 0, 2)
        points = cv2.goodFeaturesToTrack(previous, maxCorners=40, qualityLevel=0.01, minDistance=3)
        bbox = [20.0, 25.0, 56.0, 61.0]
        frozen_box, frozen_points = flow_bbox(previous, current, points, bbox, 96, 96)
        instrumented_box, instrumented_points, health = a1.flow_bbox_with_health(
            previous, current, points, bbox, 96, 96
        )
        self.assertEqual(frozen_box, instrumented_box)
        np.testing.assert_array_equal(frozen_points, instrumented_points)
        self.assertEqual(set(health), set(a1.FEATURE_DIRECTIONS) - {"initial_anchor_appearance"})
        self.assertTrue(all(np.isfinite(value) for value in health.values()))


class CompactGateTest(unittest.TestCase):
    def grid(self):
        return {
            feature: [{"quantile": quantile, "threshold": quantile} for quantile in a1.QUANTILES]
            for feature in a1.FEATURE_DIRECTIONS
        }

    def test_gate_family_is_exactly_bounded(self):
        gates = list(a1._gate_family(self.grid()))
        self.assertEqual(len(gates), 3069)
        self.assertEqual(sum(len(gate) == 1 for gate in gates), 72)
        self.assertEqual(sum(len(gate) == 2 for gate in gates), 2268)
        triples = [gate for gate in gates if len(gate) == 3]
        self.assertEqual(len(triples), 729)
        self.assertTrue(all(tuple(item["feature"] for item in gate) == a1.TRIPLE_FEATURES for gate in triples))

    def test_predicates_use_frozen_health_direction(self):
        healthy = {
            "point_survival_ratio": 0.9,
            "fb_error_median_px": 0.1,
            "affine_ransac_inlier_ratio": 0.9,
            "tracked_point_spatial_coverage": 0.9,
            "flow_residual_dispersion": 0.1,
            "bbox_center_jump": 0.1,
            "affine_scale_jump": 0.1,
            "initial_anchor_appearance": 0.9,
        }
        predicates = [
            {"feature": feature, "op": direction, "quantile": 0.5, "threshold": 0.5}
            for feature, direction in a1.FEATURE_DIRECTIONS.items()
        ]
        self.assertTrue(a1._passes(healthy, predicates))
        unhealthy = dict(healthy)
        unhealthy["fb_error_median_px"] = 0.8
        self.assertFalse(a1._passes(unhealthy, predicates))

    def terminal_row(self, retention: bool, signal: bool, name: str):
        return {
            "retention_hard_pass": retention,
            "meaningful_mechanism_pass": signal,
            "episode_macro_wrong_reduction": float(signal),
            "frame_aggregate_wrong_reduction": float(signal),
            "max_wrong_lock_duration_reduction": float(signal),
            "correct_assertion_retention": 0.9 if retention else 0.1,
            "predicate_count": 1,
            "canonical": name,
        }

    def test_terminal_rules_are_exhaustive_and_retention_first(self):
        terminal, _ = a1._choose_terminal([self.terminal_row(True, True, "admitted-signal")])
        self.assertEqual(terminal, "CONSERVATIVE_LOCAL_VALIDITY_SIGNAL_ESTABLISHED")
        terminal, _ = a1._choose_terminal([
            self.terminal_row(True, False, "admissible-no-signal"),
            self.terminal_row(False, True, "abstention-signal"),
        ])
        self.assertEqual(terminal, "VALIDITY_GAIN_ONLY_BY_ABSTENTION")
        terminal, _ = a1._choose_terminal([self.terminal_row(True, False, "no-signal")])
        self.assertEqual(terminal, "LOCAL_FLOW_VALIDITY_NOT_IDENTIFIABLE_FROM_CURRENT_RGB_HEALTH_FEATURES")


if __name__ == "__main__":
    unittest.main()
