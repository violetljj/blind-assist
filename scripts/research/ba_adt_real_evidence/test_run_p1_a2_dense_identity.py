from __future__ import annotations

import unittest

import cv2
import numpy as np

import run_p1_a2_dense_identity as a2


class DenseConsensusTest(unittest.TestCase):
    def features(self):
        rng = np.random.default_rng(11)
        values = rng.normal(size=(a2.PATCH_COUNT, a2.FEATURE_DIM)).astype("float32")
        return values / np.linalg.norm(values, axis=1, keepdims=True)

    def test_identical_patch_grid_has_full_dense_consensus(self):
        features = self.features()
        result = a2.dense_consensus(features, features.copy())
        self.assertEqual(result["mutual_match_count"], a2.PATCH_COUNT)
        self.assertAlmostEqual(result["anchor_match_fraction"], 1.0)
        self.assertAlmostEqual(result["match_confidence"], 1.0, places=5)
        self.assertAlmostEqual(result["spatial_consistency"], 1.0)
        self.assertAlmostEqual(result["anchor_coverage"], 1.0)
        self.assertAlmostEqual(result["correspondence_dispersion"], 0.0, places=6)

    def test_permuted_grid_keeps_identity_matches_but_loses_spatial_consistency(self):
        features = self.features()
        rng = np.random.default_rng(23)
        permuted = features[rng.permutation(a2.PATCH_COUNT)]
        result = a2.dense_consensus(features, permuted)
        self.assertAlmostEqual(result["anchor_match_fraction"], 1.0)
        self.assertAlmostEqual(result["match_confidence"], 1.0, places=5)
        self.assertLess(result["spatial_consistency"], 0.25)


class CropAndPolicyTest(unittest.TestCase):
    def test_crop_contract_is_fixed_chw_224(self):
        image = np.zeros((80, 120, 3), dtype=np.uint8)
        cv2.rectangle(image, (10, 20), (50, 60), (20, 100, 220), -1)
        tensor = a2._crop_tensor(image, [10.2, 20.3, 50.8, 60.1])
        self.assertEqual(tensor.shape, (3, a2.INPUT_SIZE, a2.INPUT_SIZE))
        self.assertTrue(np.isfinite(tensor).all())

    def grid(self):
        return {
            feature: [{"quantile": quantile, "threshold": quantile} for quantile in a2.POLICY_QUANTILES]
            for feature in a2.POLICY_FEATURES
        }

    def test_policy_family_is_single_bounded_four_way_consensus(self):
        policies = list(a2._policy_family(self.grid()))
        self.assertEqual(len(policies), 625)
        self.assertTrue(all(tuple(row["feature"] for row in policy) == a2.POLICY_FEATURES for policy in policies))

    def test_three_state_decision_and_fail_closed_boundary(self):
        predicates = [
            {"feature": feature, "op": "ge", "quantile": 0.5, "threshold": 0.5}
            for feature in a2.POLICY_FEATURES
        ]
        valid = {feature: 0.8 for feature in a2.POLICY_FEATURES}
        self.assertEqual(a2._decision(valid, predicates), "VALID")
        uncertain = dict(valid, anchor_coverage=0.2)
        self.assertEqual(a2._decision(uncertain, predicates), "UNCERTAIN")
        invalid = dict(uncertain, spatial_consistency=0.2)
        self.assertEqual(a2._decision(invalid, predicates), "INVALID")


class TerminalTest(unittest.TestCase):
    def row(self, retention: bool, signal: bool, name: str):
        return {
            "admission_pass": retention and signal,
            "retention_hard_pass": retention,
            "meaningful_mechanism_pass": signal,
            "frame_aggregate_wrong_reduction": float(signal),
            "max_wrong_lock_duration_reduction": float(signal),
            "episode_macro_wrong_reduction": float(signal),
            "correct_assertion_retention": 0.91 if retention else 0.2,
            "canonical": name,
        }

    def test_exhaustive_three_terminals(self):
        terminal, _ = a2._choose_terminal([self.row(True, True, "admitted")])
        self.assertEqual(terminal, "DENSE_IDENTITY_VALIDITY_SIGNAL_ESTABLISHED")
        terminal, _ = a2._choose_terminal([self.row(True, False, "retained"), self.row(False, True, "abstain")])
        self.assertEqual(terminal, "DENSE_IDENTITY_GAIN_ONLY_BY_ABSTENTION")
        terminal, _ = a2._choose_terminal([self.row(True, False, "insufficient")])
        self.assertEqual(terminal, "DENSE_IDENTITY_NOT_SUFFICIENT")


if __name__ == "__main__":
    unittest.main()
