from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sanpo_deterministic_linear_probe as probe


class DeterministicLinearProbeTest(unittest.TestCase):
    def test_evenly_spaced_indices_are_stable_and_cover_range(self) -> None:
        first = probe.evenly_spaced_indices(10, 4)
        second = probe.evenly_spaced_indices(10, 4)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(np.array([0, 3, 6, 9]), first)
        np.testing.assert_array_equal(np.arange(3), probe.evenly_spaced_indices(3, 8))

    def test_balanced_class_samples_are_order_invariant_after_sorting(self) -> None:
        features = np.arange(6 * 2, dtype=np.float64).reshape(6, 2)
        labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        sample_ids = np.array(["b", "a", "c", "b", "a", "c"])
        selected_features, selected_labels, selected_ids = probe.balance_samples(
            features, labels, sample_ids, class_count=2, maximum_per_class=2,
        )
        np.testing.assert_array_equal(selected_labels, np.array([0, 0, 1, 1]))
        np.testing.assert_array_equal(selected_ids, np.array(["a", "c", "a", "c"]))
        np.testing.assert_array_equal(selected_features, features[[1, 2, 4, 5]])

    def test_ridge_solution_and_raw_coefficients_are_repeatable(self) -> None:
        features = np.array([
            [-2.0, 0.0], [-1.0, 0.1], [1.0, -0.1], [2.0, 0.0],
        ])
        labels = np.array([0, 0, 1, 1])
        first = probe.fit_ridge_probe(features, labels, class_count=2, ridge=1e-3)
        second = probe.fit_ridge_probe(features, labels, class_count=2, ridge=1e-3)
        self.assertEqual(first["coefficient_sha256"], second["coefficient_sha256"])
        np.testing.assert_array_equal(first["kernel"], second["kernel"])
        np.testing.assert_array_equal(first["bias"], second["bias"])
        predictions = probe.predict_labels(features, first["kernel"], first["bias"])
        np.testing.assert_array_equal(labels, predictions)

    def test_standardized_and_raw_logits_match(self) -> None:
        features = np.array([[-3.0, 2.0], [0.0, 1.0], [4.0, -2.0]])
        labels = np.array([0, 1, 1])
        result = probe.fit_ridge_probe(features, labels, class_count=2, ridge=0.1)
        standardized = (features - result["feature_mean"]) / result["feature_scale"]
        expected = standardized @ result["standardized_kernel"] + result["standardized_bias"]
        actual = features @ result["kernel"] + result["bias"]
        np.testing.assert_allclose(expected, actual, rtol=0.0, atol=1e-12)

    def test_repeat_contract_detects_identical_predictions(self) -> None:
        kernel = np.arange(8, dtype=np.float64).reshape(4, 2)
        bias = np.array([0.5, -0.5])
        predictions = [np.array([[0, 1], [1, 0]], dtype=np.uint8)]
        repeats = probe.repeat_consistency([
            (kernel, bias, predictions),
            (kernel.copy(), bias.copy(), [predictions[0].copy()]),
        ])
        self.assertTrue(repeats["exact_coefficient_match"])
        self.assertEqual(0.0, repeats["maximum_coefficient_absolute_difference"])
        self.assertEqual(1.0, repeats["dev_argmax_agreement"])


if __name__ == "__main__":
    unittest.main()
