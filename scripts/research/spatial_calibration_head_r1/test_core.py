#!/usr/bin/env python3

import unittest

import numpy as np
import torch

from core import (
    SpatialCalibrationHead,
    SpatialCalibrationNoConfidence,
    apply_global_affine,
    evaluate_predictions,
    fit_ridge,
    predict_ridge,
    predict_spatial,
    regional_feature_inputs,
    spatial_loss,
    train_spatial_model,
    trainable_parameters,
)


class ModelAndFeatureTest(unittest.TestCase):
    def test_exact_parameter_budgets(self) -> None:
        self.assertEqual(trainable_parameters(SpatialCalibrationHead()), 9423)
        self.assertEqual(trainable_parameters(SpatialCalibrationNoConfidence()), 9410)

    def test_regional_features_are_three_by_781(self) -> None:
        y, x = np.mgrid[0:60, 0:90]
        depth = (0.8 + y / 100 + x / 200).astype(np.float32)
        tokens = np.arange(6 * 9 * 384, dtype=np.float32).reshape(6, 9, 384) / 10000
        result = regional_feature_inputs(tokens, depth, [80.0, 82.0, 45.0, 30.0])
        self.assertEqual(result.shape, (3, 781))
        self.assertTrue(np.all(np.isfinite(result)))
        np.testing.assert_allclose(result[:, -5], [1 / 6, 1 / 2, 5 / 6], atol=1e-6)

    def test_output_bounds_and_shape(self) -> None:
        model = SpatialCalibrationHead()
        output = model(torch.zeros(2, 3, 781), torch.ones(2, 3))
        self.assertEqual(output["clearance"].shape, (2, 3))
        self.assertTrue(torch.all(output["scale"] >= 0.25))
        self.assertTrue(torch.all(output["scale"] <= 4.0))
        self.assertTrue(torch.all(output["offset"] >= -3.0))
        self.assertTrue(torch.all(output["offset"] <= 3.0))

    def test_spatial_loss_is_finite_and_backpropagates(self) -> None:
        model = SpatialCalibrationHead()
        output = model(torch.randn(4, 3, 781), torch.full((4, 3), 1.5))
        components = spatial_loss(output, torch.full((4, 3), 1.2), torch.ones(4, 3, dtype=torch.bool))
        components["total"].backward()
        self.assertTrue(torch.isfinite(components["total"]))
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))


class BaselineAndTrainingTest(unittest.TestCase):
    def test_ridge_recovers_two_output_mapping(self) -> None:
        rng = np.random.default_rng(9)
        features = rng.normal(size=(100, 8))
        kernel = rng.normal(size=(8, 2))
        targets = features @ kernel + [1.2, -0.1]
        model = fit_ridge(features, targets, ridge_lambda=1e-6)
        self.assertLess(np.mean(np.abs(predict_ridge(model, features) - targets)), 1e-5)

    def test_global_affine_invalid_parameters_become_unknown(self) -> None:
        raw = np.ones((2, 3))
        prediction, known = apply_global_affine(raw, np.asarray([[2.0, 0.1], [8.0, 0.0]]))
        np.testing.assert_allclose(prediction[0], 2.1)
        self.assertTrue(np.all(known[0]))
        self.assertFalse(np.any(known[1]))

    def test_short_synthetic_training_is_deterministic(self) -> None:
        rng = np.random.default_rng(4)
        features = rng.normal(size=(12, 3, 781)).astype(np.float32)
        raw = rng.uniform(0.8, 2.5, size=(12, 3)).astype(np.float32)
        truth = (1.1 * raw + 0.05).astype(np.float32)
        valid = np.ones((12, 3), dtype=bool)
        config = {"seed": 20260804, "learning_rate": 0.001, "weight_decay": 0.0001, "epochs": 2, "batch_size_frames": 4, "gradient_clip_norm": 1.0}
        first, first_standardizer, first_losses = train_spatial_model(features, raw, truth, valid, np.arange(10), config)
        second, second_standardizer, second_losses = train_spatial_model(features, raw, truth, valid, np.arange(10), config)
        self.assertEqual(first_losses, second_losses)
        for left, right in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        first_prediction = predict_spatial(first, first_standardizer, features, raw)
        second_prediction = predict_spatial(second, second_standardizer, features, raw)
        np.testing.assert_array_equal(first_prediction[0], second_prediction[0])

    def test_training_skips_only_batches_without_supervised_regions(self) -> None:
        rng = np.random.default_rng(8)
        features = rng.normal(size=(130, 3, 781)).astype(np.float32)
        raw = rng.uniform(0.8, 2.5, size=(130, 3)).astype(np.float32)
        truth = (raw + 0.1).astype(np.float32)
        valid = np.zeros((130, 3), dtype=bool)
        valid[0, 1] = True
        raw[~valid] = np.nan
        config = {"seed": 20260804, "learning_rate": 0.001, "weight_decay": 0.0001, "epochs": 1, "batch_size_frames": 64, "gradient_clip_norm": 1.0}
        model, _standardizer, losses = train_spatial_model(features, raw, truth, valid, np.arange(130), config)
        self.assertEqual(len(losses), 1)
        self.assertTrue(np.isfinite(losses[0]))
        self.assertTrue(all(torch.isfinite(parameter).all() for parameter in model.parameters()))


class MetricsTest(unittest.TestCase):
    def test_parent_macro_gates_and_conditional_false_clear(self) -> None:
        records = []
        for parent in ("a", "b"):
            for index in range(3):
                records.append({"parent_id": parent, "video_id": f"{parent}-v", "timestamp": float(index)})
        truth = np.full((6, 3), 1.2)
        prediction = truth.copy()
        known = np.ones_like(truth, dtype=bool)
        valid = np.ones_like(truth, dtype=bool)
        confidence = np.full_like(truth, 0.99)
        result = evaluate_predictions(records, prediction, known, truth, valid, confidence)
        self.assertTrue(all(result["gates"].values()))
        self.assertEqual(result["parent_macro"]["false_clear_rate"], 0.0)
        self.assertLess(result["parent_macro"]["confidence_ece"], 0.02)

        prediction[0, 0] = 3.0
        failed = evaluate_predictions(records, prediction, known, truth, valid)
        self.assertGreater(failed["parents"]["a"]["conditional_false_clear_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
