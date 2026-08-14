import unittest
from types import SimpleNamespace

import numpy as np
import torch

from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_rgb_query_interaction_ranker as subject


class RgbQueryInteractionRankerTest(unittest.TestCase):
    def test_rgb_plane_contract_is_finite_and_illumination_normalized(self) -> None:
        ramp = np.tile(np.arange(256, dtype=np.uint8), (192, 1))
        rgb = np.stack((ramp, ramp, ramp), axis=-1)
        planes = subject._rgb_planes(rgb)
        self.assertEqual((3, 192, 256), planes.shape)
        self.assertTrue(np.all(np.isfinite(planes)))
        self.assertLess(abs(float(np.mean(planes[0]))), 0.01)
        self.assertGreater(float(np.max(planes[1])), 0.0)

    def test_feature_and_model_capacity_contract(self) -> None:
        self.assertEqual(720, subject.RGB_TOKEN_WIDTH)
        self.assertEqual(1008, subject.CANDIDATE_TOKEN_WIDTH)
        self.assertEqual(subject.r23.TOTAL_FEATURE_COUNT + 9 * 720, subject.TOTAL_FEATURE_COUNT)
        parameter_count = sum(parameter.numel() for parameter in subject.RgbQueryInteractionRanker().parameters())
        self.assertGreaterEqual(parameter_count, 100_000)
        self.assertLessEqual(parameter_count, 1_000_000)

    def test_transform_keeps_reference_local_pose_blocks(self) -> None:
        records = []
        for reference, value in (("a", 1.0), ("a", 3.0), ("b", 100.0), ("b", 104.0)):
            features = np.zeros(subject.TOTAL_FEATURE_COUNT, dtype=np.float32)
            features[: subject.r23.BASE_FEATURE_COUNT] = value
            records.append(SimpleNamespace(reference_id=reference, features=features))
        transform = subject.RgbStructuredFeatureTransform.fit(records)
        inputs = transform.apply(records)
        position = len(subject.r23.POSE_FEATURE_INDICES)
        np.testing.assert_allclose(inputs.pose[:, position], [-1.0, 1.0, -1.0, 1.0])
        self.assertEqual((4, 9, 1008), inputs.candidate_tokens.shape)

    def test_model_forward_outputs_utility_distribution(self) -> None:
        torch.manual_seed(3)
        model = subject.RgbQueryInteractionRanker().eval()
        with torch.no_grad():
            mean, log_variance = model(
                torch.zeros(2, subject.r23.POSE_TRANSFORMED_WIDTH),
                torch.zeros(2, 9, subject.r23.STATIC_TOKEN_WIDTH),
                torch.zeros(2, 9, subject.CANDIDATE_TOKEN_WIDTH),
                torch.zeros(2),
            )
        self.assertEqual((2,), tuple(mean.shape))
        self.assertEqual((2,), tuple(log_variance.shape))
        self.assertTrue(torch.all(torch.isfinite(mean)))
        self.assertTrue(torch.all(torch.isfinite(log_variance)))


if __name__ == "__main__":
    unittest.main()
