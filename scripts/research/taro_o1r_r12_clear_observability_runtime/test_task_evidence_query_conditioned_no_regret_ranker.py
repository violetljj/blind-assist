import unittest
from types import SimpleNamespace

import numpy as np
import torch

from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_query_conditioned_no_regret_ranker as subject


def _record(reference: str, neighbor: str, translation: float, target: int = 0) -> SimpleNamespace:
    pair = SimpleNamespace(
        translation_m=translation,
        rotation_deg=0.0,
        gap_s=0.1,
        neighbor=SimpleNamespace(frame_id=neighbor),
    )
    return SimpleNamespace(reference_id=reference, pair=pair, target_gain=target)


class QueryConditionedNoRegretRankerTest(unittest.TestCase):
    def test_feature_and_model_capacity_contract(self) -> None:
        self.assertEqual((6, 3, 4), subject.CELL_SHAPE)
        self.assertEqual(72, subject.STATIC_TOKEN_WIDTH)
        self.assertEqual(288, subject.GEOMETRY_TOKEN_WIDTH)
        self.assertEqual(
            subject.BASE_FEATURE_COUNT + 9 * 72 + 9 * 288,
            subject.TOTAL_FEATURE_COUNT,
        )
        parameter_count = sum(parameter.numel() for parameter in subject.QueryConditionedUtilityRanker().parameters())
        self.assertGreaterEqual(parameter_count, 100_000)
        self.assertLessEqual(parameter_count, 1_000_000)

    def test_cross_attention_responds_to_task_candidate_alignment(self) -> None:
        torch.manual_seed(7)
        model = subject.QueryConditionedUtilityRanker().eval()
        pose = torch.zeros(2, subject.POSE_TRANSFORMED_WIDTH)
        task = torch.zeros(2, subject.QUERY_COUNT, subject.STATIC_TOKEN_WIDTH)
        candidate = torch.zeros(2, subject.QUERY_COUNT, subject.GEOMETRY_TOKEN_WIDTH)
        task[:, 0, 0] = 1.0
        candidate[0, 0, 0] = 1.0
        candidate[1, 1, 0] = 1.0
        with torch.no_grad():
            mean, log_variance = model(pose, task, candidate, torch.zeros(2))
        self.assertEqual((2,), tuple(mean.shape))
        self.assertEqual((2,), tuple(log_variance.shape))
        self.assertFalse(torch.allclose(mean[0], mean[1]))

    def test_gate_overrides_only_for_positive_conservative_advantage(self) -> None:
        records = [_record("r", "learned", 0.5), _record("r", "generic", 1.0)]
        confident_means = np.asarray([[1.0, 0.0], [0.9, 0.0], [1.1, 0.0]], dtype=np.float64)
        low_variance = np.full_like(confident_means, -5.0)
        scores, receipt = subject.no_regret_gate(records, confident_means, low_variance)
        self.assertEqual(0, int(np.argmax(scores)))
        self.assertEqual(1, receipt["learned_override_count"])

        uncertain_means = np.asarray([[1.0, 0.0], [-1.0, 0.0], [0.2, 0.0]], dtype=np.float64)
        scores, receipt = subject.no_regret_gate(records, uncertain_means, low_variance)
        self.assertEqual(1, int(np.argmax(scores)))
        self.assertEqual(1, receipt["generic_fallback_count"])

    def test_gate_is_independent_of_teacher_targets(self) -> None:
        records = [_record("r", "learned", 0.5, target=100), _record("r", "generic", 1.0, target=0)]
        means = np.asarray([[0.8, 0.0], [0.9, 0.0], [1.0, 0.0]], dtype=np.float64)
        log_variances = np.full_like(means, -5.0)
        before, receipt_before = subject.no_regret_gate(records, means, log_variances)
        records[0].target_gain = -999
        records[1].target_gain = 999
        after, receipt_after = subject.no_regret_gate(records, means, log_variances)
        np.testing.assert_array_equal(before, after)
        self.assertEqual(receipt_before["selection_receipt_sha256"], receipt_after["selection_receipt_sha256"])


if __name__ == "__main__":
    unittest.main()
