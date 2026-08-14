import unittest
from types import SimpleNamespace

import numpy as np

from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_opportunity_aware_no_regret_ranker as subject


def _record(
    neighbor: str,
    translation: float,
    gain: int,
    coverage: float,
) -> SimpleNamespace:
    return SimpleNamespace(
        parent_id="p",
        reference_id="r",
        target_gain=gain,
        coverage=coverage,
        pair=SimpleNamespace(
            translation_m=translation,
            rotation_deg=0.0,
            gap_s=0.1,
            neighbor=SimpleNamespace(frame_id=neighbor),
        ),
    )


class OpportunityAwareNoRegretRankerTest(unittest.TestCase):
    def test_model_capacity_contract(self) -> None:
        parameter_count = sum(parameter.numel() for parameter in subject.OpportunityAwareUtilityRanker().parameters())
        self.assertGreaterEqual(parameter_count, 100_000)
        self.assertLessEqual(parameter_count, 1_000_000)

    def test_teacher_positive_must_beat_generic_and_passive(self) -> None:
        records = [
            _record("candidate", 0.5, 8, 0.2),
            _record("passive", 0.4, 7, 1.0),
            _record("generic", 1.0, 6, 0.1),
        ]
        labels, _weights, audit = subject.opportunity_targets(records, ["S"] * len(records))
        np.testing.assert_array_equal(labels, [1.0, 0.0, 0.0])
        self.assertEqual(1, audit["positive_candidate_count"])
        records[0].target_gain = 7
        with self.assertRaisesRegex(RuntimeError, "no positives"):
            subject.opportunity_targets(records, ["S"] * len(records))

    def test_gate_requires_both_opportunity_and_utility_bounds(self) -> None:
        records = [_record("learned", 0.5, 0, 0.0), _record("generic", 1.0, 0, 0.0)]
        means = np.asarray([[1.0, 0.0], [0.9, 0.0], [1.1, 0.0]], dtype=np.float64)
        log_variances = np.full_like(means, -5.0)
        high_opportunity = np.asarray([[4.0, -4.0], [4.2, -4.0], [3.8, -4.0]], dtype=np.float64)
        scores, receipt = subject.opportunity_no_regret_gate(records, means, log_variances, high_opportunity)
        self.assertEqual(0, int(np.argmax(scores)))
        self.assertEqual(1, receipt["learned_override_count"])

        low_opportunity = np.asarray([[-4.0, -4.0], [-3.8, -4.0], [-4.2, -4.0]], dtype=np.float64)
        scores, receipt = subject.opportunity_no_regret_gate(records, means, log_variances, low_opportunity)
        self.assertEqual(1, int(np.argmax(scores)))
        self.assertEqual(1, receipt["generic_fallback_count"])

    def test_gate_does_not_read_teacher_fields(self) -> None:
        records = [_record("learned", 0.5, 100, 1.0), _record("generic", 1.0, 0, 0.0)]
        means = np.asarray([[1.0, 0.0], [0.9, 0.0], [1.1, 0.0]], dtype=np.float64)
        log_variances = np.full_like(means, -5.0)
        logits = np.asarray([[4.0, -4.0], [4.2, -4.0], [3.8, -4.0]], dtype=np.float64)
        before, receipt_before = subject.opportunity_no_regret_gate(records, means, log_variances, logits)
        records[0].target_gain = -999
        records[0].coverage = -999.0
        records[1].target_gain = 999
        records[1].coverage = 999.0
        after, receipt_after = subject.opportunity_no_regret_gate(records, means, log_variances, logits)
        np.testing.assert_array_equal(before, after)
        self.assertEqual(receipt_before["selection_receipt_sha256"], receipt_after["selection_receipt_sha256"])


if __name__ == "__main__":
    unittest.main()
