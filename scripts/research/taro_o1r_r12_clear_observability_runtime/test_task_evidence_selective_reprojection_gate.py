import unittest
from types import SimpleNamespace

import numpy as np

from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_selective_reprojection_gate as subject


def _record(name: str, translation: float, novel: float, target: int) -> SimpleNamespace:
    analytic = {
        "reprojection_novel_cell_count": novel,
        "unexplained_warp_hole_cell_count": novel,
        "photometric_inconsistent_cell_count": 0.0,
        "novel_appearance_strength_sum": novel / 2.0,
        "candidate_visible_unknown_cell_count": 10.0,
        "direct_warp_coverage_fraction": 0.5,
        "explained_warp_coverage_fraction": 0.8,
        "photometric_residual_threshold": 0.2,
    }
    return SimpleNamespace(
        parent_id="parent",
        reference_id="reference",
        pair=SimpleNamespace(
            translation_m=translation,
            rotation_deg=0.0,
            gap_s=0.5,
            neighbor=SimpleNamespace(frame_id=name),
        ),
        analytic=analytic,
        target_gain=target,
    )


class SelectiveReprojectionGateTest(unittest.TestCase):
    def test_held_example_features_do_not_read_target(self) -> None:
        records = [_record("generic", 1.0, 2.0, -999), _record("proposal", 0.5, 4.0, 999)]
        before = subject.proposal_examples("HELD", records, include_labels=False)
        records[0].target_gain = 999
        records[1].target_gain = -999
        after = subject.proposal_examples("HELD", records, include_labels=False)
        self.assertEqual(1, len(before))
        self.assertIsNone(before[0].label)
        np.testing.assert_allclose(before[0].features, after[0].features)

    def test_training_label_is_proposal_strictly_better_than_generic(self) -> None:
        records = [_record("generic", 1.0, 2.0, 5), _record("proposal", 0.5, 4.0, 6)]
        examples = subject.proposal_examples("TRAIN", records, include_labels=True)
        self.assertEqual(1.0, examples[0].label)
        records[1].target_gain = 5
        examples = subject.proposal_examples("TRAIN", records, include_labels=True)
        self.assertEqual(0.0, examples[0].label)

    def test_absent_direct_correspondence_has_finite_gate_encoding(self) -> None:
        record = _record("candidate", 0.5, 4.0, 0)
        record.analytic["direct_warp_coverage_fraction"] = 0.0
        record.analytic["photometric_residual_threshold"] = float("inf")
        vector = subject._record_vector(record)
        self.assertTrue(np.all(np.isfinite(vector)))
        self.assertEqual(0.0, vector[7])

    def test_gate_lcb_accepts_or_rejects_without_target(self) -> None:
        records = [_record("generic", 1.0, 2.0, 0), _record("proposal", 0.5, 4.0, 0)]
        examples = subject.proposal_examples("HELD", records, include_labels=False)
        accepted, receipt = subject.gated_selection_scores(
            records, examples, np.asarray([[2.0], [1.8], [2.2], [1.9], [2.1]])
        )
        self.assertEqual([0.0, 1.0], accepted.tolist())
        self.assertEqual(1, receipt["accepted_override_count"])
        rejected, receipt = subject.gated_selection_scores(
            records, examples, np.asarray([[0.2], [-0.2], [0.1], [-0.1], [0.0]])
        )
        self.assertEqual([1.0, 0.0], rejected.tolist())
        self.assertEqual(1, receipt["rejected_override_count"])


if __name__ == "__main__":
    unittest.main()
