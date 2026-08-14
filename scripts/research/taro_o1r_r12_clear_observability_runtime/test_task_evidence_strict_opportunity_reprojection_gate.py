import unittest

from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_selective_reprojection_gate as r29
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_strict_opportunity_reprojection_gate as subject


class StrictOpportunityReprojectionGateTest(unittest.TestCase):
    def test_training_label_requires_beating_generic_and_passive(self) -> None:
        generic = r29_test_record("generic", 1.0, 2.0, target=5, coverage=0.5)
        proposal = r29_test_record("proposal", 0.5, 4.0, target=6, coverage=0.4)
        passive = r29_test_record("passive", 0.4, 1.0, target=7, coverage=0.9)
        records = [generic, proposal, passive]
        examples = subject.strict_proposal_examples("TRAIN", records, include_labels=True)
        self.assertEqual(0.0, examples[0].label)
        passive.target_gain = 5
        examples = subject.strict_proposal_examples("TRAIN", records, include_labels=True)
        self.assertEqual(1.0, examples[0].label)

    def test_held_features_do_not_read_target_or_passive_coverage(self) -> None:
        records = [
            r29_test_record("generic", 1.0, 2.0, target=-999, coverage=0.1),
            r29_test_record("proposal", 0.5, 4.0, target=999, coverage=0.9),
        ]
        before = subject.strict_proposal_examples("HELD", records, include_labels=False)
        records[0].target_gain = 999
        records[1].target_gain = -999
        records[0].coverage = 1.0
        records[1].coverage = 0.0
        after = subject.strict_proposal_examples("HELD", records, include_labels=False)
        self.assertIsNone(before[0].label)
        self.assertEqual(before[0].features.tolist(), after[0].features.tolist())


def r29_test_record(name: str, translation: float, novel: float, target: int, coverage: float):
    from types import SimpleNamespace

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
        coverage=coverage,
    )


if __name__ == "__main__":
    unittest.main()
