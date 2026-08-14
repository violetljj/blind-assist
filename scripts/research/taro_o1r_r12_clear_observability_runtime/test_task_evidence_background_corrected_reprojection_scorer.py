import unittest
from types import SimpleNamespace

import numpy as np

from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_background_corrected_reprojection_scorer as subject


def _record(name: str, translation: float, excess: float, enrichment: float, novel: float, target: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        reference_id="reference",
        pair=SimpleNamespace(
            translation_m=translation,
            rotation_deg=0.0,
            gap_s=0.5,
            neighbor=SimpleNamespace(frame_id=name),
        ),
        analytic={
            "background_corrected_excess_novel_cell_count": excess,
            "task_novel_enrichment_ratio": enrichment,
            "reprojection_novel_cell_count": novel,
        },
        target_gain=target,
    )


class BackgroundCorrectedReprojectionScorerTest(unittest.TestCase):
    def test_background_expected_count_is_removed_without_target_parameters(self) -> None:
        corrected = subject.background_corrected_analytic(
            {
                "explained_warp_coverage_fraction": 0.8,
                "direct_warp_coverage_fraction": 0.5,
                "candidate_visible_unknown_cell_count": 100.0,
                "reprojection_novel_cell_count": 30.0,
            }
        )
        self.assertAlmostEqual(0.25, corrected["global_reprojection_novel_fraction"])
        self.assertAlmostEqual(25.0, corrected["background_expected_task_novel_cell_count"])
        self.assertAlmostEqual(5.0, corrected["background_corrected_excess_novel_cell_count"])

    def test_primary_gate_and_target_firewall(self) -> None:
        records = [
            _record("generic", 1.0, 2.0, 1.0, 10.0, target=-999),
            _record("task", 0.5, 3.0, 1.1, 9.0, target=999),
        ]
        before, receipt = subject.primary_selection_scores(records)
        self.assertEqual([0.0, 1.0], before.tolist())
        self.assertEqual(1, receipt["generic_override_count"])
        records[0].target_gain = 999
        records[1].target_gain = -999
        after, _ = subject.primary_selection_scores(records)
        np.testing.assert_array_equal(before, after)
        records[1].analytic["background_corrected_excess_novel_cell_count"] = 2.5
        fallback, receipt = subject.primary_selection_scores(records)
        self.assertEqual([1.0, 0.0], fallback.tolist())
        self.assertEqual(1, receipt["generic_fallback_count"])


if __name__ == "__main__":
    unittest.main()
