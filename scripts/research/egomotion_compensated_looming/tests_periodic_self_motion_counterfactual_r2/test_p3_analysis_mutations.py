from __future__ import annotations

import copy
import unittest

from ..periodic_self_motion_counterfactual_r2 import p3_analysis_r0 as analysis


def pair_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(analysis.PAIR_COUNT):
        response = 0.02 if index not in {3, 7} else 0.01
        rows.append(
            {
                "pair_index": index,
                "evaluable": True,
                "compensated_expansion_median_per_s": response,
                "detected_feature_count": 100,
                "forward_backward_consistent_count": 90,
                "forward_backward_consistent_fraction": 0.9,
                "median_forward_backward_error_px": 0.2,
            }
        )
    return rows


class P3AnalysisMutationTests(unittest.TestCase):
    def test_pair_reducer_resets_on_abstention_and_threshold_equality(self) -> None:
        rows = pair_rows()
        rows[1]["evaluable"] = False
        rows[1]["compensated_expansion_median_per_s"] = None
        reduced = analysis.reduce_pair_rows(rows)
        self.assertEqual(reduced["trigger_count"], 592)
        self.assertEqual(reduced["scheduled_pair_count"], 601)

    def test_pair_count_order_and_forged_trigger_fail_closed(self) -> None:
        with self.assertRaisesRegex(analysis.InvalidAnalysis, "PAIR_COUNT"):
            analysis.reduce_pair_rows(pair_rows()[:-1])
        mutated = pair_rows()
        mutated[4]["pair_index"] = 3
        with self.assertRaisesRegex(analysis.InvalidAnalysis, "PAIR_ORDER"):
            analysis.reduce_pair_rows(mutated)
        mutated = pair_rows()
        mutated[0]["compensated_three_pair_trigger"] = True
        with self.assertRaisesRegex(analysis.InvalidAnalysis, "FORGED"):
            analysis.reduce_pair_rows(mutated)

    def test_abstention_response_and_nonfinite_fail_closed(self) -> None:
        mutated = pair_rows()
        mutated[0]["evaluable"] = False
        with self.assertRaisesRegex(analysis.InvalidAnalysis, "ABSTENTION"):
            analysis.reduce_pair_rows(mutated)
        mutated = pair_rows()
        mutated[0]["compensated_expansion_median_per_s"] = float("nan")
        with self.assertRaisesRegex(analysis.InvalidAnalysis, "RESPONSE_INVALID"):
            analysis.reduce_pair_rows(mutated)

    def test_frozen_cluster_grid_and_six_arm_keyset(self) -> None:
        fixture = analysis.fixture_clusters()
        with self.assertRaisesRegex(analysis.InvalidAnalysis, "CLUSTER_COUNT"):
            analysis.validate_clusters(fixture[:-1])
        mutated = copy.deepcopy(fixture)
        mutated[-1] = copy.deepcopy(mutated[0])
        with self.assertRaisesRegex(analysis.InvalidAnalysis, "DUPLICATE"):
            analysis.validate_clusters(mutated)
        mutated = copy.deepcopy(fixture)
        mutated[0]["arms"].pop("STATIC_CAMERA__BLUR")
        with self.assertRaisesRegex(analysis.InvalidAnalysis, "SIX_ARM"):
            analysis.validate_clusters(mutated)
        mutated = copy.deepcopy(fixture)
        mutated[0]["arms"]["GUARDRAIL__CLEAN"] = mutated[0]["arms"].pop(
            "STATIC_CAMERA__BLUR"
        )
        with self.assertRaisesRegex(analysis.InvalidAnalysis, "SIX_ARM"):
            analysis.validate_clusters(mutated)

    def test_bootstrap_lock_and_shared_family_receipt(self) -> None:
        result = analysis.analyze(analysis.fixture_clusters())
        self.assertEqual(result["family"], list(analysis.FAMILY))
        self.assertEqual(result["bootstrap"]["replicates"], 20_000)
        self.assertEqual(result["bootstrap"]["seed"], 20260728)
        self.assertEqual(result["bootstrap"]["sd_ddof"], 1)
        self.assertEqual(result["bootstrap"]["quantile_method"], "linear_type_7")
        self.assertEqual(len(result["estimands"]), 9)
        with self.assertRaisesRegex(analysis.InvalidAnalysis, "BOOTSTRAP_LOCK"):
            analysis.analyze(analysis.fixture_clusters(), replicates=19_999)
        with self.assertRaisesRegex(analysis.InvalidAnalysis, "BOOTSTRAP_LOCK"):
            analysis.analyze(analysis.fixture_clusters(), seed=1)

    def test_contrast_formulas_and_equal_block_weighting(self) -> None:
        fixture = analysis.fixture_clusters()
        result = analysis.analyze(fixture)
        motion = result["estimands"]["MOTION_CLEAN"]
        self.assertAlmostEqual(motion["theta"], 0.03)
        self.assertEqual(len(motion["block_point_estimates"]), 4)
        blur_failure = result["estimands"]["BLUR_FAILURE_UNION_STATIC"]
        self.assertAlmostEqual(blur_failure["theta"], 0.12)
        self.assertEqual(blur_failure["classification"], "SUPPORTED")

    def test_nonfinite_arm_metric_fails_closed(self) -> None:
        mutated = analysis.fixture_clusters()
        mutated[0]["arms"]["STATIC_CAMERA__CLEAN"]["trigger_density"] = float("inf")
        with self.assertRaisesRegex(analysis.InvalidAnalysis, "NONFINITE"):
            analysis.analyze(mutated)


if __name__ == "__main__":
    unittest.main()
