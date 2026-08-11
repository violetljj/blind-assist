#!/usr/bin/env python3
"""Focused tests for the frozen TARO R6 factor-level compositor."""

from __future__ import annotations

import copy
import unittest

from scripts.research.taro_o0r_candidate_scale_runtime import r5_confirmation as r5
from scripts.research.taro_o0r_candidate_scale_runtime import r6_factor_split as r6
from scripts.research.taro_o0r_candidate_scale_runtime.test_r5_confirmation import _positive_query_record


def _new_parent_components(parent: str, grid: int) -> dict[str, object]:
    seed_parent, seed_video = r5.R5_ROSTER[0]
    seed = _positive_query_record(seed_parent, seed_video, 0, grid)
    physical = f"new-video-{parent}:1.000000000"
    return r6.build_factor_components(
        analysis_role=r6.UNTOUCHED_CONFIRMATION,
        parent_id=parent,
        physical_frame_id=physical,
        query_id=f"{physical}:synthetic-{grid}",
        grid_index=grid,
        source_frame_receipt_sha256="4" * 64,
        candidate_frame_record_sha256="5" * 64,
        r6_phase_a_policy_seal_sha256="6" * 64,
        query_receipt_sha256="7" * 64,
        truth_scoring_record_sha256="8" * 64,
        source_support_available=True,
        phase_a_selected_branch="DIRECT_APPLE_SUPPORT",
        baseline=seed["baseline"],
        selected_support_boundary=seed["selected_hybrid"],
    )


class R6FactorSplitTests(unittest.TestCase):
    def test_formation_adapter_uses_factor_specific_owners_and_has_no_terminal(self) -> None:
        parent, video = r5.R5_ROSTER[0]
        pairs = []
        for grid in range(9):
            row = _positive_query_record(parent, video, 0, grid)
            components = r6.factor_components_from_r5_query_record(row)
            composite = r6.build_composite_query(components)
            self.assertEqual(components["selected_support_boundary"]["support"], composite["support"])
            self.assertEqual(components["selected_support_boundary"]["boundary"], composite["boundary"])
            self.assertEqual(components["baseline"]["query_point_clearance"], composite["query_clearance"])
            self.assertEqual(components["selected_support_boundary"]["depth_array_sha256"], composite["factor_depth_sha256"]["SUPPORT"])
            self.assertEqual(components["baseline"]["depth_array_sha256"], composite["factor_depth_sha256"]["QUERY_CLEARANCE"])
            pairs.append((components, composite))
        summary = r6.summarize_factor_split_pairs(pairs, analysis_role=r6.FORMATION_REPLAY, expected_parent_frame_counts={parent: 1})
        self.assertTrue(summary["all_gate_landscape_would_pass"])
        self.assertFalse(summary["promotion_allowed"])
        self.assertNotIn("passed", summary)
        self.assertNotIn("terminal", summary)

    def test_composite_rejects_query_block_from_wrong_owner_even_when_resealed(self) -> None:
        parent, video = r5.R5_ROSTER[0]
        row = _positive_query_record(parent, video, 0, 0)
        direct = copy.deepcopy(row["selected_hybrid"])
        direct["query_point_clearance"]["value_m"] = 4.0
        row["direct_apple_support"] = copy.deepcopy(direct)
        row["selected_hybrid"] = copy.deepcopy(direct)
        row["effects"] = r5._effects(row["baseline"], direct)
        row.pop("content_sha256")
        components = r6.factor_components_from_r5_query_record(r5._seal(row))
        composite = r6.build_composite_query(components)
        tampered = copy.deepcopy(composite)
        tampered.pop("content_sha256")
        tampered["query_clearance"] = copy.deepcopy(components["selected_support_boundary"]["query_point_clearance"])
        from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter

        tampered["query_clearance_block_sha256"] = adapter.canonical_sha256(tampered["query_clearance"])
        tampered = r6._seal(tampered)
        with self.assertRaisesRegex(r6.R6FactorSplitError, "exact-copy") as caught:
            r6.validate_composite_query(tampered, factor_components=components)
        self.assertEqual("R6_EXACT_COPY_DRIFT", caught.exception.code)

    def test_confirmation_rejects_formation_parent_and_parent_floor(self) -> None:
        consumed_parent = next(iter(r6.FORBIDDEN_FORMATION_PARENTS))
        with self.assertRaises(r6.R6FactorSplitError) as caught:
            _new_parent_components(consumed_parent, 0)
        self.assertEqual("R6_CONFIRMATION_PARENT_OVERLAP", caught.exception.code)

        parent = "900001"
        components = [_new_parent_components(parent, grid) for grid in range(9)]
        pairs = [(row, r6.build_composite_query(row)) for row in components]
        with self.assertRaises(r6.R6FactorSplitError) as caught:
            r6.summarize_factor_split_pairs(pairs, analysis_role=r6.UNTOUCHED_CONFIRMATION, expected_parent_frame_counts={parent: 1})
        self.assertEqual("R6_CONFIRMATION_PARENT_FLOOR", caught.exception.code)

    def test_eight_parent_untouched_confirmation_has_formal_terminal(self) -> None:
        parents = [f"90000{index}" for index in range(1, 9)]
        pairs = []
        for parent in parents:
            for grid in range(9):
                components = _new_parent_components(parent, grid)
                pairs.append((components, r6.build_composite_query(components)))
        summary = r6.summarize_factor_split_pairs(
            pairs,
            analysis_role=r6.UNTOUCHED_CONFIRMATION,
            expected_parent_frame_counts={parent: 1 for parent in parents},
        )
        self.assertTrue(summary["passed"])
        self.assertEqual("TARO_O0R_R6_FACTOR_SPLIT_UNTOUCHED_CONFIRMATION_PASS", summary["terminal"])
        self.assertTrue(all(gate["passed"] for gate in summary["gates"]))


if __name__ == "__main__":
    unittest.main()
