"""Mutation tests for the TARO O0R source-adapter contract lock."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from scripts.research.taro_o0r_source_adapter.validate_taro_o0r_source_adapter_contract import (
    CONTRACT_PATH,
    validate_contract,
    validate_repository,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


class TaroO0RSourceAdapterContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = json.loads((REPO_ROOT / CONTRACT_PATH).read_text(encoding="utf-8"))

    def expect_invalid(self, mutate) -> None:
        payload = copy.deepcopy(self.payload)
        mutate(payload)
        with self.assertRaises(ValueError):
            validate_contract(payload)

    def test_repository_contract_is_valid(self) -> None:
        result = validate_repository(REPO_ROOT, copy.deepcopy(self.payload))
        self.assertEqual(result["status"], "VALID")
        self.assertEqual(result["scientific_status"], "NOT_RUN")
        self.assertEqual(len(result["semantic_seams_frozen"]), 5)

    def test_schema_and_protocol_identity_are_frozen(self) -> None:
        self.expect_invalid(lambda value: value.__setitem__("schema", "changed"))
        self.expect_invalid(lambda value: value.__setitem__("protocol_id", "changed"))

    def test_role_count_cannot_shrink(self) -> None:
        self.expect_invalid(
            lambda value: value["selection_contract"]["roles"]["O0R_EVAL_CANDIDATE"].pop()
        )

    def test_visit_cannot_cross_roles(self) -> None:
        def mutate(value):
            value["selection_contract"]["roles"]["O0R_EVAL_CANDIDATE"][0]["visit_id"] = (
                value["selection_contract"]["roles"]["ADAPTER_FIT"][0]["visit_id"]
            )

        self.expect_invalid(mutate)

    def test_rank_cannot_be_rewritten(self) -> None:
        self.expect_invalid(
            lambda value: value["selection_contract"]["roles"]["ADAPTER_FIT"][0].__setitem__(
                "selection_rank_sha256", "0" * 64
            )
        )

    def test_model_output_cannot_enter_fit_or_selection(self) -> None:
        self.expect_invalid(
            lambda value: value["role_contract"]["ADAPTER_FIT"].__setitem__(
                "model_outputs_forbidden", False
            )
        )
        self.expect_invalid(
            lambda value: value["selection_contract"]["invariants"].__setitem__(
                "model_output_influence", True
            )
        )

    def test_query_truth_cannot_become_model_generated(self) -> None:
        self.expect_invalid(
            lambda value: value["query_contract"].__setitem__("truth_source", "teacher output")
        )

    def test_scale_truth_only_and_candidate_relative_stages_cannot_collapse(self) -> None:
        self.expect_invalid(
            lambda value: value["factor_truth_contract"]["SCALE"].__setitem__(
                "truth_only_value_kind", "DEPTHART_RELATIVE_CORRECTION"
            )
        )
        self.expect_invalid(
            lambda value: value["factor_truth_contract"]["SCALE"].__setitem__(
                "candidate_relative_correction_after_truth_only_result", False
            )
        )

    def test_pose_watermark_must_include_the_right_bracket(self) -> None:
        self.expect_invalid(
            lambda value: value["frame_receipt_adapter"].__setitem__(
                "max_source_timestamp_rule", "CURRENT_FRAME_ONLY"
            )
        )
        self.expect_invalid(
            lambda value: value["truth_only_admission_gates"].__setitem__(
                "max_source_timestamp_includes_right_pose_bracket", False
            )
        )

    def test_nine_queries_require_nine_query_bound_receipts(self) -> None:
        self.expect_invalid(
            lambda value: value["frame_receipt_adapter"].__setitem__(
                "query_receipts_per_physical_frame", 1
            )
        )
        self.expect_invalid(
            lambda value: value["query_contract"].__setitem__(
                "query_grid_order", "UNSPECIFIED"
            )
        )

    def test_o0r_receipt_cannot_claim_missing_p0_fields(self) -> None:
        self.expect_invalid(
            lambda value: value["frame_receipt_adapter"].__setitem__(
                "p0_frame_receipt_projection", "FULLY_VALID"
            )
        )
        self.expect_invalid(
            lambda value: value["implementation_interface_contract"].__setitem__(
                "p0_full_frame_receipt_claim", True
            )
        )

    def test_legacy_three_band_interfaces_cannot_become_the_taro_runtime(self) -> None:
        self.expect_invalid(
            lambda value: value["implementation_interface_contract"].__setitem__(
                "legacy_geometry_r2_reducer_runtime_role", "TARO_QUERY_REDUCER"
            )
        )
        self.expect_invalid(
            lambda value: value["implementation_interface_contract"].__setitem__(
                "source_io_in_implementation_lock", True
            )
        )

    def test_constant_uncertainty_cannot_be_invented(self) -> None:
        self.expect_invalid(
            lambda value: value["factor_truth_contract"].__setitem__(
                "invented_constant_uncertainty", True
            )
        )
        self.expect_invalid(
            lambda value: value["factor_truth_contract"]["uncertainty_fit_cells"].__setitem__(
                "parent_aggregation", "POOLED_SAMPLES"
            )
        )
        self.expect_invalid(
            lambda value: value["factor_truth_contract"]["canonicalization"].__setitem__(
                "float_decimal_places", 6
            )
        )

    def test_registration_and_boundary_semantics_cannot_drift(self) -> None:
        self.expect_invalid(
            lambda value: value["registration_and_geometry_contract"].__setitem__(
                "lowres_to_highres_scale_xy", [8.0, 8.0]
            )
        )
        self.expect_invalid(
            lambda value: value["registration_and_geometry_contract"].__setitem__(
                "boundary_sign", "UNSPECIFIED"
            )
        )

    def test_checkpoint_and_training_are_frozen(self) -> None:
        self.expect_invalid(
            lambda value: value["baseline_contract"].__setitem__(
                "checkpoint_sha256", "0" * 64
            )
        )
        self.expect_invalid(
            lambda value: value["baseline_contract"].__setitem__("training_steps", 1)
        )

    def test_k_cannot_become_a_factorial_factor(self) -> None:
        self.expect_invalid(
            lambda value: value["factorial_contract"]["arms"].append("K")
        )

    def test_truth_frontdoors_cannot_be_lowered(self) -> None:
        self.expect_invalid(
            lambda value: value["truth_only_admission_gates"].__setitem__(
                "minimum_evaluable_o0r_parents", 11
            )
        )
        self.expect_invalid(
            lambda value: value["truth_only_admission_gates"].__setitem__(
                "minimum_complete_factor_query_fraction_within_source_eligible_frames", 0.99
            )
        )
        self.expect_invalid(
            lambda value: value["truth_only_admission_gates"].__setitem__(
                "query_receipts_required_per_source_eligible_frame", 8
            )
        )

    def test_primary_effect_and_guardrails_cannot_drift(self) -> None:
        self.expect_invalid(
            lambda value: value["o0r_metrics_and_gates"].__setitem__(
                "minimum_meaningful_effect_m", 0.0
            )
        )
        self.expect_invalid(
            lambda value: value["o0r_metrics_and_gates"]["guardrails"].__setitem__(
                "all_unknown_forbidden", False
            )
        )

    def test_o0m_and_future_roots_cannot_collide(self) -> None:
        self.expect_invalid(
            lambda value: value["artifact_isolation"].__setitem__(
                "future_o0r_evidence_root",
                value["artifact_isolation"]["historical_o0m_root"],
            )
        )

    def test_execution_authority_cannot_expand(self) -> None:
        self.expect_invalid(
            lambda value: value["execution_authority"].__setitem__(
                "source_payload_download", True
            )
        )

    def test_successor_is_unique(self) -> None:
        self.expect_invalid(lambda value: value.__setitem__("unique_successor", "O0R_RUN"))


if __name__ == "__main__":
    unittest.main()
