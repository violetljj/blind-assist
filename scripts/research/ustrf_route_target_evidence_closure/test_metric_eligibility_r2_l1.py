"""Mutation tests for the R2-L1 eligibility validator."""

from __future__ import annotations

import copy
import unittest
from pathlib import Path

from contract import ContractError, load_json, sha256_file
from metric_eligibility import (
    BoundLoader,
    METRICS,
    _base_event_row,
    _classify_event,
    _continuous_post_clear_identity_ms,
    _require_matching_terminal_clear,
    validate_config,
)
from validate_metric_eligibility_r2_l1 import validate_materialized


class MetricEligibilityR2L1Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = Path(__file__).resolve().parents[3]
        cls.config_path = (
            cls.repo / "configs/ustrf_route_target_metric_eligibility_r2_l1.json"
        )
        cls.config = load_json(cls.config_path)
        root = cls.repo / cls.config["outputs"]["root"]
        cls.mask = load_json(root / cls.config["outputs"]["event_mask"])
        cls.receipt = load_json(root / cls.config["outputs"]["denominator_receipt"])

    def validate(
        self,
        *,
        mask: dict | None = None,
        receipt: dict | None = None,
        config: dict | None = None,
    ) -> list[str]:
        return validate_materialized(
            config or self.config,
            mask or self.mask,
            receipt or self.receipt,
            repo=self.repo,
            config_path=self.config_path,
        )

    def test_frozen_artifacts_are_valid(self) -> None:
        checks = self.validate()
        self.assertIn("zero_over_zero_cannot_pass_or_fail", checks)
        self.assertIn("pre_clear_events_excluded_from_clearance", checks)

    def test_candidate_output_read_is_rejected(self) -> None:
        changed = copy.deepcopy(self.receipt)
        changed["candidate_outputs_read"] = ["forbidden.json"]
        with self.assertRaisesRegex(ContractError, "candidate output access"):
            self.validate(receipt=changed)

    def test_candidate_execution_is_rejected(self) -> None:
        changed = copy.deepcopy(self.receipt)
        changed["candidate_outputs_executed"] = True
        with self.assertRaisesRegex(ContractError, "candidate output access"):
            self.validate(receipt=changed)

    def test_zero_denominator_cannot_be_evaluable(self) -> None:
        changed = copy.deepcopy(self.receipt)
        changed["metrics"]["event_recall"]["support_status"] = "evaluable_powered"
        with self.assertRaisesRegex(ContractError, "empty denominator"):
            self.validate(receipt=changed)

    def test_zero_denominator_cannot_pass(self) -> None:
        changed = copy.deepcopy(self.receipt)
        changed["metrics"]["event_recall"]["gate_result"] = "pass"
        with self.assertRaisesRegex(ContractError, "gate result"):
            self.validate(receipt=changed)

    def test_preclear_event_cannot_enter_clearance(self) -> None:
        event = next(
            row
            for row in self.mask["events"]
            if row["anchors"]["truth_terminal_clear_frame"] is None
        )
        original = copy.deepcopy(event["metrics"]["clearance"])
        event["metrics"]["clearance"] = {"classification": "eligible", "reasons": []}
        try:
            with self.assertRaisesRegex(ContractError, "pre-clear"):
                self.validate()
        finally:
            event["metrics"]["clearance"] = original

    def test_identity_loss_before_clear_is_preclear_not_survival_censor(self) -> None:
        row = _base_event_row(
            dataset_group="test",
            provenance_family="test",
            source_id="source",
            sequence_id="sequence",
            event_id="event",
            truth_status="test",
            critical=False,
            onset_frame=0,
            alertable_start_frame=1,
            clear_frame=5,
            end_frame=10,
            raw_exclusion_reasons=[],
        )
        _classify_event(
            row,
            identity_onset_to_alertable=True,
            route_onset_to_alertable=True,
            identity_through_clear=False,
            complete_active_episode=False,
            critical_interval=None,
            critical_reasons=["not_critical_event"],
            post_clear_followup_ms=2000.0,
            post_clear_identity_followup_ms=None,
            lifecycle_identity_loss=True,
        )
        self.assertEqual(
            "not_evaluable_pre_clear",
            row["metrics"]["clearance"]["censor_state"],
        )
        self.assertEqual(
            "not_evaluable_pre_clear",
            row["metrics"]["regeneration"]["censor_state"],
        )
        self.assertFalse(
            row["observability"]["same_person_truth_terminal_clear_observed"]
        )

    def test_terminal_clear_observability_is_separate_from_followup(self) -> None:
        row = _base_event_row(
            dataset_group="test",
            provenance_family="test",
            source_id="source",
            sequence_id="sequence",
            event_id="event",
            truth_status="test",
            critical=False,
            onset_frame=0,
            alertable_start_frame=1,
            clear_frame=5,
            end_frame=6,
            raw_exclusion_reasons=[],
        )
        _classify_event(
            row,
            identity_onset_to_alertable=True,
            route_onset_to_alertable=True,
            identity_through_clear=True,
            complete_active_episode=True,
            critical_interval=None,
            critical_reasons=["not_critical_event"],
            post_clear_followup_ms=500.0,
            post_clear_identity_followup_ms=500.0,
        )
        self.assertTrue(
            row["observability"]["same_person_truth_terminal_clear_observed"]
        )
        self.assertEqual(
            "ineligible", row["metrics"]["clearance"]["classification"]
        )

    def test_repeat_identity_loss_is_not_administrative_censor(self) -> None:
        repeat_censors = self.receipt["censor_counts"]["repeat"]
        self.assertNotIn("right_censored_administrative", repeat_censors)
        self.assertGreater(
            repeat_censors.get("right_censored_identity_loss", 0), 0
        )

    def test_post_clear_identity_stops_at_timestamp_gap(self) -> None:
        observed = _continuous_post_clear_identity_ms(
            clear_frame=5,
            clear_timestamp_ns=1_000_000_000,
            identity_frames={
                6: 1_100_000_000,
                7: 3_500_000_000,
                8: 3_600_000_000,
            },
            maximum_gap_ns=1_000_000_000,
        )
        self.assertEqual(100.0, observed)

    def test_post_clear_identity_reappearance_after_frame_gap_is_ignored(self) -> None:
        observed = _continuous_post_clear_identity_ms(
            clear_frame=5,
            clear_timestamp_ns=1_000_000_000,
            identity_frames={7: 1_100_000_000, 8: 1_200_000_000},
            maximum_gap_ns=1_000_000_000,
        )
        self.assertIsNone(observed)

    def test_liloc_terminal_clear_anchor_must_match_proxy(self) -> None:
        with self.assertRaisesRegex(ContractError, "terminal clear anchor drifted"):
            _require_matching_terminal_clear(
                10, 11, event_key=("source", "event")
            )

    def test_event_must_have_exact_metric_roster(self) -> None:
        event = self.mask["events"][0]
        removed = event["metrics"].pop(METRICS[0])
        try:
            with self.assertRaisesRegex(ContractError, "metric roster"):
                self.validate()
        finally:
            event["metrics"][METRICS[0]] = removed

    def test_ineligible_metric_requires_reason(self) -> None:
        event = self.mask["events"][0]
        reasons = event["metrics"]["event_recall"]["reasons"]
        original = list(reasons)
        reasons.clear()
        try:
            with self.assertRaisesRegex(ContractError, "exclusion lacks reason"):
                self.validate()
        finally:
            reasons.extend(original)

    def test_unknown_reason_is_rejected(self) -> None:
        event = self.mask["events"][0]
        reasons = event["metrics"]["event_recall"]["reasons"]
        original = list(reasons)
        reasons[:] = sorted(original + ["not_in_taxonomy"])
        try:
            with self.assertRaisesRegex(ContractError, "unknown reason"):
                self.validate()
        finally:
            reasons[:] = original

    def test_duplicate_event_id_is_rejected(self) -> None:
        original = self.mask["events"][1]["unit_id"]
        self.mask["events"][1]["unit_id"] = self.mask["events"][0]["unit_id"]
        try:
            with self.assertRaisesRegex(ContractError, "unique and sorted"):
                self.validate()
        finally:
            self.mask["events"][1]["unit_id"] = original

    def test_metric_reason_counts_must_recompute(self) -> None:
        changed = copy.deepcopy(self.receipt)
        changed["reason_counts"]["event_recall"]["alertable_deadline_not_frozen"] -= 1
        with self.assertRaisesRegex(ContractError, "reason counts"):
            self.validate(receipt=changed)

    def test_raw_exclusion_counts_must_recompute(self) -> None:
        changed = copy.deepcopy(self.receipt)
        key = next(iter(changed["raw_exclusion_reason_counts"]))
        changed["raw_exclusion_reason_counts"][key] -= 1
        with self.assertRaisesRegex(ContractError, "raw exclusion"):
            self.validate(receipt=changed)

    def test_censor_counts_must_recompute(self) -> None:
        changed = copy.deepcopy(self.receipt)
        key = next(iter(changed["censor_counts"]["clearance"]))
        changed["censor_counts"]["clearance"][key] -= 1
        with self.assertRaisesRegex(ContractError, "censor counts"):
            self.validate(receipt=changed)

    def test_event_denominator_must_recompute(self) -> None:
        changed = copy.deepcopy(self.receipt)
        changed["metrics"]["clearance"]["denominator"] += 1
        with self.assertRaisesRegex(ContractError, "clearance denominator"):
            self.validate(receipt=changed)

    def test_repeat_truth_pool_is_not_candidate_denominator(self) -> None:
        changed = copy.deepcopy(self.receipt)
        changed["metrics"]["repeat"]["denominator"] = changed["metrics"]["repeat"][
            "truth_observation_pool_count"
        ]
        with self.assertRaisesRegex(ContractError, "repeat preoutput pool"):
            self.validate(receipt=changed)

    def test_negative_exposure_duration_must_equal_interval(self) -> None:
        interval = self.mask["negative_exposure_intervals"][0]
        interval["duration_ns"] += 1
        try:
            with self.assertRaisesRegex(
                ContractError,
                "negative exposure (intervals do not rebuild|interval)",
            ):
                self.validate()
        finally:
            interval["duration_ns"] -= 1

    def test_negative_exposure_pair_primary_reason_must_recompute(self) -> None:
        pair = next(
            row
            for row in self.mask["negative_exposure_pair_audit"]
            if row["classification"] == "ineligible"
            and len(row["exclusion_reasons"]) > 1
        )
        original = pair["primary_exclusion_reason"]
        pair["primary_exclusion_reason"] = pair["exclusion_reasons"][-1]
        try:
            with self.assertRaisesRegex(ContractError, "primary exclusion"):
                self.validate()
        finally:
            pair["primary_exclusion_reason"] = original

    def test_pair_audit_covers_every_sequence_adjacency(self) -> None:
        self.assertEqual(
            len(self.mask["preoutput_frame_ledger"])
            - len(self.mask["preoutput_frame_masks"]),
            len(self.mask["negative_exposure_pair_audit"]),
        )

    def test_liloc_cross_window_pair_is_explicitly_excluded(self) -> None:
        self.assertTrue(
            any(
                pair["provenance_family"] == "lilocbench"
                and pair["end_frame_id"] == pair["start_frame_id"] + 1
                and "route_relevant_person_truth_incomplete_at_endpoint"
                in pair["exclusion_reasons"]
                for pair in self.mask["negative_exposure_pair_audit"]
            )
        )

    def test_frame_denominator_must_recompute(self) -> None:
        changed = copy.deepcopy(self.receipt)
        changed["metrics"]["unknown_or_stale_alert"]["denominator"] -= 1
        with self.assertRaisesRegex(ContractError, "frame denominator"):
            self.validate(receipt=changed)

    def test_explicit_frame_ledger_route_state_is_validated(self) -> None:
        frame = self.mask["preoutput_frame_ledger"][0]
        original = frame["route_validity_state"]
        frame["route_validity_state"] = "stale_without_route_state"
        try:
            with self.assertRaisesRegex(ContractError, "frame ledger row"):
                self.validate()
        finally:
            frame["route_validity_state"] = original

    def test_terminal_clear_zero_denominator_stays_not_evaluable(self) -> None:
        changed = copy.deepcopy(self.receipt)
        changed["terminal_clear_observability"]["support_status"] = "evaluable_powered"
        with self.assertRaisesRegex(ContractError, "terminal-clear"):
            self.validate(receipt=changed)

    def test_l1_routing_must_recompute(self) -> None:
        changed = copy.deepcopy(self.receipt)
        changed["l1_routing"]["l1_exploratory_eligible_metrics"] = []
        with self.assertRaisesRegex(ContractError, "does not recompute"):
            self.validate(receipt=changed)

    def test_l1_status_must_follow_frozen_floor(self) -> None:
        changed = copy.deepcopy(self.receipt)
        changed["metrics"]["critical_miss"][
            "l1_readiness"
        ] = "L0_ENGINEERING_DIAGNOSTIC"
        changed["l1_routing"]["l1_exploratory_eligible_metrics"].remove(
            "critical_miss"
        )
        changed["l1_routing"]["l0_engineering_diagnostic_metrics"].append(
            "critical_miss"
        )
        with self.assertRaisesRegex(ContractError, "L1 floor"):
            self.validate(receipt=changed)

    def test_mask_hash_binding_rejects_mutation(self) -> None:
        event = self.mask["events"][0]
        original = event["truth_status"]
        event["truth_status"] = original + "_mutated"
        try:
            with self.assertRaisesRegex(ContractError, "mask hash binding"):
                self.validate()
        finally:
            event["truth_status"] = original

    def test_critical_boolean_without_interval_is_rejected(self) -> None:
        event = next(
            row
            for row in self.mask["events"]
            if row["metrics"]["critical_miss"]["classification"] == "eligible"
        )
        original = copy.deepcopy(event["metrics"]["critical_miss"]["details"])
        event["metrics"]["critical_miss"]["details"] = {}
        try:
            with self.assertRaisesRegex(ContractError, "without frozen interval"):
                self.validate()
        finally:
            event["metrics"]["critical_miss"]["details"] = original

    def test_authority_above_l1_is_rejected(self) -> None:
        changed = copy.deepcopy(self.receipt)
        changed["authority"]["selection"] = True
        with self.assertRaisesRegex(ContractError, "authority selection opened"):
            self.validate(receipt=changed)

    def test_forbidden_candidate_dependency_path_is_rejected(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["candidate_blind_dependency_manifest"][0][
            "path"
        ] = "artifacts.local/evidence/candidate-profile/result.json"
        with self.assertRaisesRegex(ContractError, "forbidden output"):
            BoundLoader(changed, self.repo)

    def test_parent_standard_hash_drift_is_rejected(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["parent_standard"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ContractError, "parent evidence maturity"):
            validate_config(changed, repo=self.repo)

    def test_parent_l1_floor_cannot_be_weakened(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["l1_readiness"]["clearance"]["minimum_denominator"] = 4
        with self.assertRaisesRegex(ContractError, "inherit parent V2"):
            validate_config(changed, repo=self.repo)

    def test_candidate_blind_dependency_policy_cannot_open_globs(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["dependency_policy"][
            "directory_scan_or_glob_for_inputs_allowed"
        ] = True
        with self.assertRaisesRegex(ContractError, "dependency policy drifted"):
            validate_config(changed, repo=self.repo)

    def test_bound_loader_rejects_nonallowlisted_read(self) -> None:
        parent_path = self.repo / self.config["parent_standard"]["path"]
        loader = BoundLoader(
            {
                "candidate_blind_dependency_manifest": [
                    {
                        "id": "parent",
                        "path": self.config["parent_standard"]["path"],
                        "sha256": sha256_file(parent_path),
                        "assertions": {},
                    }
                ],
                "dependency_policy": {"forbidden_path_fragments": []},
            },
            self.repo,
        )
        with self.assertRaisesRegex(ContractError, "non-allowlisted"):
            loader.load("candidate_result")


if __name__ == "__main__":
    unittest.main()
