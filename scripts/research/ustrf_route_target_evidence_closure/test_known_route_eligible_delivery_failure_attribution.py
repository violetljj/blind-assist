#!/usr/bin/env python3
"""Mutation and integration tests for delivery-failure attribution R1."""

from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR))

from known_route_eligible_delivery_failure_attribution import (  # noqa: E402
    AttributionContractError,
    CANDIDATES,
    _qualifying_support,
    atomic_write_json,
    assert_blind,
    load_json,
    validate_outputs,
)


class KnownRouteEligibleDeliveryFailureAttributionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = Path(__file__).resolve().parents[3]
        cls.config_path = (
            cls.repo
            / "configs"
            / (
                "ustrf_route_target_known_route_eligible_delivery_"
                "failure_attribution_r1.json"
            )
        )
        cls.config = load_json(cls.config_path)
        cls.output_root = cls.repo / cls.config["outputs"]["root"]

    def test_canonical_outputs_recompute_exactly(self) -> None:
        receipt = validate_outputs(
            self.repo,
            self.config_path,
            self.output_root / self.config["outputs"]["terminal_receipt"],
        )
        self.assertEqual("VALID", receipt["status"])
        self.assertEqual(36, receipt["candidate_event_cells_recomputed"])
        self.assertEqual(0, receipt["unexplained_gap_count"])

    def test_truth_field_is_rejected_from_blind_ledger(self) -> None:
        with self.assertRaisesRegex(
            AttributionContractError, "blind_trace_forbidden_field"
        ):
            assert_blind({"frame_id": 1, "event_id": "forbidden"})

    def test_c2_global_risk_run_can_precede_target_attribution(self) -> None:
        frames = [
            {
                "source_id": "source",
                "sequence_id": "sequence",
                "frame_id": frame_id,
                "source_capture_timestamp_ns": frame_id,
                "state_reset_before_frame": frame_id == 0,
                "route_known": True,
                "active_relation_track_ids": track_ids,
                "baseline_active_keys_after": [
                    {"local_key": 1, "opening_frame": 1}
                ],
            }
            for frame_id, track_ids in ((0, [7]), (1, [9]))
        ]
        qualification = _qualifying_support(
            CANDIDATES[1],
            frames,
            "target-event",
            {},
            0.3,
            2,
        )
        self.assertEqual(1, qualification["qualification_frame"])
        self.assertEqual(0, qualification["support_start_frame"])

    def test_exact_counts_and_claim_boundary(self) -> None:
        terminal = load_json(
            self.output_root / self.config["outputs"]["terminal_receipt"]
        )
        self.assertEqual(
            self.config["evaluation"]["expected_aggregate_counts"],
            terminal["aggregate_label_counts"],
        )
        self.assertTrue(terminal["attribution_gate"]["passed"])
        self.assertTrue(
            all(
                value is False
                for value in terminal["claim_boundary"].values()
            )
        )

    def test_terminal_mutation_is_not_canonical(self) -> None:
        terminal_path = (
            self.output_root / self.config["outputs"]["terminal_receipt"]
        )
        terminal = load_json(terminal_path)
        mutated = copy.deepcopy(terminal)
        mutated["aggregate_label_counts"]["unexplained_gap"] = 1
        with tempfile.TemporaryDirectory() as temporary:
            mutated_path = Path(temporary) / "terminal.json"
            atomic_write_json(mutated_path, mutated)
            with patch(
                (
                    "known_route_eligible_delivery_failure_attribution."
                    "build_terminal_from_persisted_blind"
                ),
                return_value=terminal,
            ):
                with self.assertRaisesRegex(
                    AttributionContractError,
                    "terminal_receipt_not_exact_canonical_recomputation",
                ):
                    validate_outputs(
                        self.repo, self.config_path, mutated_path
                    )


if __name__ == "__main__":
    unittest.main()
