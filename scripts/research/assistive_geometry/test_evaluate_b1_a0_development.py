from __future__ import annotations

import copy
import unittest

from scripts.research.assistive_geometry.evaluate_b1_a0_development import (
    BANDS,
    HORIZONS,
    ROLE,
    aggregate,
    compute_seed_metrics,
    training_protocol_binding_for_seed,
    validate_observations,
)


GATES = {
    "known_coverage_min": 0.9,
    "ground_recovery_min": 0.9,
    "clearance_mae_m_max": 0.2,
    "false_clear_all_known_max": 0.08,
    "false_block_given_clear_max": 0.02,
    "temporal_clearance_delta_mae_m_max": 0.15,
    "geometry_transition_agreement_min": 0.9,
    "valid_to_unknown_rate_max": 0.1,
    "worst_parent_false_clear_all_known_max": 0.12,
}


def observations(seed: int = 17) -> list[dict]:
    rows: list[dict] = []
    for parent_index, parent in enumerate(("p0", "p1")):
        for sequence in range(4):
            bands = []
            for band_index, band in enumerate(BANDS):
                truth_clearance = 0.7 + 0.1 * band_index + 0.03 * sequence
                cells = []
                for horizon_index, horizon in enumerate(HORIZONS):
                    occupied = (parent_index + sequence + band_index + horizon_index) % 2 == 0
                    state = "OCCUPIED_OBSERVED" if occupied else "CLEAR_OBSERVED"
                    cells.append({"horizon_m": horizon, "truth_state": state, "predicted_state": state})
                bands.append(
                    {
                        "band": band,
                        "truth_clearance_valid": True,
                        "truth_clearance_m": truth_clearance,
                        "predicted_clearance_valid": True,
                        "predicted_clearance_m": truth_clearance + 0.02,
                        "cells": cells,
                    }
                )
            rows.append(
                {
                    "schema": "blindassist_assistive_geometry_b1_a0_development_frame_v1",
                    "seed": seed,
                    "data_role": ROLE,
                    "parent_id": parent,
                    "session_id": f"{parent}-session",
                    "sequence_index": sequence,
                    "orientation": "portrait" if parent_index == 0 else "landscape",
                    "environment": "indoor_arkitscenes",
                    "near_field": True,
                    "low_light_blur": False,
                    "truth_ground_valid": True,
                    "predicted_ground_valid": True,
                    "bands": bands,
                }
            )
    return rows


class DevelopmentEvaluatorTests(unittest.TestCase):
    def test_seed_29_uses_frozen_retry_protocol_without_changing_other_seeds(self) -> None:
        protocol = {
            "bindings": {
                "formal_train_protocol": {"path": "formal.json", "sha256": "FORMAL"},
                "seed_29_retry_protocol": {"path": "retry.json", "sha256": "RETRY"},
            }
        }
        self.assertEqual(training_protocol_binding_for_seed(protocol, 17)["sha256"], "FORMAL")
        self.assertEqual(training_protocol_binding_for_seed(protocol, 29)["sha256"], "RETRY")
        self.assertEqual(training_protocol_binding_for_seed(protocol, 43)["sha256"], "FORMAL")

    def test_perfect_fixture_passes_all_task_gates(self) -> None:
        result = compute_seed_metrics(observations(), 17, GATES)
        self.assertTrue(all(result["gates"].values()))
        self.assertAlmostEqual(result["clearance_mae_m"]["value"], 0.02)
        self.assertEqual(result["pooled"]["unknown_truth_excluded"], 0)

    def test_unknown_truth_is_excluded_not_negative(self) -> None:
        rows = observations()
        rows[0]["bands"][0]["cells"][0]["truth_state"] = "UNKNOWN"
        rows[0]["bands"][0]["cells"][0]["predicted_state"] = "CLEAR_OBSERVED"
        result = compute_seed_metrics(rows, 17, GATES)
        self.assertEqual(result["pooled"]["unknown_truth_excluded"], 1)
        self.assertEqual(result["pooled"]["known_coverage"], 1.0)

    def test_missing_predicted_clearance_reduces_coverage(self) -> None:
        rows = observations()
        for row in rows[:2]:
            for band in row["bands"]:
                band["predicted_clearance_valid"] = False
                band["predicted_clearance_m"] = None
        result = compute_seed_metrics(rows, 17, GATES)
        self.assertFalse(result["gates"]["clearance_known_coverage"])
        self.assertEqual(result["clearance_known_coverage"]["value"], 0.75)

    def test_aggregate_never_selects_a_seed(self) -> None:
        metrics = []
        for seed in (17, 29, 43):
            metrics.append(compute_seed_metrics(observations(seed), seed, GATES))
        result = aggregate(metrics)
        self.assertIsNone(result["selected_seed"])
        self.assertTrue(result["overall_pass"])

    def test_schema_rejects_independent_validity_value_drift(self) -> None:
        rows = copy.deepcopy(observations())
        rows[0]["bands"][0]["predicted_clearance_valid"] = False
        with self.assertRaises(Exception):
            validate_observations(rows, 17)


if __name__ == "__main__":
    unittest.main()
