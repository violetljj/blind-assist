#!/usr/bin/env python3
"""Focused tests for the diagnostic-only B1-A0 failure anatomy."""

from __future__ import annotations

import copy
import unittest

from scripts.research.assistive_geometry.analyze_b1_a0_failure_anatomy import (
    AnatomyError,
    compute_seed_anatomy,
    validate_cross_seed_order,
    validate_package_metadata,
)


def make_rows(seed: int, *, prediction: str = "OCCUPIED_OBSERVED") -> list[dict]:
    rows: list[dict] = []
    for sequence in range(3):
        bands = []
        for band_index, band_name in enumerate(("left", "center", "right")):
            truth_clearance = 1.7 + 0.1 * band_index
            predicted_clearance = 0.5 + 0.1 * band_index
            cells = []
            for horizon in (1.0, 1.5, 2.0):
                truth_state = "CLEAR_OBSERVED" if truth_clearance >= horizon else "OCCUPIED_OBSERVED"
                predicted_state = prediction if horizon <= 1.5 else "OCCUPIED_OBSERVED"
                cells.append({"horizon_m": horizon, "truth_state": truth_state, "predicted_state": predicted_state})
            bands.append(
                {
                    "band": band_name,
                    "truth_clearance_valid": True,
                    "truth_clearance_m": truth_clearance,
                    "predicted_clearance_valid": True,
                    "predicted_clearance_m": predicted_clearance,
                    "cells": cells,
                }
            )
        rows.append(
            {
                "schema": "blindassist_assistive_geometry_b1_a0_development_frame_v1",
                "seed": seed,
                "data_role": "DEVELOPMENT_SELECTION",
                "parent_id": "p0",
                "session_id": "s0",
                "sequence_index": sequence,
                "frame_id": f"f{sequence}",
                "orientation": "landscape",
                "environment": "indoor_arkitscenes",
                "near_field": True,
                "low_light_blur": False,
                "truth_ground_valid": True,
                "predicted_ground_valid": True,
                "bands": bands,
            }
        )
    return rows


def protocol() -> dict:
    return {
        "accepted_observation_protocol_sha256": "OBS_PROTOCOL",
        "observation_sha256_by_seed": {"17": "S17", "29": "S29", "43": "S43"},
    }


def package() -> dict:
    return {
        "schema": "blindassist_assistive_geometry_b1_a0_development_evaluation_package_v1",
        "data_role": "DEVELOPMENT_SELECTION",
        "development_content_opened": True,
        "development_calibration_content_opened": False,
        "confirmation_content_opened": False,
        "evaluation_protocol_sha256": "OBS_PROTOCOL",
        "seed_runs": [
            {"seed": 17, "observation_count": 1200, "observations_sha256": "S17"},
            {"seed": 29, "observation_count": 1200, "observations_sha256": "S29"},
            {"seed": 43, "observation_count": 1200, "observations_sha256": "S43"},
        ],
    }


class FailureAnatomyTests(unittest.TestCase):
    def test_false_blocks_are_clearance_crossings_not_head_output(self) -> None:
        result, _ = compute_seed_anatomy(make_rows(17), 17, (0.1, 0.25, 0.5), 0.1)
        anatomy = result["false_block_anatomy"]
        self.assertEqual(anatomy["assistive_occupancy_or_task_head"], "NOT_APPLICABLE_A0_ASSISTIVE_HEADS_NOT_READ")
        self.assertEqual(anatomy["exclusive_source_decomposition"]["clearance_threshold_crossing"]["count"], 18)
        self.assertEqual(anatomy["boundary_margin_strata"]["gt_0.50m"]["false_blocks"], 9)

    def test_unknown_truth_is_excluded_and_never_counted_as_false_block(self) -> None:
        rows = make_rows(17)
        rows[0]["bands"][0]["cells"][0]["truth_state"] = "UNKNOWN"
        result, internal = compute_seed_anatomy(rows, 17, (0.1, 0.25, 0.5), 0.1)
        state = result["state_distribution"]
        self.assertEqual(state["truth_all_cells"]["counts"]["UNKNOWN"], 1)
        self.assertEqual(state["false_blocks"], 17)
        self.assertIn("never negative", state["unknown_policy"])
        unknown_key = (*(
            rows[0][name]
            for name in ("parent_id", "session_id", "sequence_index", "frame_id", "orientation")
        ), "left", 1.0)
        self.assertNotIn(unknown_key, internal["false_block_by_key"])

    def test_cross_seed_order_drift_fails_closed(self) -> None:
        rows = {seed: make_rows(seed) for seed in (17, 29, 43)}
        rows[29] = [rows[29][1], rows[29][0], rows[29][2]]
        with self.assertRaises(AnatomyError) as caught:
            validate_cross_seed_order(rows, ("p0",))
        self.assertEqual(caught.exception.code, "CROSS_SEED_FRAME_ORDER_DRIFT")

    def test_package_seed_order_and_sha_are_frozen(self) -> None:
        validate_package_metadata(package(), protocol())
        wrong_order = copy.deepcopy(package())
        wrong_order["seed_runs"][0], wrong_order["seed_runs"][1] = wrong_order["seed_runs"][1], wrong_order["seed_runs"][0]
        with self.assertRaises(AnatomyError) as order_error:
            validate_package_metadata(wrong_order, protocol())
        self.assertEqual(order_error.exception.code, "SEED_ORDER_DRIFT")
        wrong_sha = copy.deepcopy(package())
        wrong_sha["seed_runs"][0]["observations_sha256"] = "DRIFT"
        with self.assertRaises(AnatomyError) as sha_error:
            validate_package_metadata(wrong_sha, protocol())
        self.assertEqual(sha_error.exception.code, "OBSERVATION_SHA_BINDING_DRIFT")

    def test_truth_must_match_across_all_seeds(self) -> None:
        rows = {seed: make_rows(seed) for seed in (17, 29, 43)}
        rows[43][0]["bands"][0]["truth_clearance_m"] = 1.6
        with self.assertRaises(AnatomyError) as caught:
            validate_cross_seed_order(rows, ("p0",))
        self.assertEqual(caught.exception.code, "CROSS_SEED_TRUTH_DRIFT")


if __name__ == "__main__":
    unittest.main()
