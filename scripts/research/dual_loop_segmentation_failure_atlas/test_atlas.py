"""Unit tests for the R0 segmentation-failure atlas."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.research.dual_loop_segmentation_failure_atlas.atlas import (
    _mechanism_tags,
    _validate_input_contract,
    assign_temporal_tracks,
    causal_temporal_probe,
    decode_packed_mask,
    encode_packed_mask,
    expansion_decision,
    primary_mechanism,
    select_gate_cases,
    spearman_rank_correlation,
    spatial_probe_mask,
)


class PackedMaskTest(unittest.TestCase):
    def test_round_trip_preserves_non_byte_aligned_shape(self) -> None:
        source = np.zeros((3, 5), dtype=bool)
        source[0, 1] = True
        source[2, 4] = True

        decoded = decode_packed_mask(encode_packed_mask(source), source.shape)

        np.testing.assert_array_equal(decoded, source)


class ProbeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.bands = {
            "lower_field_y_fraction": 0.5,
            "central_body_x_fraction": [0.25, 0.75],
            "central_body_y_min_fraction": 0.35,
            "upper_head_y_max_fraction": 0.35,
        }

    def test_spatial_probe_boundaries_are_deterministic(self) -> None:
        lower = spatial_probe_mask((10, 8), "LOWER_FIELD", self.bands)
        corridor = spatial_probe_mask((10, 8), "CENTRAL_BODY_CORRIDOR", self.bands)
        upper = spatial_probe_mask((10, 8), "UPPER_HEAD_BAND", self.bands)

        self.assertEqual(int(lower.sum()), 40)
        self.assertTrue(lower[5, 0])
        self.assertFalse(lower[4, 0])
        self.assertTrue(corridor[3, 2])
        self.assertTrue(corridor[9, 5])
        self.assertFalse(corridor[9, 6])
        self.assertEqual(int(upper.sum()), 32)

    def test_temporal_probes_are_causal_and_require_current(self) -> None:
        current = np.array([[True, True, False]])
        previous = np.array([[True, False, True]])
        previous_previous = np.array([[False, True, True]])

        two_of_three = causal_temporal_probe(
            current, previous, previous_previous, "CAUSAL_2_OF_3"
        )
        three = causal_temporal_probe(
            current, previous, previous_previous, "CAUSAL_3_CONSECUTIVE"
        )

        np.testing.assert_array_equal(two_of_three, np.array([[True, True, False]]))
        np.testing.assert_array_equal(three, np.array([[False, False, False]]))


class TemporalTrackingTest(unittest.TestCase):
    def test_tracks_only_across_adjacent_materialized_observations(self) -> None:
        first_mask = np.zeros((4, 4), dtype=bool)
        first_mask[1:3, 1:3] = True
        second_mask = first_mask.copy()
        records = [
            {
                "sequence_id": "sequence",
                "frame_id": 10,
                "predicted_class": "obstacle",
                "component_index": 0,
                "temporal_track_id": None,
                "previous_observation_iou": None,
                "next_observation_iou": None,
                "false_activation": True,
                "_mask": first_mask,
            },
            {
                "sequence_id": "sequence",
                "frame_id": 12,
                "predicted_class": "obstacle",
                "component_index": 0,
                "temporal_track_id": None,
                "previous_observation_iou": None,
                "next_observation_iou": None,
                "false_activation": True,
                "_mask": second_mask,
            },
        ]
        order = [
            {"sequence_id": "sequence", "frame_id": 10},
            {"sequence_id": "sequence", "frame_id": 12},
        ]

        assign_temporal_tracks(records, order, minimum_iou=0.1)

        self.assertEqual(records[0]["temporal_track_id"], records[1]["temporal_track_id"])
        self.assertEqual(records[0]["persistence_observations"], 2)
        self.assertEqual(records[0]["false_activation_run_observations"], 2)
        self.assertEqual(records[1]["previous_observation_iou"], 1.0)


class MechanismTest(unittest.TestCase):
    def test_mechanism_tags_are_nonexclusive_but_primary_is_stable(self) -> None:
        record = {
            "false_activation": True,
            "area_pixels": 40,
            "dominant_truth_class": "walkable",
            "boundary_proximity_fraction": 0.75,
            "yolo_overlapped_truth_hazard_intersection_pixels": 0,
            "nearest_yolo_box_distance_pixels": 100.0,
            "persistence_observations": 1,
            "false_activation_run_observations": 1,
            "top1_confidence_median": 0.8,
            "spatial_bands": [],
        }
        rules = {
            "small_fragment_max_area_pixels": 63,
            "large_component_min_area_pixels": 512,
            "boundary_dilation_min_fraction": 0.5,
            "yolo_attribution_gap_pixels": 3.0,
            "stable_minimum_frames": 3,
            "high_confidence_minimum": 0.65,
        }

        tags = _mechanism_tags(record, rules)

        self.assertIn("SMALL_FRAGMENT_NOISE", tags)
        self.assertIn("BOUNDARY_DILATION", tags)
        self.assertIn("TEMPORAL_FLICKER", tags)
        self.assertEqual(primary_mechanism(tags), "BOUNDARY_DILATION")


class ExpansionDecisionTest(unittest.TestCase):
    def test_spearman_supports_ties(self) -> None:
        self.assertAlmostEqual(
            spearman_rank_correlation([1.0, 2.0, 2.0, 4.0], [1.0, 3.0, 3.0, 4.0]),
            1.0,
        )

    def test_decision_tree_stops_before_gate_or_source_branches(self) -> None:
        self.assertEqual(
            expansion_decision(
                mechanisms_reproduced=False,
                aggregate_rank_correlation=0.9,
                minimum_rank_correlation=0.6,
                gating_axis="PARTIAL",
                source_dependent=True,
            ),
            "PILOT_NOT_GENERALIZABLE",
        )
        self.assertEqual(
            expansion_decision(
                mechanisms_reproduced=True,
                aggregate_rank_correlation=0.9,
                minimum_rank_correlation=0.6,
                gating_axis="PARTIAL",
                source_dependent=True,
            ),
            "GATING_PARTIAL",
        )
        self.assertEqual(
            expansion_decision(
                mechanisms_reproduced=True,
                aggregate_rank_correlation=0.9,
                minimum_rank_correlation=0.6,
                gating_axis="SUFFICIENT",
                source_dependent=False,
            ),
            "GATING_SUFFICIENT",
        )

    def test_input_contract_rejects_duplicate_frozen_frame(self) -> None:
        frame = {
            "view_row_id": "dev:one",
            "source_id": "source",
            "session_id": "session",
            "frame_id": 0,
            "rehearsal_role": "dev",
        }
        config = {
            "input_contract": {
                "allowed_roles": ["dev"],
                "expected_frame_count": 2,
                "expected_session_frame_counts": {"session": 2},
            }
        }
        with self.assertRaisesRegex(ValueError, "duplicate view_row_id"):
            _validate_input_contract(
                config=config,
                frame_rows=[frame, dict(frame)],
                component_rows=[],
            )

    def test_case_selection_is_deterministic(self) -> None:
        truth = np.array([[True, True, False, False]])
        frames = [
            {
                "view_row_id": "a",
                "session_id": "s",
                "role": "dev",
                "scene_bucket": "scene",
                "frame_id": 0,
                "residual_truth": truth,
            },
            {
                "view_row_id": "b",
                "session_id": "s",
                "role": "dev",
                "scene_bucket": "scene",
                "frame_id": 1,
                "residual_truth": truth,
            },
        ]
        baseline = [
            np.array([[True, True, True, True]]),
            np.array([[True, True, True, True]]),
        ]
        probes = [
            np.array([[True, True, False, False]]),
            np.array([[False, False, False, False]]),
        ]

        cases = select_gate_cases(
            frame_data=frames,
            baseline_masks=baseline,
            probe_masks=probes,
            success_minimum_recall_retention=0.9,
        )

        self.assertEqual(cases["success"]["view_row_id"], "a")
        self.assertEqual(cases["failure"]["view_row_id"], "b")


if __name__ == "__main__":
    unittest.main()
