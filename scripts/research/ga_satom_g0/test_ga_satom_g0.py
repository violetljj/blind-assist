from __future__ import annotations

import unittest

import numpy as np

from scripts.research.ga_satom_g0.core import (
    ANCHOR_ZONE_IDS,
    GroundAnchorPolicy,
    MeasurementFrame,
    TruthFrame,
    ZoneMeasurement,
    estimate_ground_anchor,
    evaluate_g0,
)
from scripts.research.ga_satom_g0.run_ga_satom_g0 import (
    validate_activation_roster,
    validate_expected_schedule,
)


def make_frame(parent: int, frame: int, *, height: float = 1.5, invalid_anchor_count: int = 0) -> MeasurementFrame:
    zones = []
    for row in range(8):
        for column in range(8):
            zone_id = f"r{row}c{column}"
            raw = np.asarray([(column - 3.5) * 0.03, 0.72 + row * 0.01, 0.65], dtype=np.float64)
            ray = raw / np.linalg.norm(raw)
            anchor_index = ANCHOR_ZONE_IDS.index(zone_id) if zone_id in ANCHOR_ZONE_IDS else -1
            invalid = 0 <= anchor_index < invalid_anchor_count
            zones.append(
                ZoneMeasurement(
                    zone_id=zone_id,
                    origin_rgb_m=np.zeros(3),
                    ray_rgb_unit=ray,
                    range_m=height / ray[1],
                    sigma_m=0.01,
                    status="INVALID" if invalid else "VALID",
                )
            )
    return MeasurementFrame(
        parent_id=f"fresh-parent-{parent:02d}", episode_id="episode-00",
        frame_id=f"frame-{frame:03d}", timestamp_ns=frame * 66_666_667,
        gravity_down_rgb_unit=np.asarray([0.0, 1.0, 0.0]), zones=tuple(zones),
    )


def make_truth(frame: MeasurementFrame, *, height: float = 1.5, non_ground: bool = False) -> TruthFrame:
    return TruthFrame(
        parent_id=frame.parent_id, episode_id=frame.episode_id, frame_id=frame.frame_id,
        reference_rgb_camera_height_m=height, reference_height_uncertainty_m=0.005,
        ground_labels={zone_id: "NON_GROUND" if non_ground else "GROUND" for zone_id in ANCHOR_ZONE_IDS},
    )


class GaSatomG0Test(unittest.TestCase):
    def test_frozen_anchor_budget_is_12_of_64(self) -> None:
        policy = GroundAnchorPolicy()
        policy.validate()
        self.assertEqual(len(policy.anchor_zone_ids), 12)
        self.assertEqual(policy.anchor_budget_fraction, 0.1875)

    def test_ground_height_estimate_uses_only_measurement(self) -> None:
        frame = make_frame(0, 0)
        estimate = estimate_ground_anchor(frame, GroundAnchorPolicy())
        self.assertTrue(estimate["valid"])
        self.assertAlmostEqual(estimate["height_m"], 1.5, places=6)
        altered_truth = make_truth(frame, height=1.9)
        self.assertAlmostEqual(estimate_ground_anchor(frame, GroundAnchorPolicy())["height_m"], 1.5, places=6)
        self.assertEqual(altered_truth.reference_rgb_camera_height_m, 1.9)

    def test_insufficient_anchor_support_fails_closed(self) -> None:
        estimate = estimate_ground_anchor(make_frame(0, 0, invalid_anchor_count=7), GroundAnchorPolicy())
        self.assertFalse(estimate["valid"])
        self.assertEqual(estimate["status"], "UNKNOWN_INSUFFICIENT_ANCHOR_ZONES")

    def test_synthetic_mechanics_can_pass_all_frozen_gates(self) -> None:
        measurements = [make_frame(parent, frame) for parent in range(8) for frame in range(3)]
        truth = [make_truth(frame) for frame in measurements]
        result = evaluate_g0(measurements, truth)
        self.assertTrue(result["passed"])
        self.assertFalse(result["causality"]["candidate_truth_access"])

    def test_false_ground_labels_cannot_improve_candidate_and_fail_gate(self) -> None:
        measurements = [make_frame(parent, frame) for parent in range(8) for frame in range(3)]
        truth = [make_truth(frame, non_ground=True) for frame in measurements]
        first = estimate_ground_anchor(measurements[0], GroundAnchorPolicy())
        result = evaluate_g0(measurements, truth)
        second = estimate_ground_anchor(measurements[0], GroundAnchorPolicy())
        self.assertEqual(first, second)
        self.assertFalse(result["passed"])
        self.assertEqual(result["parent_macro"]["false_ground_support_rate"], 1.0)

    def test_truth_requires_the_complete_frozen_anchor_label_set(self) -> None:
        frame = make_frame(0, 0)
        truth = make_truth(frame)
        truth.ground_labels.pop(ANCHOR_ZONE_IDS[0])
        with self.assertRaisesRegex(ValueError, "every frozen G0 anchor"):
            truth.validate()

    def test_schedule_rejects_a_silently_missing_time_slot(self) -> None:
        parents = [f"fresh-parent-{index:02d}" for index in range(8)]
        episodes = [f"episode-{index:02d}" for index in range(9)]
        measurements = []
        for parent_index in range(8):
            for episode_index, episode_id in enumerate(episodes):
                frame = make_frame(parent_index, 0)
                object.__setattr__(frame, "episode_id", episode_id)
                measurements.append(frame)
        truth = [make_truth(frame) for frame in measurements]
        validate_expected_schedule(measurements, truth, parents, episodes, ["frame-000"])
        with self.assertRaisesRegex(ValueError, "every frozen time slot"):
            validate_expected_schedule(measurements[:-1], truth[:-1], parents, episodes, ["frame-000"])

    def test_activation_roster_enforces_sites_strata_grid_and_occluders(self) -> None:
        parents = [f"fresh-parent-{index:02d}" for index in range(8)]
        episodes = [f"episode-{index:02d}" for index in range(9)]
        strata = [
            "matte_light_hard_floor", "dark_low_reflectance_floor_or_mat",
            "textured_or_carpet_floor", "specular_or_bright_tile_floor",
        ]
        activation = {
            "frozen_parent_ids": parents,
            "frozen_episode_ids": episodes,
            "parent_records": [
                {
                    "parent_id": parent_id, "site_id": f"site-{index % 2}",
                    "surface_stratum": strata[index % 4], "occluder_episode_id": episodes[index % 9],
                }
                for index, parent_id in enumerate(parents)
            ],
            "episode_records": [
                {
                    "episode_id": episode_id,
                    "reference_rgb_camera_height_m": height,
                    "rig_pitch_degrees": pitch,
                }
                for episode_id, (height, pitch) in zip(
                    episodes,
                    ((height, pitch) for height in (1.2, 1.5, 1.8) for pitch in (-5.0, 0.0, 5.0)),
                )
            ],
        }
        validate_activation_roster(activation)
        activation["parent_records"][0]["site_id"] = "site-1"
        activation["parent_records"][2]["site_id"] = "site-1"
        activation["parent_records"][4]["site_id"] = "site-1"
        activation["parent_records"][6]["site_id"] = "site-1"
        with self.assertRaisesRegex(ValueError, "at least two physical sites"):
            validate_activation_roster(activation)


if __name__ == "__main__":
    unittest.main()
