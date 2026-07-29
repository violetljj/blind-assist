from __future__ import annotations

import copy
from pathlib import Path
import statistics
import tempfile
import unittest

from scripts.research.egomotion_compensated_looming.periodic_self_motion_counterfactual_r2 import (
    motion_component_localization_r0 as producer,
)
from scripts.research.egomotion_compensated_looming.periodic_self_motion_counterfactual_r2 import (
    validate_motion_component_localization_independent_r0 as validator,
)


ROOT = Path(__file__).resolve().parents[4]
CONTRACT = (
    ROOT
    / "docs/research/rcle/"
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "QMS_R1_CLEAN_MOTION_COMPONENT_LOCALIZATION_R0_"
    "CONTRACT_2026-07-29.json"
)
LOCK = (
    ROOT
    / "docs/research/rcle/"
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "QMS_R1_CLEAN_MOTION_COMPONENT_LOCALIZATION_R0_"
    "IDENTITY_LOCK_2026-07-29.json"
)
ACTIVATION = (
    ROOT
    / "docs/research/rcle/"
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "QMS_R1_CLEAN_MOTION_COMPONENT_LOCALIZATION_R0_"
    "STAGE_1_ACTIVATION_2026-07-29.json"
)


def _static_identity_and_poses() -> tuple[dict, list[dict]]:
    lock = validator.load_json(LOCK)
    identity = next(
        item
        for item in lock["identities"]
        if item["stage_number"] == 1 and item["motion"] == "STATIC"
    )
    trajectory = validator.load_json(ROOT / validator.TRAJECTORY_MANIFEST)[
        identity["block"]
    ]
    return identity, validator._poses(trajectory, "STATIC")


def _cell(expansion: float | None, residual: float = 0.2) -> dict:
    return {
        "evaluable": expansion is not None,
        "expansion": expansion,
        "fit_residual_pixels_per_frame": (
            residual if expansion is not None else None
        ),
    }


def _rows(
    cell_values: list[float] | None = None,
) -> tuple[list[dict], dict, list[dict]]:
    identity, poses = _static_identity_and_poses()
    values = cell_values or [-4.0, -3.0, -2.0, -1.0, 0.01, 1.0, 2.0, 3.0, 4.0]
    signed = float(statistics.median(values))
    absolute = float(statistics.median(abs(value) for value in values))
    raw_cells = [_cell(value) for value in values]
    compensated_cells = [_cell(value) for value in values]
    rows = []
    streak = 0
    for index in range(validator.PAIR_COUNT):
        positive = signed > validator.THRESHOLD
        streak = streak + 1 if positive else 0
        rows.append(
            {
                **{
                    key: identity[key]
                    for key in (
                        "sequence_id",
                        "cluster_id",
                        "block",
                        "ordinal",
                        "stage_number",
                        "role",
                        "arm",
                        "motion",
                    )
                },
                "pair_index": index,
                "evaluable": True,
                "compensated_expansion_median_per_s": signed,
                "raw_expansion_median_per_s": signed,
                "compensated_abs_expansion_median_per_s": absolute,
                "raw_abs_expansion_median_per_s": absolute,
                "compensated_three_pair_trigger": streak >= 3,
                "common_cell_count": 9,
                "raw_track_count": 90,
                "compensated_track_count": 90,
                "angular_speed_rad_s": 0.0,
                "translation_speed_m_s": 0.0,
                "cell_diagnostics": {
                    "raw_cells": copy.deepcopy(raw_cells),
                    "compensated_cells": copy.deepcopy(compensated_cells),
                    "common_cell_indices": list(range(9)),
                },
            }
        )
    return rows, identity, poses


class MotionComponentLocalizationTests(unittest.TestCase):
    def test_frozen_contract_identity_and_activation_are_valid(self) -> None:
        validator.validate_contract(ROOT, CONTRACT)
        lock, expected = validator.validate_identity_lock(
            ROOT, LOCK, CONTRACT
        )
        validator.validate_stage_1_activation(
            ROOT, ACTIVATION, CONTRACT, LOCK, lock
        )
        self.assertEqual(len(expected), 32)
        self.assertEqual(
            sum(item["stage_number"] == 1 for item in expected.values()), 16
        )
        self.assertEqual(
            sum(item["stage_number"] == 2 for item in expected.values()), 16
        )
        excluded = validator._exclusion_union(
            ROOT, lock["exclusion_sources"]
        )
        for field, old_values in excluded.items():
            self.assertFalse(
                validator._collect(
                    lock["seeds"] + lock["identities"], field
                )
                & old_values,
                field,
            )

    def test_pose_arms_preserve_timestamps_and_exact_semantics(self) -> None:
        trajectories = validator.load_json(ROOT / validator.TRAJECTORY_MANIFEST)
        identity_matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        zero = [0.0, 0.0, 0.0]
        for block, trajectory in trajectories.items():
            source = trajectory["poses"]
            for motion in validator.MOTIONS:
                poses = validator._poses(trajectory, motion)
                self.assertEqual(
                    [item["timestamp_s"] for item in poses],
                    [item["timestamp_s"] for item in source],
                )
                for derived, original in zip(poses, source, strict=True):
                    self.assertEqual(
                        derived["rotation_matrix"],
                        (
                            original["rotation_matrix"]
                            if motion in {"ROTATION_ONLY", "FULL_6DOF"}
                            else identity_matrix
                        ),
                    )
                    self.assertEqual(
                        derived["translation_m"],
                        (
                            original["translation_m"]
                            if motion in {"TRANSLATION_ONLY", "FULL_6DOF"}
                            else zero
                        ),
                    )
            self.assertEqual(
                validator._trajectory_hash(trajectory, "FULL_6DOF"),
                trajectory["periodic_pose_sha256"],
                block,
            )

    def test_signed_response_and_absolute_leakage_are_distinct(self) -> None:
        rows, identity, poses = _rows()
        summary = validator.reduce_rows(rows, identity, poses)
        self.assertEqual(summary["compensated_signed_p50"], 0.01)
        self.assertEqual(summary["compensated_absolute_p50"], 2.0)
        self.assertEqual(summary["trigger_count"], 0)

    def test_abstention_resets_strict_three_pair_streak(self) -> None:
        rows, identity, poses = _rows([0.02] * 9)
        row = rows[2]
        row["evaluable"] = False
        row["compensated_expansion_median_per_s"] = None
        row["raw_expansion_median_per_s"] = None
        row["compensated_abs_expansion_median_per_s"] = None
        row["raw_abs_expansion_median_per_s"] = None
        row["compensated_three_pair_trigger"] = False
        row["common_cell_count"] = 4
        row["cell_diagnostics"]["common_cell_indices"] = list(range(4))
        for index in range(4, 9):
            row["cell_diagnostics"]["raw_cells"][index] = _cell(None)
            row["cell_diagnostics"]["compensated_cells"][index] = _cell(None)
        rows[3]["compensated_three_pair_trigger"] = False
        rows[4]["compensated_three_pair_trigger"] = False
        summary = validator.reduce_rows(rows, identity, poses)
        self.assertEqual(
            summary["trigger_count"], validator.PAIR_COUNT - 5
        )

    def test_forged_pair_scalar_is_rejected_from_cell_primitives(self) -> None:
        rows, identity, poses = _rows()
        rows[0]["compensated_expansion_median_per_s"] = 123.0
        with self.assertRaisesRegex(
            validator.InvalidIndependentMotionComponent, "PAIR_SIGNED"
        ):
            validator.reduce_rows(rows, identity, poses)

    def test_frozen_routing_uses_three_of_four_cluster_directions(self) -> None:
        _, expected = validator.validate_identity_lock(
            ROOT, LOCK, CONTRACT
        )
        summaries = {}
        for identity in expected.values():
            if identity["stage_number"] != 1:
                continue
            block_index = validator.BLOCKS.index(identity["block"])
            positive = block_index < 3
            motion = identity["motion"]
            summaries[identity["sequence_id"]] = {
                "compensated_absolute_p90": {
                    "STATIC": 0.10,
                    "ROTATION_ONLY": 0.20 if positive else 0.05,
                    "TRANSLATION_ONLY": 9.0,
                    "FULL_6DOF": 9.0,
                }[motion],
                "compensated_signed_p90": {
                    "STATIC": 100.0,
                    "ROTATION_ONLY": 0.20,
                    "TRANSLATION_ONLY": 0.30 if positive else 0.10,
                    "FULL_6DOF": 0.40 if positive else 0.05,
                }[motion],
            }
        result = validator.analyze_stage(expected, summaries, 1)
        for contrast in result["routing_direction_summary"].values():
            self.assertEqual(contrast["positive_count"], 3)
            self.assertTrue(contrast["opens_stage_2"])

    def test_stage_2_refuses_without_independent_decision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                producer.InvalidMotionComponent,
                "STAGE_2_DECISION_REQUIRED",
            ):
                producer.run_stage(
                    LOCK,
                    Path(directory) / "stage2",
                    stage_number=2,
                )
            self.assertFalse((Path(directory) / "stage2").exists())


if __name__ == "__main__":
    unittest.main()
