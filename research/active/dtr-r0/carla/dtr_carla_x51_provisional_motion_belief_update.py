"""X51 provisional-motion updates for an already authorized risk belief.

X51 starts from X45.  A current track that has not yet accumulated enough
repeated occupancy translations to originate risk may nevertheless update an
existing authorized belief when its current non-zero lattice translation
closes the inherited position-and-velocity state cycle with that belief.  The
provisional observation can continue authority, but can never create it.

When no provisional observation closes the cycle, X51 keeps X47's propagated
belief and depth-free-space release.  C22 is consumed posthoc synthetic
Development and cannot confirm X51.
"""

from __future__ import annotations

import copy
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import dtr_carla_x24_plan_adherent_predictor as x24  # noqa: E402
import dtr_carla_x27_occupancy_authority_predictor as x27  # noqa: E402
import dtr_carla_x31_ambiguity_preserving_transport_predictor as x31  # noqa: E402
import dtr_carla_x45_causal_state_cycle_credential as x45  # noqa: E402
import dtr_carla_x47_depth_free_space_permanence_release as x47  # noqa: E402
import dtr_carla_x48_evidence_updated_object_permanence as x48  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X51_PROVISIONAL_MOTION_BELIEF_UPDATE"
ARM_X51 = "X51_ISSUED_PLAN_PROVISIONAL_MOTION_BELIEF_UPDATE"
PROVISIONAL_DISPOSITION = "AUTHORIZED_BELIEF_PROVISIONAL_MOTION_UPDATE"


def fixed_constants() -> dict[str, Any]:
    return {
        **x47.fixed_constants(),
        "representation": "AUTHORIZED_BELIEF_WITH_PROVISIONAL_MOTION_UPDATE",
        "provisional_admission_rule": (
            "EXISTING_AUTHORIZED_BELIEF_REQUIRED_AND_CURRENT_MEASURED_FOOTPRINT_"
            "HAS_NONZERO_OCCUPANCY_TRANSLATION"
        ),
        "provisional_update_rule": (
            "INHERITED_X24_POSITION_ASSOCIATION_AND_POSITIVE_VELOCITY_DOT_"
            "WITH_PROPAGATED_AUTHORIZED_BELIEF"
        ),
        "provisional_authority_origination": False,
        "provisional_velocity_source": "CURRENT_OCCUPANCY_LATTICE_TRANSLATION",
        "provisional_position_distance_m": x24.ASSOCIATION_DISTANCE_M,
        "provisional_position_threshold_source": "INHERITED_X24_ASSOCIATION_DISTANCE",
        "depth_free_space_release": "INHERITED_X47_CENTER_RAY_RULE",
        "new_numeric_threshold_added": False,
        "detector_threshold_change": False,
        "route_threshold_change": False,
        "score_threshold_change": False,
    }


def provisional_row(
    row: Mapping[str, Any], sample_period_s: float
) -> dict[str, Any] | None:
    shift = row.get("world_lattice_shift_cells")
    if (
        row.get("disposition") != "MEASURED"
        or row.get("support_footprint_mode") != "MEASURED_CONVEX_CELL_HULL"
        or "footprint_xy" not in row
        or not isinstance(shift, Sequence)
        or len(shift) != 2
        or (int(shift[0]) == 0 and int(shift[1]) == 0)
    ):
        return None
    x24.require(sample_period_s > 0.0, "x51_sample_period")
    value = copy.deepcopy(dict(row))
    value["velocity_forward_mps"] = (
        float(shift[0]) * x27.LATTICE_CELL_SIZE_M / sample_period_s
    )
    value["velocity_right_mps"] = (
        float(shift[1]) * x27.LATTICE_CELL_SIZE_M / sample_period_s
    )
    value["x51_provisional_motion_witness"] = True
    return value


def continued_row(
    witness: Mapping[str, Any], belief: Mapping[str, Any], now_s: float
) -> dict[str, Any]:
    value = copy.deepcopy(dict(witness))
    value["x51_observed_parent_track_id"] = x48.parent_id(witness)
    value["parent_track_id"] = x48.parent_id(belief)
    value["motion_authority"] = x27.RIGID_DYNAMIC
    value["risk_eligible"] = True
    value["disposition"] = PROVISIONAL_DISPOSITION
    value["x48_belief_state_time_s"] = float(now_s)
    value["x51_authority_originated"] = False
    return value


def apply_provisional_motion_episode(
    episode: Any,
    value: dict[str, Any],
    calibration: Any,
) -> dict[str, Any]:
    value = copy.deepcopy(value)
    active: dict[str, list[dict[str, Any]]] = {}
    previous_mode: str | None = None
    receipt_cache: dict[Path, dict[str, Any]] = {}
    provisional_update_frames = 0
    provisional_parent_changes = 0
    occluded_continuation_frames = 0
    observed_conflict_releases = 0
    route_exit_releases = 0
    depth_free_space_releases = 0

    observations = episode.observations
    for ordinal, (observation, frame) in enumerate(
        zip(observations, value["frames"], strict=True)
    ):
        neighbour = observations[ordinal - 1] if ordinal else observations[ordinal + 1]
        sample_period_s = abs(float(observation.time_s) - float(neighbour.time_s))
        wearer_position, wearer_velocity = x24.wearer_anchor_state(
            observation, episode.route_frame
        )
        receipt = x24.load_receipt(observation, receipt_cache)
        selection = x24.route.select_route(
            receipt,
            session_id=observation.navigation_session_id,
            now_s=observation.time_s,
            wearer_position_xy=wearer_position,
            wearer_velocity_xy=wearer_velocity,
            previous_mode=previous_mode,
        )
        previous_mode = selection.mode
        segments = x24.route.build_route_segments(
            selection,
            receipt=receipt,
            now_s=observation.time_s,
            wearer_position_xy=wearer_position,
            wearer_velocity_xy=wearer_velocity,
        )
        arm = frame["arms"][x45.ARM_X45]
        tracks = {str(row["track_id"]): row for row in frame["tracks"]}
        arm["x51_provisional_motion_update_used"] = False
        arm["x51_provisional_motion_update_track_ids"] = []

        if bool(arm.get("route_risk")):
            confirmed = [
                tracks[str(track_id)]
                for track_id in arm.get("confirmed_risk_track_ids", [])
                if str(track_id) in tracks
            ]
            active = x48.group_rows(
                x48.state_rows(
                    x48.authorized_dynamic_rows(confirmed), observation.time_s
                )
            )
            continue

        if bool(arm.get("x45_state_cycle_suppressed", False)):
            active = {}
            continue

        provisional = [
            candidate
            for row in frame["tracks"]
            if (candidate := provisional_row(row, sample_period_s)) is not None
        ]
        current_valid_parents = {
            x48.parent_id(row) for row in provisional
        }
        continued: list[dict[str, Any]] = []
        entries: list[float] = []
        used_witness_ids: set[str] = set()

        for belief_parent, beliefs in active.items():
            propagated = [
                x48.propagate_x48(row, float(observation.time_s))
                for row in beliefs
            ]
            matches = [
                row
                for row in provisional
                if str(row["track_id"]) not in used_witness_ids
                and any(x45.closes_state_cycle(belief, row) for belief in propagated)
            ]
            matched, matched_entries = x48.route_entries(matches, segments)
            if matched:
                provisional_update_frames += 1
                provisional_parent_changes += int(
                    any(x48.parent_id(row) != belief_parent for row in matched)
                )
                for row in matched:
                    source_belief = min(
                        propagated,
                        key=lambda belief: x45.position_distance(belief, row),
                    )
                    updated = continued_row(
                        row, source_belief, float(observation.time_s)
                    )
                    continued.append(updated)
                    used_witness_ids.add(str(row["track_id"]))
                entries.extend(matched_entries)
                continue

            if belief_parent in current_valid_parents:
                observed_conflict_releases += 1
                continue
            eligible, propagated_entries = x48.route_entries(propagated, segments)
            if not eligible:
                route_exit_releases += 1
            elif x47.visible_free_space(
                observation, eligible, calibration, episode.route_frame
            ):
                depth_free_space_releases += 1
            else:
                continued.extend(eligible)
                entries.extend(propagated_entries)
                occluded_continuation_frames += 1

        if not continued:
            active = {}
            continue
        for row in continued:
            if str(row["track_id"]) not in tracks:
                frame["tracks"].append(row)
                frame["risk_eligible_tracks"] = int(frame["risk_eligible_tracks"]) + 1
        track_ids = sorted(str(row["track_id"]) for row in continued)
        parent_ids = sorted({x48.parent_id(row) for row in continued})
        arm.update(
            {
                "route_risk": True,
                "minimum_entry_s": min(entries),
                "candidate_risk_track_ids": track_ids,
                "confirmed_risk_track_ids": track_ids,
                "candidate_risk_parent_track_ids": parent_ids,
                "confirmed_risk_parent_track_ids": parent_ids,
                "x51_provisional_motion_update_used": any(
                    row.get("disposition") == PROVISIONAL_DISPOSITION
                    for row in continued
                ),
                "x51_provisional_motion_update_track_ids": sorted(
                    str(row["track_id"])
                    for row in continued
                    if row.get("disposition") == PROVISIONAL_DISPOSITION
                ),
            }
        )
        active = x48.group_rows(x48.state_rows(continued, observation.time_s))

    value["arms"][ARM_X51] = value["arms"].pop(x45.ARM_X45)
    value["diagnostics"]["x51_route_mode_counts"] = value["diagnostics"].pop(
        "x45_route_mode_counts"
    )
    value["diagnostics"].update(
        {
            "x51_provisional_update_frames": provisional_update_frames,
            "x51_provisional_parent_changes": provisional_parent_changes,
            "x51_occluded_continuation_frames": occluded_continuation_frames,
            "x51_observed_conflict_releases": observed_conflict_releases,
            "x51_route_exit_releases": route_exit_releases,
            "x51_depth_free_space_releases": depth_free_space_releases,
        }
    )
    for frame in value["frames"]:
        frame["arms"][ARM_X51] = frame["arms"].pop(x45.ARM_X45)
    return value


def predict_episode(
    episode: Any,
    candidate_values: Sequence[Mapping[str, Any]],
    calibration: Any,
) -> dict[str, Any]:
    return apply_provisional_motion_episode(
        episode,
        x45.predict_episode(episode, candidate_values, calibration),
        calibration,
    )


def self_check() -> dict[str, Any]:
    inherited = x47.self_check()
    witness = provisional_row(
        {
            "disposition": "MEASURED",
            "support_footprint_mode": "MEASURED_CONVEX_CELL_HULL",
            "world_lattice_shift_cells": [-1, 0],
            "footprint_xy": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        },
        0.1,
    )
    x24.require(
        witness is not None
        and float(witness["velocity_forward_mps"]) < 0.0
        and abs(float(witness["velocity_right_mps"])) <= x31.EPSILON,
        "x51_current_lattice_motion_witness",
    )
    return {
        "status": "X51_PROVISIONAL_MOTION_BELIEF_UPDATE_FALSIFIER_MET",
        "x47_structural_status": inherited["status"],
        "authority_origination_allowed": False,
        "existing_authorized_belief_required": True,
        "position_threshold_inherited_from_x24": True,
        "depth_free_space_release_preserved": True,
        "new_numeric_threshold_added": False,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), sort_keys=True))
