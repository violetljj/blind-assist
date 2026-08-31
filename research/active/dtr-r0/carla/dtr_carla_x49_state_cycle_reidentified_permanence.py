"""X49 state-cycle reidentified permanence across parent fragmentation.

X49 starts from X45.  When an active risk parent disappears, its propagated
state may be reidentified as any current authorized rigid-dynamic track that
closes X45's inherited position-and-velocity cycle and still enters the issued
route.  Otherwise the belief follows X47's depth-aware occlusion rule.

C22 is consumed posthoc synthetic Development and cannot confirm X49.
"""

from __future__ import annotations

import copy
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import dtr_carla_x24_plan_adherent_predictor as x24  # noqa: E402
import dtr_carla_x31_ambiguity_preserving_transport_predictor as x31  # noqa: E402
import dtr_carla_x45_causal_state_cycle_credential as x45  # noqa: E402
import dtr_carla_x47_depth_free_space_permanence_release as x47  # noqa: E402
import dtr_carla_x48_evidence_updated_object_permanence as x48  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X49_STATE_CYCLE_REIDENTIFIED_PERMANENCE"
ARM_X49 = "X49_ISSUED_PLAN_STATE_CYCLE_REIDENTIFIED_PERMANENCE"


def fixed_constants() -> dict[str, Any]:
    return {
        **x48.fixed_constants(),
        "representation": "EVIDENCE_UPDATED_PERMANENCE_WITH_STATE_CYCLE_REIDENTIFICATION",
        "reidentification_rule": (
            "PROPAGATED_BELIEF_MAY_BIND_ANY_CURRENT_AUTHORIZED_RIGID_DYNAMIC_"
            "TRACK_THAT_CLOSES_INHERITED_X24_POSITION_ASSOCIATION_AND_POSITIVE_"
            "VELOCITY_DOT_AND_STILL_ENTERS_ISSUED_ROUTE"
        ),
        "reidentification_parent_identity_required": False,
        "reidentification_class_identity_required": False,
        "reidentification_position_distance_m": x24.ASSOCIATION_DISTANCE_M,
        "reidentification_threshold_source": "INHERITED_X24_ASSOCIATION_DISTANCE",
        "permanence_fixed_timeout_seconds": None,
        "new_numeric_threshold_added": False,
        "detector_threshold_change": False,
        "route_threshold_change": False,
        "score_threshold_change": False,
    }


def apply_state_reidentified_episode(
    episode: Any,
    value: dict[str, Any],
    calibration: Any,
) -> dict[str, Any]:
    active: dict[str, list[dict[str, Any]]] = {}
    previous_mode: str | None = None
    receipt_cache: dict[Path, dict[str, Any]] = {}
    reidentification_frames = 0
    reidentification_parent_changes = 0
    occluded_continuation_frames = 0
    state_cycle_releases = 0
    observed_conflict_releases = 0
    propagated_route_exit_releases = 0
    depth_free_space_releases = 0

    for observation, frame in zip(
        episode.observations, value["frames"], strict=True
    ):
        x24.require(
            int(observation.sample_index) == int(frame["sample_index"])
            and abs(float(observation.time_s) - float(frame["time_s"]))
            <= x31.EPSILON,
            "x49_observation_frame_alignment",
        )
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
        current_authorized = x48.authorized_dynamic_rows(frame["tracks"])
        current_parent_ids = {x48.parent_id(row) for row in frame["tracks"]}
        arm["x49_object_permanence_used"] = False
        arm["x49_object_permanence_update_modes"] = []

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
            state_cycle_releases += int(bool(active))
            active = {}
            continue

        continued: list[dict[str, Any]] = []
        entries: list[float] = []
        modes: list[str] = []
        used_current_ids: set[str] = set()
        for belief_parent, beliefs in active.items():
            propagated = [
                x48.propagate_x48(row, float(observation.time_s))
                for row in beliefs
            ]
            matches = [
                row
                for row in current_authorized
                if str(row["track_id"]) not in used_current_ids
                and any(
                    x45.closes_state_cycle(belief, row) for belief in propagated
                )
            ]
            matched_entries, entry_values = x48.route_entries(matches, segments)
            if matched_entries:
                reidentification_frames += 1
                reidentification_parent_changes += int(
                    any(x48.parent_id(row) != belief_parent for row in matched_entries)
                )
                for row in matched_entries:
                    row["x49_update_mode"] = "STATE_CYCLE_REIDENTIFICATION"
                    row["x48_belief_state_time_s"] = float(observation.time_s)
                    used_current_ids.add(str(row["track_id"]))
                continued.extend(matched_entries)
                entries.extend(entry_values)
                modes.append("STATE_CYCLE_REIDENTIFICATION")
                continue

            if belief_parent in current_parent_ids:
                observed_conflict_releases += 1
                continue
            eligible, propagated_entries = x48.route_entries(propagated, segments)
            if not eligible:
                propagated_route_exit_releases += 1
            elif x47.visible_free_space(
                observation, eligible, calibration, episode.route_frame
            ):
                depth_free_space_releases += 1
            else:
                continued.extend(eligible)
                entries.extend(propagated_entries)
                modes.append("OCCLUDED_KINEMATIC_BELIEF")

        if not continued:
            active = {}
            continue
        occluded_continuation_frames += int("OCCLUDED_KINEMATIC_BELIEF" in modes)
        for row in continued:
            track_id = str(row["track_id"])
            if track_id not in tracks:
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
                "x49_object_permanence_used": True,
                "x49_object_permanence_update_modes": sorted(set(modes)),
            }
        )
        active = x48.group_rows(x48.state_rows(continued, observation.time_s))

    value["arms"][ARM_X49] = value["arms"].pop(x45.ARM_X45)
    value["diagnostics"]["x49_route_mode_counts"] = value["diagnostics"].pop(
        "x45_route_mode_counts"
    )
    value["diagnostics"].update(
        {
            "x49_reidentification_frames": reidentification_frames,
            "x49_reidentification_parent_changes": reidentification_parent_changes,
            "x49_occluded_continuation_frames": occluded_continuation_frames,
            "x49_state_cycle_releases": state_cycle_releases,
            "x49_observed_conflict_releases": observed_conflict_releases,
            "x49_propagated_route_exit_releases": propagated_route_exit_releases,
            "x49_depth_free_space_releases": depth_free_space_releases,
        }
    )
    for frame in value["frames"]:
        frame["arms"][ARM_X49] = frame["arms"].pop(x45.ARM_X45)
    return value


def predict_episode(
    episode: Any,
    candidate_values: Sequence[Mapping[str, Any]],
    calibration: Any,
) -> dict[str, Any]:
    return apply_state_reidentified_episode(
        episode,
        x45.predict_episode(episode, candidate_values, calibration),
        calibration,
    )


def self_check() -> dict[str, Any]:
    inherited = x48.self_check()
    prior = {
        "position_forward_m": 4.0,
        "position_right_m": 0.0,
        "velocity_forward_mps": -2.0,
        "velocity_right_mps": 0.1,
    }
    reidentified = {
        "position_forward_m": 4.5,
        "position_right_m": 0.1,
        "velocity_forward_mps": -1.0,
        "velocity_right_mps": 0.1,
    }
    x24.require(
        x45.closes_state_cycle(prior, reidentified),
        "x49_parent_independent_state_cycle_reidentification",
    )
    return {
        "status": "X49_STATE_CYCLE_REIDENTIFIED_PERMANENCE_FALSIFIER_MET",
        "x48_structural_status": inherited["status"],
        "parent_identity_required": False,
        "class_identity_required": False,
        "position_threshold_inherited_from_x24": True,
        "depth_free_space_release_preserved": True,
        "fixed_timeout_added": False,
        "new_numeric_threshold_added": False,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), sort_keys=True))
