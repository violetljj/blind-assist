"""X46 evidence-terminated object permanence for issued-route risk.

X46 preserves X45 and carries an already authorized rigid-dynamic risk while
its parent is unobserved and its propagated footprint still enters the issued
route.  The belief terminates on same-parent reobservation without X45 risk or
when its propagated footprint has no route entry.  There is no fixed timeout.

C22 is consumed posthoc synthetic Development and cannot confirm X46.
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
import dtr_carla_x27_occupancy_authority_predictor as x27  # noqa: E402
import dtr_carla_x30_adaptive_surface_interval_predictor as x30  # noqa: E402
import dtr_carla_x31_ambiguity_preserving_transport_predictor as x31  # noqa: E402
import dtr_carla_x45_causal_state_cycle_credential as x45  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X46_EVIDENCE_TERMINATED_OBJECT_PERMANENCE"
ARM_X46 = "X46_ISSUED_PLAN_EVIDENCE_TERMINATED_OBJECT_PERMANENCE"
BELIEF_DISPOSITION = "OBJECT_PERMANENCE_BELIEF_HOLD"


def fixed_constants() -> dict[str, Any]:
    return {
        **x45.fixed_constants(),
        "representation": "CAUSAL_STATE_CYCLE_WITH_EVIDENCE_TERMINATED_PERMANENCE",
        "permanence_admission_rule": (
            "ONLY_PREVIOUS_X45_CONFIRMED_RIGID_DYNAMIC_RISK_MAY_SEED_BELIEF"
        ),
        "permanence_continuation_rule": (
            "PARENT_UNOBSERVED_AND_PROPAGATED_FOOTPRINT_STILL_ENTERS_ISSUED_ROUTE"
        ),
        "permanence_release_rule": (
            "SAME_PARENT_REOBSERVED_WITHOUT_X45_RISK_OR_NO_PROPAGATED_ROUTE_ENTRY"
        ),
        "permanence_fixed_timeout_seconds": None,
        "permanence_class_rule": "CLASS_INDEPENDENT",
        "new_numeric_threshold_added": False,
        "detector_threshold_change": False,
        "route_threshold_change": False,
        "score_threshold_change": False,
    }


def parent_id(row: Mapping[str, Any]) -> str:
    return str(row.get("parent_track_id") or row["track_id"])


def propagate_belief(row: Mapping[str, Any], now_s: float) -> dict[str, Any]:
    value = copy.deepcopy(dict(row))
    prior_s = float(value.get("x46_belief_state_time_s", now_s))
    dt = max(0.0, float(now_s) - prior_s)
    delta_forward = float(value["velocity_forward_mps"]) * dt
    delta_right = float(value["velocity_right_mps"]) * dt
    value["position_forward_m"] = float(value["position_forward_m"]) + delta_forward
    value["position_right_m"] = float(value["position_right_m"]) + delta_right
    value["footprint_xy"] = [
        [float(point[0]) + delta_forward, float(point[1]) + delta_right]
        for point in value["footprint_xy"]
    ]
    value["disposition"] = BELIEF_DISPOSITION
    value["motion_authority"] = x27.RIGID_DYNAMIC
    value["risk_eligible"] = True
    value["depth_grid_support"] = None
    value["evidence_age_s"] = float(value.get("evidence_age_s") or 0.0) + dt
    value["x46_object_permanence_belief"] = True
    value["x46_belief_source_track_id"] = str(
        value.get("x46_belief_source_track_id") or value["track_id"]
    )
    value["x46_belief_state_time_s"] = float(now_s)
    return value


def group_beliefs(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        output[parent_id(row)].append(copy.deepcopy(dict(row)))
    return dict(output)


def apply_permanence_episode(episode: Any, value: dict[str, Any]) -> dict[str, Any]:
    active: dict[str, list[dict[str, Any]]] = {}
    previous_mode: str | None = None
    receipt_cache: dict[Path, dict[str, Any]] = {}
    continuation_frames = 0
    continued_tracks = 0
    reobservation_releases = 0
    route_exit_releases = 0

    for observation, frame in zip(
        episode.observations, value["frames"], strict=True
    ):
        x24.require(
            int(observation.sample_index) == int(frame["sample_index"])
            and abs(float(observation.time_s) - float(frame["time_s"]))
            <= x31.EPSILON,
            "x46_observation_frame_alignment",
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
        current_parents = {parent_id(row) for row in frame["tracks"]}
        arm["x46_object_permanence_used"] = False
        arm["x46_object_permanence_track_ids"] = []
        arm["x46_object_permanence_parent_track_ids"] = []

        if bool(arm.get("route_risk")):
            confirmed = [
                tracks[str(track_id)]
                for track_id in arm.get("confirmed_risk_track_ids", [])
                if str(track_id) in tracks
            ]
            authorized = [
                row
                for row in confirmed
                if row.get("motion_authority") == x27.RIGID_DYNAMIC
                and bool(row.get("risk_eligible", False))
                and "footprint_xy" in row
            ]
            seeded = []
            for row in authorized:
                belief = copy.deepcopy(dict(row))
                belief["x46_belief_state_time_s"] = float(observation.time_s)
                seeded.append(belief)
            active = group_beliefs(seeded)
            continue

        carriers: list[dict[str, Any]] = []
        entries: list[float] = []
        for belief_parent, beliefs in active.items():
            if belief_parent in current_parents:
                reobservation_releases += 1
                continue
            parent_carriers: list[dict[str, Any]] = []
            parent_entries: list[float] = []
            for belief in beliefs:
                propagated = propagate_belief(belief, float(observation.time_s))
                entry = x30.first_contact_interval_entry_s(
                    propagated["footprint_xy"],
                    (
                        float(propagated["velocity_forward_mps"]),
                        float(propagated["velocity_right_mps"]),
                    ),
                    segments,
                )
                if entry is not None:
                    parent_carriers.append(propagated)
                    parent_entries.append(float(entry))
            if parent_carriers:
                carriers.extend(parent_carriers)
                entries.extend(parent_entries)
            else:
                route_exit_releases += 1

        if carriers:
            continuation_frames += 1
            continued_tracks += len(carriers)
            for carrier in carriers:
                track_id = str(carrier["track_id"])
                x24.require(track_id not in tracks, "x46_belief_track_id_collision")
                frame["tracks"].append(carrier)
                frame["risk_eligible_tracks"] = int(frame["risk_eligible_tracks"]) + 1
            track_ids = sorted(str(row["track_id"]) for row in carriers)
            parent_ids = sorted({parent_id(row) for row in carriers})
            arm.update(
                {
                    "route_risk": True,
                    "minimum_entry_s": min(entries),
                    "candidate_risk_track_ids": track_ids,
                    "confirmed_risk_track_ids": track_ids,
                    "candidate_risk_parent_track_ids": parent_ids,
                    "confirmed_risk_parent_track_ids": parent_ids,
                    "x46_object_permanence_used": True,
                    "x46_object_permanence_track_ids": track_ids,
                    "x46_object_permanence_parent_track_ids": parent_ids,
                }
            )
            active = group_beliefs(carriers)
        else:
            active = {}

    value["arms"][ARM_X46] = value["arms"].pop(x45.ARM_X45)
    value["diagnostics"]["x46_route_mode_counts"] = value["diagnostics"].pop(
        "x45_route_mode_counts"
    )
    value["diagnostics"]["x46_permanence_continuation_frames"] = (
        continuation_frames
    )
    value["diagnostics"]["x46_permanence_continued_tracks"] = continued_tracks
    value["diagnostics"]["x46_reobservation_releases"] = reobservation_releases
    value["diagnostics"]["x46_route_exit_releases"] = route_exit_releases
    for frame in value["frames"]:
        frame["arms"][ARM_X46] = frame["arms"].pop(x45.ARM_X45)
    return value


def predict_episode(
    episode: Any,
    candidate_values: Sequence[Mapping[str, Any]],
    calibration: Any,
) -> dict[str, Any]:
    return apply_permanence_episode(
        episode, x45.predict_episode(episode, candidate_values, calibration)
    )


def self_check() -> dict[str, Any]:
    inherited = x45.self_check()
    belief = {
        "track_id": "risk-a",
        "parent_track_id": "parent-a",
        "position_forward_m": 4.0,
        "position_right_m": 0.2,
        "velocity_forward_mps": -1.0,
        "velocity_right_mps": 0.5,
        "footprint_xy": [[3.5, 0.0], [4.5, 0.0], [4.5, 0.4], [3.5, 0.4]],
        "evidence_age_s": 0.0,
    }
    belief["x46_belief_state_time_s"] = 1.0
    propagated = propagate_belief(belief, 2.0)
    x24.require(
        abs(float(propagated["position_forward_m"]) - 3.0) <= x31.EPSILON
        and abs(float(propagated["position_right_m"]) - 0.7) <= x31.EPSILON,
        "x46_belief_kinematic_propagation",
    )
    x24.require(
        propagated["disposition"] == BELIEF_DISPOSITION
        and propagated["motion_authority"] == x27.RIGID_DYNAMIC
        and propagated["risk_eligible"] is True,
        "x46_belief_authority",
    )
    return {
        "status": "X46_EVIDENCE_TERMINATED_OBJECT_PERMANENCE_FALSIFIER_MET",
        "x45_structural_status": inherited["status"],
        "fixed_timeout_added": False,
        "same_parent_reobservation_is_release_evidence": True,
        "propagated_route_exit_is_release_evidence": True,
        "class_independent": True,
        "new_numeric_threshold_added": False,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), sort_keys=True))
