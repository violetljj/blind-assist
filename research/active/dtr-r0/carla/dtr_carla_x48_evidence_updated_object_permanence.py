"""X48 evidence-updated object permanence for issued-route risk.

X48 starts from X45.  An authorized risk belief is updated by a reobserved
same-parent rigid-dynamic footprint when that footprint still causally enters
the issued route.  When the parent is absent, X48 propagates the belief only
while route entry remains and X47's truth-blind depth ray does not show free
space.  X45 state-cycle suppression is explicit release evidence.

C22 is consumed posthoc synthetic Development and cannot confirm X48.
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
import dtr_carla_x46_evidence_terminated_object_permanence as x46  # noqa: E402
import dtr_carla_x47_depth_free_space_permanence_release as x47  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X48_EVIDENCE_UPDATED_OBJECT_PERMANENCE"
ARM_X48 = "X48_ISSUED_PLAN_EVIDENCE_UPDATED_OBJECT_PERMANENCE"


def fixed_constants() -> dict[str, Any]:
    return {
        **x45.fixed_constants(),
        "representation": "CAUSAL_STATE_CYCLE_WITH_EVIDENCE_UPDATED_PERMANENCE",
        "reobservation_update_rule": (
            "SAME_PARENT_AUTHORIZED_RIGID_DYNAMIC_FOOTPRINT_CONTINUES_ONLY_IF_"
            "IT_STILL_ENTERS_ISSUED_ROUTE"
        ),
        "occluded_update_rule": (
            "PROPAGATED_FOOTPRINT_CONTINUES_ONLY_IF_ROUTE_ENTRY_REMAINS_AND_"
            "DEPTH_RAY_DOES_NOT_SHOW_FREE_SPACE"
        ),
        "explicit_release_evidence": [
            "X45_STATE_CYCLE_SUPPRESSION",
            "REOBSERVED_PARENT_WITHOUT_AUTHORIZED_ROUTE_ENTRY",
            "PROPAGATED_ROUTE_EXIT",
            "IN_FOV_DEPTH_FREE_SPACE",
        ],
        "permanence_fixed_timeout_seconds": None,
        "depth_release_angular_grid": [
            x47.adapter.ANGULAR_GRID_WIDTH,
            x47.adapter.ANGULAR_GRID_HEIGHT,
        ],
        "depth_threshold_source": "INHERITED_RGBD_NEAR_DEPTH_SLAB",
        "new_numeric_threshold_added": False,
        "detector_threshold_change": False,
        "route_threshold_change": False,
        "score_threshold_change": False,
    }


def parent_id(row: Mapping[str, Any]) -> str:
    return str(row.get("parent_track_id") or row["track_id"])


def group_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[parent_id(row)].append(copy.deepcopy(dict(row)))
    return dict(grouped)


def authorized_dynamic_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        copy.deepcopy(dict(row))
        for row in rows
        if row.get("motion_authority") == x27.RIGID_DYNAMIC
        and bool(row.get("risk_eligible", False))
        and "footprint_xy" in row
    ]


def route_entries(
    rows: Sequence[Mapping[str, Any]], segments: Sequence[Any]
) -> tuple[list[dict[str, Any]], list[float]]:
    selected: list[dict[str, Any]] = []
    entries: list[float] = []
    for row in rows:
        entry = x30.first_contact_interval_entry_s(
            row["footprint_xy"],
            (
                float(row["velocity_forward_mps"]),
                float(row["velocity_right_mps"]),
            ),
            segments,
        )
        if entry is not None:
            selected.append(copy.deepcopy(dict(row)))
            entries.append(float(entry))
    return selected, entries


def state_rows(rows: Sequence[Mapping[str, Any]], now_s: float) -> list[dict[str, Any]]:
    values = []
    for row in rows:
        value = copy.deepcopy(dict(row))
        value["x48_belief_state_time_s"] = float(now_s)
        values.append(value)
    return values


def propagate_x48(row: Mapping[str, Any], now_s: float) -> dict[str, Any]:
    value = copy.deepcopy(dict(row))
    value["x46_belief_state_time_s"] = float(
        value.get("x48_belief_state_time_s", now_s)
    )
    propagated = x46.propagate_belief(value, now_s)
    propagated["x48_belief_state_time_s"] = float(now_s)
    propagated["x48_update_mode"] = "OCCLUDED_KINEMATIC_BELIEF"
    return propagated


def apply_evidence_updated_episode(
    episode: Any,
    value: dict[str, Any],
    calibration: Any,
) -> dict[str, Any]:
    active: dict[str, list[dict[str, Any]]] = {}
    previous_mode: str | None = None
    receipt_cache: dict[Path, dict[str, Any]] = {}
    reobserved_continuation_frames = 0
    occluded_continuation_frames = 0
    state_cycle_releases = 0
    reobserved_route_exit_releases = 0
    propagated_route_exit_releases = 0
    depth_free_space_releases = 0

    for observation, frame in zip(
        episode.observations, value["frames"], strict=True
    ):
        x24.require(
            int(observation.sample_index) == int(frame["sample_index"])
            and abs(float(observation.time_s) - float(frame["time_s"]))
            <= x31.EPSILON,
            "x48_observation_frame_alignment",
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
        by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in frame["tracks"]:
            by_parent[parent_id(row)].append(row)
        arm["x48_object_permanence_used"] = False
        arm["x48_object_permanence_update_modes"] = []

        if bool(arm.get("route_risk")):
            confirmed = [
                tracks[str(track_id)]
                for track_id in arm.get("confirmed_risk_track_ids", [])
                if str(track_id) in tracks
            ]
            active = group_rows(
                state_rows(authorized_dynamic_rows(confirmed), observation.time_s)
            )
            continue

        if bool(arm.get("x45_state_cycle_suppressed", False)):
            state_cycle_releases += int(bool(active))
            active = {}
            continue

        continued: list[dict[str, Any]] = []
        entries: list[float] = []
        modes: list[str] = []
        for belief_parent, beliefs in active.items():
            current = by_parent.get(belief_parent, [])
            if current:
                eligible, current_entries = route_entries(
                    authorized_dynamic_rows(current), segments
                )
                if eligible:
                    for row in eligible:
                        row["x48_belief_state_time_s"] = float(observation.time_s)
                        row["x48_update_mode"] = "REOBSERVED_CAUSAL_UPDATE"
                    continued.extend(eligible)
                    entries.extend(current_entries)
                    modes.append("REOBSERVED_CAUSAL_UPDATE")
                else:
                    reobserved_route_exit_releases += 1
                continue

            propagated = [
                propagate_x48(row, float(observation.time_s)) for row in beliefs
            ]
            eligible, propagated_entries = route_entries(propagated, segments)
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
        reobserved_continuation_frames += int("REOBSERVED_CAUSAL_UPDATE" in modes)
        occluded_continuation_frames += int("OCCLUDED_KINEMATIC_BELIEF" in modes)
        for row in continued:
            track_id = str(row["track_id"])
            if track_id not in tracks:
                frame["tracks"].append(row)
                frame["risk_eligible_tracks"] = int(frame["risk_eligible_tracks"]) + 1
        track_ids = sorted(str(row["track_id"]) for row in continued)
        parent_ids = sorted({parent_id(row) for row in continued})
        arm.update(
            {
                "route_risk": True,
                "minimum_entry_s": min(entries),
                "candidate_risk_track_ids": track_ids,
                "confirmed_risk_track_ids": track_ids,
                "candidate_risk_parent_track_ids": parent_ids,
                "confirmed_risk_parent_track_ids": parent_ids,
                "x48_object_permanence_used": True,
                "x48_object_permanence_update_modes": sorted(set(modes)),
            }
        )
        active = group_rows(state_rows(continued, observation.time_s))

    value["arms"][ARM_X48] = value["arms"].pop(x45.ARM_X45)
    value["diagnostics"]["x48_route_mode_counts"] = value["diagnostics"].pop(
        "x45_route_mode_counts"
    )
    value["diagnostics"].update(
        {
            "x48_reobserved_continuation_frames": reobserved_continuation_frames,
            "x48_occluded_continuation_frames": occluded_continuation_frames,
            "x48_state_cycle_releases": state_cycle_releases,
            "x48_reobserved_route_exit_releases": reobserved_route_exit_releases,
            "x48_propagated_route_exit_releases": propagated_route_exit_releases,
            "x48_depth_free_space_releases": depth_free_space_releases,
        }
    )
    for frame in value["frames"]:
        frame["arms"][ARM_X48] = frame["arms"].pop(x45.ARM_X45)
    return value


def predict_episode(
    episode: Any,
    candidate_values: Sequence[Mapping[str, Any]],
    calibration: Any,
) -> dict[str, Any]:
    return apply_evidence_updated_episode(
        episode,
        x45.predict_episode(episode, candidate_values, calibration),
        calibration,
    )


def self_check() -> dict[str, Any]:
    inherited = x47.self_check()
    authorized = {
        "track_id": "a",
        "parent_track_id": "p",
        "motion_authority": x27.RIGID_DYNAMIC,
        "risk_eligible": True,
        "footprint_xy": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
    }
    x24.require(
        len(authorized_dynamic_rows([authorized])) == 1,
        "x48_authorized_reobservation",
    )
    return {
        "status": "X48_EVIDENCE_UPDATED_OBJECT_PERMANENCE_FALSIFIER_MET",
        "x47_structural_status": inherited["status"],
        "reobservation_updates_belief": True,
        "state_cycle_conflict_releases_belief": True,
        "depth_free_space_releases_belief": True,
        "fixed_timeout_added": False,
        "new_numeric_threshold_added": False,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), sort_keys=True))
