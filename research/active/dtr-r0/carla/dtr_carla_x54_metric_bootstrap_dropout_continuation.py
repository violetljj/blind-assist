"""X54 one-observation continuation for a fresh metric bootstrap dropout.

X54 preserves X53.  When a currently measured, depth-supported X24 metric
closing bootstrap carried risk in the immediately previous observation, X54
may propagate that same authorized state through the next observation only.
The propagated footprint must still enter the current issued route and current
RGB-D must not prove visible free space.  A continuation row is not itself an
eligible seed, so the exception cannot chain.

C22 is consumed posthoc synthetic Development and cannot confirm X54.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import dtr_carla_x24_plan_adherent_predictor as x24  # noqa: E402
import dtr_carla_x27_occupancy_authority_predictor as x27  # noqa: E402
import dtr_carla_x47_depth_free_space_permanence_release as x47  # noqa: E402
import dtr_carla_x53_anchor_redundant_parent_continuation as x53  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X54_METRIC_BOOTSTRAP_DROPOUT_CONTINUATION"
ARM_X54 = "X54_ISSUED_PLAN_METRIC_BOOTSTRAP_DROPOUT_CONTINUATION"
SOURCE_MODE = "X24_METRIC_POINT_BOOTSTRAP"
CONTINUATION_MODE = "X24_METRIC_BOOTSTRAP_DROPOUT_CONTINUATION"
CONTINUATION_DISPOSITION = "METRIC_BOOTSTRAP_DROPOUT_CONTINUATION"


def fixed_constants() -> dict[str, Any]:
    return {
        **x53.fixed_constants(),
        "representation": "ANCHOR_REDUNDANCY_WITH_METRIC_BOOTSTRAP_DROPOUT_CONTINUATION",
        "dropout_source": "IMMEDIATELY_PREVIOUS_CURRENT_MEASURED_X24_METRIC_POINT_BOOTSTRAP",
        "dropout_continuation_span": "ONE_NEXT_OBSERVATION_ONLY",
        "dropout_route_rule": "PROPAGATED_FOOTPRINT_STILL_ENTERS_CURRENT_ISSUED_ROUTE",
        "dropout_release_rule": "INHERITED_X47_VISIBLE_FREE_SPACE",
        "dropout_continuation_can_reseed": False,
        "new_numeric_threshold_added": False,
    }


def fresh_metric_bootstrap(row: Mapping[str, Any]) -> bool:
    return (
        row.get("disposition") == "MEASURED"
        and row.get("support_footprint_mode") == SOURCE_MODE
        and row.get("motion_authority") == x27.RIGID_DYNAMIC
        and bool(row.get("risk_eligible", False))
        and row.get("depth_grid_support") is not None
    )


def propagate_bootstrap(
    row: Mapping[str, Any], previous_time_s: float, now_s: float
) -> dict[str, Any]:
    value = copy.deepcopy(dict(row))
    dt = max(0.0, float(now_s) - float(previous_time_s))
    value["position_forward_m"] = float(value["position_forward_m"]) + float(
        value["velocity_forward_mps"]
    ) * dt
    value["position_right_m"] = float(value["position_right_m"]) + float(
        value["velocity_right_mps"]
    ) * dt
    value["track_id"] = f"x54-dropout::{row['track_id']}"
    value["disposition"] = CONTINUATION_DISPOSITION
    value["support_footprint_mode"] = CONTINUATION_MODE
    value["depth_grid_support"] = None
    value["evidence_age_s"] = float(value.get("evidence_age_s") or 0.0) + dt
    value["x54_metric_bootstrap_dropout_continuation"] = True
    return value


def apply_metric_bootstrap_dropout_episode(
    episode: Any,
    value: dict[str, Any],
    calibration: Any,
) -> dict[str, Any]:
    value = copy.deepcopy(value)
    previous_mode: str | None = None
    receipt_cache: dict[Path, dict[str, Any]] = {}
    continuation_frames = 0
    route_exit_releases = 0
    depth_free_space_releases = 0

    for ordinal, (observation, frame) in enumerate(
        zip(episode.observations, value["frames"], strict=True)
    ):
        arm = frame["arms"][x53.ARM_X53]
        arm["x54_metric_bootstrap_dropout_used"] = False
        arm["x54_metric_bootstrap_dropout_track_ids"] = []
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
        if ordinal == 0 or bool(arm.get("route_risk")):
            continue

        previous_frame = value["frames"][ordinal - 1]
        previous_arm = previous_frame["arms"][x53.ARM_X53]
        if not bool(previous_arm.get("route_risk")):
            continue
        previous_tracks = {
            str(row["track_id"]): row for row in previous_frame["tracks"]
        }
        sources = [
            previous_tracks[str(track_id)]
            for track_id in previous_arm.get("confirmed_risk_track_ids", [])
            if str(track_id) in previous_tracks
            and fresh_metric_bootstrap(previous_tracks[str(track_id)])
        ]
        if not sources:
            continue

        propagated = [
            propagate_bootstrap(
                row, float(previous_frame["time_s"]), float(frame["time_s"])
            )
            for row in sources
        ]
        carriers: list[dict[str, Any]] = []
        entries: list[float] = []
        for row in propagated:
            entry = x24.route.first_selected_route_entry_s(
                selection,
                receipt=receipt,
                now_s=observation.time_s,
                wearer_position_xy=wearer_position,
                wearer_velocity_xy=wearer_velocity,
                target_position_xy=(
                    float(row["position_forward_m"]),
                    float(row["position_right_m"]),
                ),
                target_velocity_xy=(
                    float(row["velocity_forward_mps"]),
                    float(row["velocity_right_mps"]),
                ),
            )
            if entry is not None:
                carriers.append(row)
                entries.append(float(entry))
        if not carriers:
            route_exit_releases += 1
            continue
        if x47.visible_free_space(
            observation, carriers, calibration, episode.route_frame
        ):
            depth_free_space_releases += 1
            continue

        existing_ids = {str(row["track_id"]) for row in frame["tracks"]}
        for row in carriers:
            x24.require(
                str(row["track_id"]) not in existing_ids,
                "x54_dropout_track_id_collision",
            )
            frame["tracks"].append(row)
            frame["risk_eligible_tracks"] = int(frame["risk_eligible_tracks"]) + 1
        track_ids = sorted(str(row["track_id"]) for row in carriers)
        parent_ids = sorted(str(row["parent_track_id"]) for row in carriers)
        arm.update(
            {
                "route_risk": True,
                "minimum_entry_s": min(entries),
                "candidate_risk_track_ids": track_ids,
                "confirmed_risk_track_ids": track_ids,
                "candidate_risk_parent_track_ids": parent_ids,
                "confirmed_risk_parent_track_ids": parent_ids,
                "x54_metric_bootstrap_dropout_used": True,
                "x54_metric_bootstrap_dropout_track_ids": track_ids,
            }
        )
        continuation_frames += 1

    value["arms"][ARM_X54] = value["arms"].pop(x53.ARM_X53)
    value["diagnostics"]["x54_route_mode_counts"] = value["diagnostics"].pop(
        "x53_route_mode_counts"
    )
    value["diagnostics"].update(
        {
            "x54_metric_bootstrap_dropout_continuation_frames": continuation_frames,
            "x54_metric_bootstrap_route_exit_releases": route_exit_releases,
            "x54_metric_bootstrap_depth_free_space_releases": depth_free_space_releases,
        }
    )
    for frame in value["frames"]:
        frame["arms"][ARM_X54] = frame["arms"].pop(x53.ARM_X53)
    return value


def predict_episode(
    episode: Any,
    candidate_values: Sequence[Mapping[str, Any]],
    calibration: Any,
) -> dict[str, Any]:
    return apply_metric_bootstrap_dropout_episode(
        episode,
        x53.predict_episode(episode, candidate_values, calibration),
        calibration,
    )


def self_check() -> dict[str, Any]:
    inherited = x53.self_check()
    source = {
        "track_id": "metric",
        "parent_track_id": "metric",
        "disposition": "MEASURED",
        "support_footprint_mode": SOURCE_MODE,
        "motion_authority": x27.RIGID_DYNAMIC,
        "risk_eligible": True,
        "depth_grid_support": 10,
        "position_forward_m": 4.0,
        "position_right_m": 0.0,
        "velocity_forward_mps": -1.0,
        "velocity_right_mps": 0.0,
        "evidence_age_s": 0.0,
    }
    x24.require(fresh_metric_bootstrap(source), "x54_fresh_metric_bootstrap")
    continued = propagate_bootstrap(source, 1.0, 1.1)
    x24.require(
        continued["support_footprint_mode"] == CONTINUATION_MODE
        and not fresh_metric_bootstrap(continued),
        "x54_continuation_cannot_reseed",
    )
    return {
        "status": "X54_METRIC_BOOTSTRAP_DROPOUT_CONTINUATION_FALSIFIER_MET",
        "x53_structural_status": inherited["status"],
        "one_next_observation_only": True,
        "continuation_can_reseed": False,
        "depth_free_space_release_preserved": True,
        "new_numeric_threshold_added": False,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), sort_keys=True))
