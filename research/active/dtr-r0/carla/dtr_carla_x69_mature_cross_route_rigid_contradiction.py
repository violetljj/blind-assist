"""X69 mature cross-route rigid-footprint contradiction release around X68.

The surface representation can retain a lattice-quantized cross-route conflict
after the detector-local RGB-D rigid footprint no longer enters the route.  X69
treats that disagreement as release evidence only when every current confirmed
surface carrier is measured, cross-route, spatially matched to a current
measured X25 rigid footprint, and has accumulated ambiguity for a full inherited
X24 track-history window.  This maturity rule protects genuine early crossings
while allowing an object-local footprint to falsify a persistent surface tail.

No alert can be born or prolonged by X69.  Matching, history duration, route
geometry, and all detector thresholds are inherited unchanged.  Consumed
cohorts are Development evidence only.
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
import dtr_carla_x25_rigid_footprint_predictor as x25  # noqa: E402
import dtr_carla_x68_object_local_lateral_dequantization as x68  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X69_MATURE_CROSS_ROUTE_RIGID_CONTRADICTION"
ARM_X69 = "X69_ISSUED_PLAN_MATURE_CROSS_ROUTE_RIGID_CONTRADICTION"


def fixed_constants() -> dict[str, Any]:
    return {
        **x68.fixed_constants(),
        "representation": "X68_WITH_MATURE_X25_RIGID_FOOTPRINT_CONTRADICTION",
        "retained_core": "X68",
        "contradiction_source": "CURRENT_MEASURED_X25_RIGID_FOOTPRINT",
        "surface_scope": "ALL_CONFIRMED_CARRIERS_CURRENT_MEASURED_CROSS_ROUTE",
        "maturity_rule": "INHERITED_X24_TRACK_HISTORY_WINDOW",
        "rigid_negative_rule": "ZERO_CURRENT_X25_ROUTE_CANDIDATES",
        "association_distance_m": x24.ASSOCIATION_DISTANCE_M,
        "track_history_window_seconds": x24.TRACK_HISTORY_S,
        "detector_threshold_change": False,
        "route_threshold_change": False,
        "weather_or_lighting_label_used": False,
        "new_numeric_threshold_added": False,
    }


def _distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    forward = float(left["position_forward_m"]) - float(
        right["position_forward_m"]
    )
    route_right = float(left["position_right_m"]) - float(
        right["position_right_m"]
    )
    return (forward * forward + route_right * route_right) ** 0.5


def _is_cross_route(row: Mapping[str, Any]) -> bool:
    velocity_key = row.get("surface_transport_velocity_key_cells_per_s")
    return (
        isinstance(velocity_key, Sequence)
        and not isinstance(velocity_key, (str, bytes))
        and len(velocity_key) == 2
        and int(velocity_key[1]) != 0
    )


def _is_mature(row: Mapping[str, Any], sample_period_s: float) -> bool:
    return (
        int(row.get("surface_transport_ambiguity_frames", 0)) * sample_period_s
        + x24.EPSILON
        >= x24.TRACK_HISTORY_S
    )


def _all_carriers_contradicted(
    carriers: Sequence[Mapping[str, Any]],
    rigid_rows: Sequence[Mapping[str, Any]],
    sample_period_s: float,
) -> bool:
    return bool(carriers) and all(
        carrier.get("disposition") == "MEASURED"
        and "footprint_xy" in carrier
        and _is_cross_route(carrier)
        and _is_mature(carrier, sample_period_s)
        and any(
            _distance(carrier, rigid_row)
            <= x24.ASSOCIATION_DISTANCE_M + x24.EPSILON
            for rigid_row in rigid_rows
        )
        for carrier in carriers
    )


def apply_mature_cross_route_rigid_contradiction_episode(
    core: dict[str, Any], rigid: dict[str, Any]
) -> dict[str, Any]:
    value = copy.deepcopy(core)
    released_frames = 0
    released_tracks = 0

    x24.require(
        len(value["frames"]) == len(rigid["frames"]),
        "x69_frame_count",
    )
    for ordinal, (frame, rigid_frame) in enumerate(
        zip(value["frames"], rigid["frames"], strict=True)
    ):
        x24.require(
            int(frame["sample_index"]) == int(rigid_frame["sample_index"]),
            "x69_frame_alignment",
        )
        neighbour = (
            value["frames"][ordinal - 1]
            if ordinal
            else value["frames"][ordinal + 1]
        )
        sample_period_s = abs(float(frame["time_s"]) - float(neighbour["time_s"]))
        x24.require(sample_period_s > 0.0, "x69_sample_period")

        arm = frame["arms"][x68.ARM_X68]
        arm["x69_mature_cross_route_rigid_contradiction_release_used"] = False
        arm["x69_mature_cross_route_rigid_contradicted_track_ids"] = []
        confirmed_ids = {
            str(track_id) for track_id in arm.get("confirmed_risk_track_ids", [])
        }
        if not bool(arm.get("route_risk")) or not confirmed_ids:
            continue
        rigid_arm = rigid_frame["arms"][x25.ARM_X25]
        if rigid_arm.get("candidate_risk_track_ids"):
            continue
        rows = {str(row["track_id"]): row for row in frame["tracks"]}
        x24.require(
            confirmed_ids.issubset(rows),
            "x69_confirmed_track_reference",
        )
        carriers = [rows[track_id] for track_id in sorted(confirmed_ids)]
        rigid_rows = [
            row
            for row in rigid_frame["tracks"]
            if row.get("disposition") == "MEASURED"
        ]
        if not _all_carriers_contradicted(
            carriers, rigid_rows, sample_period_s
        ):
            continue

        remaining_candidates = {
            str(track_id) for track_id in arm.get("candidate_risk_track_ids", [])
        } - confirmed_ids
        arm.update(
            {
                "route_risk": False,
                "minimum_entry_s": None,
                "candidate_risk_track_ids": sorted(remaining_candidates),
                "confirmed_risk_track_ids": [],
                "candidate_risk_parent_track_ids": sorted(
                    {
                        str(rows[track_id]["parent_track_id"])
                        for track_id in remaining_candidates
                        if track_id in rows and rows[track_id].get("parent_track_id")
                    }
                ),
                "confirmed_risk_parent_track_ids": [],
                "x69_mature_cross_route_rigid_contradiction_release_used": True,
                "x69_mature_cross_route_rigid_contradicted_track_ids": sorted(
                    confirmed_ids
                ),
            }
        )
        released_frames += 1
        released_tracks += len(confirmed_ids)

    value["arms"][ARM_X69] = value["arms"].pop(x68.ARM_X68)
    value["diagnostics"]["x69_route_mode_counts"] = value["diagnostics"].pop(
        "x68_route_mode_counts"
    )
    value["diagnostics"].update(
        {
            "x69_mature_cross_route_rigid_contradiction_release_frames": (
                released_frames
            ),
            "x69_mature_cross_route_rigid_contradicted_tracks": released_tracks,
        }
    )
    for frame in value["frames"]:
        frame["arms"][ARM_X69] = frame["arms"].pop(x68.ARM_X68)
    return value


def self_check() -> dict[str, Any]:
    crossing = {
        "disposition": "MEASURED",
        "footprint_xy": [[0, 0], [1, 0], [1, 1]],
        "surface_transport_velocity_key_cells_per_s": [-10, 5],
        "surface_transport_ambiguity_frames": 10,
        "position_forward_m": 4.0,
        "position_right_m": 1.0,
    }
    rigid = {
        "position_forward_m": 4.1,
        "position_right_m": 1.0,
    }
    x24.require(
        _all_carriers_contradicted([crossing], [rigid], 0.1)
        and not _all_carriers_contradicted(
            [{**crossing, "surface_transport_ambiguity_frames": 9}],
            [rigid],
            0.1,
        )
        and not _all_carriers_contradicted(
            [{**crossing, "surface_transport_velocity_key_cells_per_s": [-10, 0]}],
            [rigid],
            0.1,
        ),
        "x69_mature_cross_route_partition",
    )
    return {
        "status": "X69_MATURE_CROSS_ROUTE_RIGID_CONTRADICTION_FALSIFIER_MET",
        "release_only": True,
        "current_rigid_measurement_required": True,
        "track_history_window_inherited": True,
        "new_numeric_threshold_added": False,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), sort_keys=True))
