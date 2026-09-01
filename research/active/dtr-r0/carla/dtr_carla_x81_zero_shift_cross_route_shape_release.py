"""X81 applies the cross-route shape credential to zero-shift surface carry.

A zero-shift surface branch can preserve object existence, but an otherwise
uncredentialed footprint whose lateral span does not exceed its route-forward
span does not establish cross-route occupancy.  X81 retains identity, geometry,
and motion rows while clearing route risk only when every confirmed carrier is
such a zero-shift surface branch.  It reuses X80's ordinal span test and adds no
numeric threshold.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Mapping


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import dtr_carla_x24_plan_adherent_predictor as x24  # noqa: E402
import dtr_carla_x76_zero_shift_parent_hull_motion_rejection as x76  # noqa: E402
import dtr_carla_x79_collision_credentialed_lateral_only_release as x79  # noqa: E402
import dtr_carla_x80_cross_route_footprint_credential_release as x80  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X81_ZERO_SHIFT_CROSS_ROUTE_SHAPE_RELEASE"
ARM_X81 = "X81_ISSUED_PLAN_ZERO_SHIFT_CROSS_ROUTE_SHAPE_RELEASE"
SURFACE_CLASS = x79.SURFACE_CLASS


def fixed_constants() -> dict[str, Any]:
    return {
        **x80.fixed_constants(),
        "representation": "X80_WITH_ZERO_SHIFT_CROSS_ROUTE_SHAPE_RELEASE",
        "retained_core": "X80",
        "release_rule": (
            "EVERY_CONFIRMED_CARRIER_IS_AN_UNCREDENTIALED_ZERO_SHIFT_SURFACE_"
            "WHOSE_LATERAL_SPAN_DOES_NOT_EXCEED_FORWARD_SPAN"
        ),
        "cross_route_shape_credential": "LATERAL_SPAN_STRICTLY_EXCEEDS_FORWARD_SPAN",
        "span_comparison": "ORDINAL_WITH_INHERITED_NUMERIC_EPSILON",
        "identity_geometry_and_motion_retained": True,
        "detector_threshold_change": False,
        "route_threshold_change": False,
        "weather_or_lighting_label_used": False,
        "class_specific_prior_used": False,
        "new_numeric_threshold_added": False,
    }


def _is_uncredentialed_zero_shift_without_cross_route_shape(
    row: Mapping[str, Any], credentialed_parent_ids: set[str]
) -> bool:
    parent_id = str(row.get("parent_track_id") or row["track_id"])
    footprint = row.get("footprint_xy")
    if (
        row.get("class_name") != SURFACE_CLASS
        or row.get("transport_state") != x76.ZERO_SHIFT
        or not footprint
        or parent_id in credentialed_parent_ids
    ):
        return False
    forward_span, lateral_span = x80.footprint_axis_spans(footprint)
    return lateral_span <= forward_span + x24.EPSILON


def apply_zero_shift_cross_route_shape_release_episode(
    core: dict[str, Any],
) -> dict[str, Any]:
    value = copy.deepcopy(core)
    credentialed_parent_ids: set[str] = set()
    released_frames = 0
    released_tracks = 0

    for frame in value["frames"]:
        arm = frame["arms"][x80.ARM_X80]
        credentialed_parent_ids.update(
            str(parent_id)
            for parent_id in arm.get("x75_collision_credential_birth_parent_ids", [])
        )
        arm["x81_zero_shift_cross_route_shape_release_used"] = False
        arm["x81_released_track_ids"] = []
        if not bool(arm.get("route_risk")):
            continue

        rows = {str(row["track_id"]): row for row in frame["tracks"]}
        confirmed_ids = {
            str(track_id) for track_id in arm.get("confirmed_risk_track_ids", [])
        }
        x24.require(confirmed_ids and confirmed_ids.issubset(rows), "x81_carrier_reference")
        if not all(
            _is_uncredentialed_zero_shift_without_cross_route_shape(
                rows[track_id], credentialed_parent_ids
            )
            for track_id in confirmed_ids
        ):
            continue

        candidate_ids = {
            str(track_id) for track_id in arm.get("candidate_risk_track_ids", [])
        } - confirmed_ids
        arm.update(
            {
                "route_risk": False,
                "minimum_entry_s": None,
                "candidate_risk_track_ids": sorted(candidate_ids),
                "confirmed_risk_track_ids": [],
                "candidate_risk_parent_track_ids": sorted(
                    {
                        str(rows[track_id]["parent_track_id"])
                        for track_id in candidate_ids
                        if track_id in rows and rows[track_id].get("parent_track_id")
                    }
                ),
                "confirmed_risk_parent_track_ids": [],
                "x81_zero_shift_cross_route_shape_release_used": True,
                "x81_released_track_ids": sorted(confirmed_ids),
            }
        )
        released_frames += 1
        released_tracks += len(confirmed_ids)

    value["arms"][ARM_X81] = value["arms"].pop(x80.ARM_X80)
    value["diagnostics"]["x81_route_mode_counts"] = value["diagnostics"].pop(
        "x80_route_mode_counts"
    )
    value["diagnostics"].update(
        {
            "x81_zero_shift_cross_route_shape_release_frames": released_frames,
            "x81_zero_shift_cross_route_shape_released_tracks": released_tracks,
        }
    )
    for frame in value["frames"]:
        frame["arms"][ARM_X81] = frame["arms"].pop(x80.ARM_X80)
    return value


def self_check() -> dict[str, Any]:
    base = {
        "track_id": "surface-1::component-1",
        "parent_track_id": "surface-1",
        "class_name": SURFACE_CLASS,
        "transport_state": x76.ZERO_SHIFT,
        "footprint_xy": [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]],
    }
    cross_route = {
        **base,
        "footprint_xy": [[0.0, 0.0], [1.0, 0.0], [1.0, 2.0], [0.0, 2.0]],
    }
    x24.require(
        _is_uncredentialed_zero_shift_without_cross_route_shape(base, set())
        and not _is_uncredentialed_zero_shift_without_cross_route_shape(
            cross_route, set()
        )
        and not _is_uncredentialed_zero_shift_without_cross_route_shape(
            base, {"surface-1"}
        )
        and not _is_uncredentialed_zero_shift_without_cross_route_shape(
            {**base, "transport_state": "DIRECTION_CONSISTENT_BRANCH_CONTINUATION"},
            set(),
        ),
        "x81_zero_shift_cross_route_shape_partition",
    )
    return {
        "status": "X81_ZERO_SHIFT_CROSS_ROUTE_SHAPE_FALSIFIER_MET",
        "release_only": True,
        "identity_geometry_and_motion_retained": True,
        "cross_route_elongated_zero_shift_retained": True,
        "collision_credentialed_zero_shift_retained": True,
        "mixed_carrier_frames_retained": True,
        "new_numeric_threshold_added": False,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), sort_keys=True))
