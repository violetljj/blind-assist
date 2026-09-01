"""X80 requires cross-route shape support for an X71 occupancy birth.

X71 can birth route risk when an X24 metric point and X25 rigid footprint agree
at their predicted route-entry time.  That establishes a shared object and
motion hypothesis, but an otherwise uncredentialed footprint whose spatial
support is not cross-route elongated does not independently establish a
cross-route occupancy carrier.  X80 retains the track and motion rows while
clearing route risk only when every confirmed carrier is such an X71 birth.

The credential is ordinal: lateral span must exceed forward span.  It adds no
learned or numeric threshold and keeps collision-credentialed or mixed-carrier
frames conservative.
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
import dtr_carla_x71_entry_cotransport_occupancy_birth as x71  # noqa: E402
import dtr_carla_x79_collision_credentialed_lateral_only_release as x79  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X80_CROSS_ROUTE_FOOTPRINT_CREDENTIAL_RELEASE"
ARM_X80 = "X80_ISSUED_PLAN_CROSS_ROUTE_FOOTPRINT_CREDENTIAL_RELEASE"


def fixed_constants() -> dict[str, Any]:
    return {
        **x79.fixed_constants(),
        "representation": "X79_WITH_CROSS_ROUTE_FOOTPRINT_CREDENTIAL_RELEASE",
        "retained_core": "X79",
        "release_rule": (
            "EVERY_CONFIRMED_CARRIER_IS_AN_UNCREDENTIALED_X71_BIRTH_WHOSE_"
            "RIGID_FOOTPRINT_LATERAL_SPAN_DOES_NOT_EXCEED_FORWARD_SPAN"
        ),
        "cross_route_shape_credential": "LATERAL_SPAN_STRICTLY_EXCEEDS_FORWARD_SPAN",
        "span_comparison": "ORDINAL_WITH_INHERITED_NUMERIC_EPSILON",
        "identity_and_motion_memory_retained": True,
        "detector_threshold_change": False,
        "route_threshold_change": False,
        "weather_or_lighting_label_used": False,
        "class_specific_prior_used": False,
        "new_numeric_threshold_added": False,
    }


def footprint_axis_spans(footprint_xy: Sequence[Sequence[float]]) -> tuple[float, float]:
    points = [(float(point[0]), float(point[1])) for point in footprint_xy]
    x24.require(len(points) >= 3, "x80_footprint_vertices")
    forward_span = max(point[0] for point in points) - min(point[0] for point in points)
    lateral_span = max(point[1] for point in points) - min(point[1] for point in points)
    return forward_span, lateral_span


def _is_uncredentialed_x71_birth_without_cross_route_shape(
    row: Mapping[str, Any], credentialed_parent_ids: set[str]
) -> bool:
    parent_id = str(row.get("parent_track_id") or row["track_id"])
    footprint = row.get("footprint_xy")
    if (
        not bool(row.get("x71_entry_cotransport_occupancy_birth"))
        or row.get("support_footprint_mode") != x71.SUPPORT_MODE
        or not footprint
        or parent_id in credentialed_parent_ids
    ):
        return False
    forward_span, lateral_span = footprint_axis_spans(footprint)
    return lateral_span <= forward_span + x24.EPSILON


def apply_cross_route_footprint_credential_release_episode(
    core: dict[str, Any],
) -> dict[str, Any]:
    value = copy.deepcopy(core)
    credentialed_parent_ids: set[str] = set()
    released_frames = 0
    released_tracks = 0

    for frame in value["frames"]:
        arm = frame["arms"][x79.ARM_X79]
        credentialed_parent_ids.update(
            str(parent_id)
            for parent_id in arm.get("x75_collision_credential_birth_parent_ids", [])
        )
        arm["x80_cross_route_footprint_credential_release_used"] = False
        arm["x80_released_track_ids"] = []
        if not bool(arm.get("route_risk")):
            continue

        rows = {str(row["track_id"]): row for row in frame["tracks"]}
        confirmed_ids = {
            str(track_id) for track_id in arm.get("confirmed_risk_track_ids", [])
        }
        x24.require(confirmed_ids and confirmed_ids.issubset(rows), "x80_carrier_reference")
        if not all(
            _is_uncredentialed_x71_birth_without_cross_route_shape(
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
                "x80_cross_route_footprint_credential_release_used": True,
                "x80_released_track_ids": sorted(confirmed_ids),
            }
        )
        released_frames += 1
        released_tracks += len(confirmed_ids)

    value["arms"][ARM_X80] = value["arms"].pop(x79.ARM_X79)
    value["diagnostics"]["x80_route_mode_counts"] = value["diagnostics"].pop(
        "x79_route_mode_counts"
    )
    value["diagnostics"].update(
        {
            "x80_cross_route_footprint_credential_release_frames": released_frames,
            "x80_cross_route_footprint_credential_released_tracks": released_tracks,
        }
    )
    for frame in value["frames"]:
        frame["arms"][ARM_X80] = frame["arms"].pop(x79.ARM_X79)
    return value


def self_check() -> dict[str, Any]:
    base = {
        "track_id": "x71-birth::footprint-1",
        "parent_track_id": "footprint-1",
        "support_footprint_mode": x71.SUPPORT_MODE,
        "x71_entry_cotransport_occupancy_birth": True,
        "footprint_xy": [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]],
    }
    cross_route = {
        **base,
        "footprint_xy": [[0.0, 0.0], [1.0, 0.0], [1.0, 2.0], [0.0, 2.0]],
    }
    x24.require(
        _is_uncredentialed_x71_birth_without_cross_route_shape(base, set())
        and not _is_uncredentialed_x71_birth_without_cross_route_shape(
            cross_route, set()
        )
        and not _is_uncredentialed_x71_birth_without_cross_route_shape(
            base, {"footprint-1"}
        )
        and not _is_uncredentialed_x71_birth_without_cross_route_shape(
            {**base, "x71_entry_cotransport_occupancy_birth": False}, set()
        ),
        "x80_cross_route_footprint_credential_partition",
    )
    return {
        "status": "X80_CROSS_ROUTE_FOOTPRINT_CREDENTIAL_FALSIFIER_MET",
        "release_only": True,
        "identity_and_motion_memory_retained": True,
        "cross_route_elongated_birth_retained": True,
        "collision_credentialed_birth_retained": True,
        "mixed_carrier_frames_retained": True,
        "new_numeric_threshold_added": False,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), sort_keys=True))
