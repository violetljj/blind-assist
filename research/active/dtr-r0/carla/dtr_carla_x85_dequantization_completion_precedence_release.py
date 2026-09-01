"""X85 gives X68 geometric release precedence over X72 completion.

X68 can use a current object-local metric velocity to prove that every
inherited surface carrier misses the route. X72 currently may reopen risk in
the same frame by completing a credentialed surface boundary from X25 rigid
footprints. That ordering double-counts object-local rigid evidence: the
route-level geometric falsifier has already cleared the surface decision, yet
the downstream completion treats its historical credential as still live.

X85 clears only a same-frame reopening whose confirmed set consists entirely
of X72 completion proxies after an explicit X68 lateral-dequantization release.
Any direct carrier, any frame without an X68 release, and any later independent
evidence remains unchanged. No fitted metric threshold is added.
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
import dtr_carla_x72_credentialed_surface_boundary_completion as x72  # noqa: E402
import dtr_carla_x84_branch_overloaded_closing_continuation_release as x84  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X85_DEQUANTIZATION_COMPLETION_PRECEDENCE_RELEASE"
ARM_X85 = "X85_ISSUED_PLAN_DEQUANTIZATION_COMPLETION_PRECEDENCE_RELEASE"


def fixed_constants() -> dict[str, Any]:
    return {
        **x84.fixed_constants(),
        "representation": "X84_WITH_DEQUANTIZATION_COMPLETION_PRECEDENCE_RELEASE",
        "retained_core": "X84",
        "release_rule": (
            "SAME_FRAME_X68_GEOMETRIC_RELEASE_THEN_X72_COMPLETION_REOPENING_"
            "WITH_EVERY_CONFIRMED_CARRIER_AN_X72_COMPLETION_PROXY"
        ),
        "authority_rule": (
            "CURRENT_OBJECT_LOCAL_GEOMETRIC_FALSIFIER_PRECEDES_HISTORICAL_"
            "SURFACE_CREDENTIAL_BOUNDARY_COMPLETION"
        ),
        "direct_carriers_retained": True,
        "later_independent_evidence_retained": True,
        "all_track_geometry_motion_and_lineage_retained": True,
        "detector_threshold_change": False,
        "route_threshold_change": False,
        "weather_or_lighting_label_used": False,
        "class_specific_prior_used": False,
        "new_metric_threshold_added": False,
    }


def _is_x72_completion(row: Mapping[str, Any]) -> bool:
    return bool(row.get("x72_credentialed_surface_boundary_completion")) and (
        row.get("support_footprint_mode") == x72.SUPPORT_MODE
    )


def _release_partition(
    arm: Mapping[str, Any],
    rows: Mapping[str, Mapping[str, Any]],
    confirmed_ids: set[str],
) -> bool:
    return (
        bool(arm.get("x68_object_local_lateral_dequantization_used", False))
        and bool(arm.get("x68_object_local_lateral_released_track_ids", []))
        and bool(arm.get("x72_credentialed_surface_boundary_completion_used", False))
        and bool(confirmed_ids)
        and all(_is_x72_completion(rows[track_id]) for track_id in confirmed_ids)
    )


def apply_dequantization_completion_precedence_release_episode(
    core: dict[str, Any],
) -> dict[str, Any]:
    value = copy.deepcopy(core)
    released_frames = 0
    released_tracks = 0

    for frame in value["frames"]:
        arm = frame["arms"][x84.ARM_X84]
        arm["x85_dequantization_completion_precedence_release_used"] = False
        arm["x85_released_track_ids"] = []
        if not bool(arm.get("route_risk")):
            continue

        rows = {str(row["track_id"]): row for row in frame["tracks"]}
        confirmed_ids = {
            str(track_id) for track_id in arm.get("confirmed_risk_track_ids", [])
        }
        x24.require(confirmed_ids and confirmed_ids.issubset(rows), "x85_carrier_reference")
        if not _release_partition(arm, rows, confirmed_ids):
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
                        str(rows[track_id].get("parent_track_id") or track_id)
                        for track_id in candidate_ids
                        if track_id in rows
                    }
                ),
                "confirmed_risk_parent_track_ids": [],
                "x85_dequantization_completion_precedence_release_used": True,
                "x85_released_track_ids": sorted(confirmed_ids),
            }
        )
        released_frames += 1
        released_tracks += len(confirmed_ids)

    value["arms"][ARM_X85] = value["arms"].pop(x84.ARM_X84)
    value["diagnostics"]["x85_route_mode_counts"] = value["diagnostics"].pop(
        "x84_route_mode_counts"
    )
    value["diagnostics"].update(
        {
            "x85_dequantization_completion_precedence_release_frames": released_frames,
            "x85_dequantization_completion_precedence_released_tracks": released_tracks,
        }
    )
    for frame in value["frames"]:
        frame["arms"][ARM_X85] = frame["arms"].pop(x84.ARM_X84)
    return value


def self_check() -> dict[str, Any]:
    completion = {
        "track_id": "x72-completion::footprint-1",
        "x72_credentialed_surface_boundary_completion": True,
        "support_footprint_mode": x72.SUPPORT_MODE,
    }
    rows = {completion["track_id"]: completion}
    arm = {
        "x68_object_local_lateral_dequantization_used": True,
        "x68_object_local_lateral_released_track_ids": ["surface-1"],
        "x72_credentialed_surface_boundary_completion_used": True,
    }
    x24.require(
        _release_partition(arm, rows, set(rows))
        and not _release_partition(
            {**arm, "x68_object_local_lateral_released_track_ids": []}, rows, set(rows)
        )
        and not _release_partition(
            {**arm, "x72_credentialed_surface_boundary_completion_used": False},
            rows,
            set(rows),
        )
        and not _release_partition(
            arm,
            {
                completion["track_id"]: {
                    **completion,
                    "x72_credentialed_surface_boundary_completion": False,
                }
            },
            set(rows),
        ),
        "x85_release_precedence_partition",
    )
    return {
        "status": "X85_DEQUANTIZATION_COMPLETION_PRECEDENCE_FALSIFIER_MET",
        "release_only": True,
        "same_frame_x68_release_required": True,
        "pure_x72_completion_reopening_required": True,
        "direct_carriers_retained": True,
        "later_independent_evidence_retained": True,
        "new_metric_threshold_added": False,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), sort_keys=True))
