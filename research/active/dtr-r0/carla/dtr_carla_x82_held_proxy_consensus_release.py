"""X82 denies route-risk ownership to held-only completion consensus.

X72 boundary completion is an association proxy: it transfers an already
earned surface credential onto a matching rigid footprint.  Multiple carried
completion proxies do not become independent current observations merely by
agreeing.  X82 therefore retains every track, footprint, motion estimate, and
identity row while clearing route risk when every confirmed carrier is an X72
completion, more than one distinct proxy is present, and every proxy is HOLD.
Any current measurement, any direct surface carrier, or a single completion
proxy retains the conservative X81 decision.  No metric threshold is added.
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
import dtr_carla_x81_zero_shift_cross_route_shape_release as x81  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X82_HELD_PROXY_CONSENSUS_RELEASE"
ARM_X82 = "X82_ISSUED_PLAN_HELD_PROXY_CONSENSUS_RELEASE"


def fixed_constants() -> dict[str, Any]:
    return {
        **x81.fixed_constants(),
        "representation": "X81_WITH_HELD_PROXY_CONSENSUS_RELEASE",
        "retained_core": "X81",
        "release_rule": (
            "EVERY_CONFIRMED_CARRIER_IS_AN_X72_COMPLETION_PROXY_AND_DISTINCT_"
            "PROXIES_ARE_PLURAL_AND_EVERY_PROXY_IS_HOLD"
        ),
        "authority_rule": (
            "CARRIED_ASSOCIATION_PROXIES_DO_NOT_GAIN_CURRENT_ROUTE_RISK_"
            "AUTHORITY_BY_MULTIPLICITY"
        ),
        "single_proxy_retained": True,
        "any_current_measurement_retained": True,
        "mixed_direct_and_proxy_carriers_retained": True,
        "identity_geometry_and_motion_retained": True,
        "detector_threshold_change": False,
        "route_threshold_change": False,
        "weather_or_lighting_label_used": False,
        "class_specific_prior_used": False,
        "new_metric_threshold_added": False,
    }


def _is_held_x72_completion(row: Mapping[str, Any]) -> bool:
    return bool(row.get("x72_credentialed_surface_boundary_completion")) and (
        row.get("support_footprint_mode") == x72.SUPPORT_MODE
        and row.get("disposition") == "HOLD"
    )


def _is_held_proxy_consensus(
    rows: Mapping[str, Mapping[str, Any]], confirmed_ids: set[str]
) -> bool:
    return (
        len(confirmed_ids) > 1
        and len(
            {
                str(rows[track_id].get("parent_track_id") or track_id)
                for track_id in confirmed_ids
            }
        )
        > 1
        and all(_is_held_x72_completion(rows[track_id]) for track_id in confirmed_ids)
    )


def apply_held_proxy_consensus_release_episode(core: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(core)
    released_frames = 0
    released_tracks = 0

    for frame in value["frames"]:
        arm = frame["arms"][x81.ARM_X81]
        arm["x82_held_proxy_consensus_release_used"] = False
        arm["x82_released_track_ids"] = []
        if not bool(arm.get("route_risk")):
            continue

        rows = {str(row["track_id"]): row for row in frame["tracks"]}
        confirmed_ids = {
            str(track_id) for track_id in arm.get("confirmed_risk_track_ids", [])
        }
        x24.require(confirmed_ids and confirmed_ids.issubset(rows), "x82_carrier_reference")
        if not _is_held_proxy_consensus(rows, confirmed_ids):
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
                "x82_held_proxy_consensus_release_used": True,
                "x82_released_track_ids": sorted(confirmed_ids),
            }
        )
        released_frames += 1
        released_tracks += len(confirmed_ids)

    value["arms"][ARM_X82] = value["arms"].pop(x81.ARM_X81)
    value["diagnostics"]["x82_route_mode_counts"] = value["diagnostics"].pop(
        "x81_route_mode_counts"
    )
    value["diagnostics"].update(
        {
            "x82_held_proxy_consensus_release_frames": released_frames,
            "x82_held_proxy_consensus_released_tracks": released_tracks,
        }
    )
    for frame in value["frames"]:
        frame["arms"][ARM_X82] = frame["arms"].pop(x81.ARM_X81)
    return value


def self_check() -> dict[str, Any]:
    held = {
        "track_id": "x72-completion::footprint-1",
        "parent_track_id": "footprint-1",
        "x72_credentialed_surface_boundary_completion": True,
        "support_footprint_mode": x72.SUPPORT_MODE,
        "disposition": "HOLD",
    }
    second = {
        **held,
        "track_id": "x72-completion::footprint-2",
        "parent_track_id": "footprint-2",
    }
    rows = {str(row["track_id"]): row for row in (held, second)}
    ids = set(rows)
    x24.require(
        _is_held_proxy_consensus(rows, ids)
        and not _is_held_proxy_consensus(rows, {held["track_id"]})
        and not _is_held_proxy_consensus(
            {**rows, second["track_id"]: {**second, "disposition": "MEASURED"}},
            ids,
        )
        and not _is_held_proxy_consensus(
            {
                **rows,
                second["track_id"]: {
                    **second,
                    "x72_credentialed_surface_boundary_completion": False,
                },
            },
            ids,
        )
        and not _is_held_proxy_consensus(
            {**rows, second["track_id"]: {**second, "parent_track_id": "footprint-1"}},
            ids,
        ),
        "x82_held_proxy_consensus_partition",
    )
    return {
        "status": "X82_HELD_PROXY_CONSENSUS_FALSIFIER_MET",
        "release_only": True,
        "identity_geometry_and_motion_retained": True,
        "single_proxy_retained": True,
        "current_measurement_retained": True,
        "mixed_carrier_frames_retained": True,
        "duplicate_parent_aliases_do_not_create_plurality": True,
        "new_metric_threshold_added": False,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), sort_keys=True))
