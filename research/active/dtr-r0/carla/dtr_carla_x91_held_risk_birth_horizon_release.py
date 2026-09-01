"""X91 prevents stale held evidence from birthing a post-horizon alert.

X86 and X87 establish that a route forecast cannot outlive the evidence
authority supporting it. X91 applies that same rule at the alert lifecycle
boundary. If the inherited X90 decision was clear in the previous frame, a new
alert cannot be born solely from held lineage envelopes when their predicted
route entry occurs after X24's evidence hold window. There is no current
footprint measurement capable of renewing that forecast's authority.

X91 clears only inherited risk births whose every confirmed carrier uses a
held lineage envelope and whose minimum route-entry time exceeds the existing
X24 hold window. Ongoing alerts, currently measured or mixed-support carriers,
and current or within-horizon entries remain unchanged. No new numeric
threshold is added.
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
import dtr_carla_x90_collision_credentialed_lateral_dominant_release as x90  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X91_HELD_RISK_BIRTH_HORIZON_RELEASE"
ARM_X91 = "X91_ISSUED_PLAN_HELD_RISK_BIRTH_HORIZON_RELEASE"
HELD_LINEAGE_SUPPORT = "HOLD_AXIS_ALIGNED_LINEAGE_ENVELOPE"


def fixed_constants() -> dict[str, Any]:
    return {
        **x90.fixed_constants(),
        "representation": "X90_WITH_HELD_RISK_BIRTH_HORIZON_RELEASE",
        "retained_core": "X90",
        "release_rule": (
            "INHERITED_ROUTE_RISK_BIRTH_AND_EVERY_CONFIRMED_CARRIER_USES_"
            "HELD_LINEAGE_SUPPORT_AND_MINIMUM_ROUTE_ENTRY_EXCEEDS_"
            "INHERITED_X24_HOLD_WINDOW"
        ),
        "authority_rule": (
            "STALE_HELD_EVIDENCE_CANNOT_BIRTH_A_NEW_ALERT_WHOSE_FORECAST_"
            "OUTLIVES_ITS_EVIDENCE_AUTHORITY"
        ),
        "evidence_authority_horizon_seconds": x24.HOLD_WINDOW_S,
        "ongoing_alerts_retained": True,
        "current_or_mixed_support_retained": True,
        "current_or_within_horizon_entries_retained": True,
        "inherited_sequence_used_without_release_cascade": True,
        "detector_threshold_change": False,
        "route_threshold_change": False,
        "weather_or_lighting_label_used": False,
        "class_specific_prior_used": False,
        "new_metric_threshold_added": False,
    }


def _is_held_lineage_carrier(row: Mapping[str, Any]) -> bool:
    return row.get("support_footprint_mode") == HELD_LINEAGE_SUPPORT


def _release_partition(
    arm: Mapping[str, Any],
    rows: Mapping[str, Mapping[str, Any]],
    confirmed_ids: set[str],
    previous_inherited_route_risk: bool,
) -> bool:
    entry = arm.get("minimum_entry_s")
    return (
        not previous_inherited_route_risk
        and bool(confirmed_ids)
        and entry is not None
        and float(entry) > x24.HOLD_WINDOW_S + x24.EPSILON
        and all(_is_held_lineage_carrier(rows[track_id]) for track_id in confirmed_ids)
    )


def apply_held_risk_birth_horizon_release_episode(core: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(core)
    released_frames = 0
    released_tracks = 0
    previous_inherited_route_risk = False

    for frame in value["frames"]:
        arm = frame["arms"][x90.ARM_X90]
        inherited_route_risk = bool(arm.get("route_risk"))
        arm["x91_held_risk_birth_horizon_release_used"] = False
        arm["x91_released_track_ids"] = []
        if inherited_route_risk:
            rows = {str(row["track_id"]): row for row in frame["tracks"]}
            confirmed_ids = {
                str(track_id) for track_id in arm.get("confirmed_risk_track_ids", [])
            }
            x24.require(
                confirmed_ids and confirmed_ids.issubset(rows),
                "x91_carrier_reference",
            )
            if _release_partition(
                arm, rows, confirmed_ids, previous_inherited_route_risk
            ):
                candidate_ids = {
                    str(track_id)
                    for track_id in arm.get("candidate_risk_track_ids", [])
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
                        "x91_held_risk_birth_horizon_release_used": True,
                        "x91_released_track_ids": sorted(confirmed_ids),
                    }
                )
                released_frames += 1
                released_tracks += len(confirmed_ids)
        previous_inherited_route_risk = inherited_route_risk

    value["arms"][ARM_X91] = value["arms"].pop(x90.ARM_X90)
    value["diagnostics"]["x91_route_mode_counts"] = value["diagnostics"].pop(
        "x90_route_mode_counts"
    )
    value["diagnostics"].update(
        {
            "x91_held_risk_birth_horizon_release_frames": released_frames,
            "x91_held_risk_birth_horizon_released_tracks": released_tracks,
        }
    )
    for frame in value["frames"]:
        frame["arms"][ARM_X91] = frame["arms"].pop(x90.ARM_X90)
    return value


def self_check() -> dict[str, Any]:
    track_id = "surface-1::branch-1"
    held = {
        "track_id": track_id,
        "support_footprint_mode": HELD_LINEAGE_SUPPORT,
    }
    rows = {track_id: held}
    late = {"minimum_entry_s": x24.HOLD_WINDOW_S + 0.1}
    x24.require(
        _release_partition(late, rows, set(rows), False)
        and not _release_partition(late, rows, set(rows), True)
        and not _release_partition(
            {"minimum_entry_s": x24.HOLD_WINDOW_S}, rows, set(rows), False
        )
        and not _release_partition(
            late,
            {
                track_id: {
                    **held,
                    "support_footprint_mode": "MEASURED_CONVEX_CELL_HULL",
                }
            },
            set(rows),
            False,
        ),
        "x91_held_risk_birth_horizon_partition",
    )
    return {
        "status": "X91_HELD_RISK_BIRTH_HORIZON_FALSIFIER_MET",
        "release_only": True,
        "inherited_x24_hold_window_reused": True,
        "inherited_risk_birth_required": True,
        "held_lineage_support_required": True,
        "ongoing_alerts_retained": True,
        "current_or_mixed_support_retained": True,
        "current_or_within_horizon_entries_retained": True,
        "release_cascade_prevented": True,
        "new_metric_threshold_added": False,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), sort_keys=True))
