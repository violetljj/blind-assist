"""X86 bounds receding X57 handback authority by the evidence hold horizon.

X57 may hand a current X24 metric route decision back when the retained surface
core has no eligible carrier. A forward-receding handback can still be a true
positive when it is already on the route or enters immediately. It must not,
however, authorize a later lateral entry that occurs only after the inherited
metric evidence hold window has expired. In that partition, the route forecast
outlives the observation authority supporting it.

X86 clears risk only when every confirmed carrier is an X57 metric handback,
every carrier is forward-receding, and the predicted minimum route-entry time
is beyond X24's existing hold window. Closing carriers, current/immediate
entries, direct surface carriers, and mixed evidence remain unchanged. No new
numeric threshold is added.
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
import dtr_carla_x57_retained_core_metric_handback as x57  # noqa: E402
import dtr_carla_x85_dequantization_completion_precedence_release as x85  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X86_RECEDING_HANDBACK_HORIZON_RELEASE"
ARM_X86 = "X86_ISSUED_PLAN_RECEDING_HANDBACK_HORIZON_RELEASE"


def fixed_constants() -> dict[str, Any]:
    return {
        **x85.fixed_constants(),
        "representation": "X85_WITH_RECEDING_HANDBACK_HORIZON_RELEASE",
        "retained_core": "X85",
        "release_rule": (
            "EVERY_CONFIRMED_CARRIER_IS_FORWARD_RECEDING_X57_METRIC_HANDBACK_"
            "AND_MINIMUM_ROUTE_ENTRY_EXCEEDS_INHERITED_X24_HOLD_WINDOW"
        ),
        "authority_rule": "ROUTE_FORECAST_CANNOT_OUTLIVE_SUPPORTING_EVIDENCE_AUTHORITY",
        "evidence_authority_horizon_seconds": x24.HOLD_WINDOW_S,
        "closing_handbacks_retained": True,
        "current_or_immediate_entries_retained": True,
        "direct_or_mixed_carriers_retained": True,
        "detector_threshold_change": False,
        "route_threshold_change": False,
        "weather_or_lighting_label_used": False,
        "class_specific_prior_used": False,
        "new_metric_threshold_added": False,
    }


def _is_receding_x57_handback(row: Mapping[str, Any]) -> bool:
    return (
        row.get("support_footprint_mode") == "X24_CONFIRMED_METRIC_TRACK"
        and row.get("disposition") == x57.HANDBACK_DISPOSITION
        and bool(row.get("x57_zero_eligible_metric_handback"))
        and float(row.get("velocity_forward_mps", 0.0)) > x24.EPSILON
    )


def _release_partition(
    arm: Mapping[str, Any],
    rows: Mapping[str, Mapping[str, Any]],
    confirmed_ids: set[str],
) -> bool:
    entry = arm.get("minimum_entry_s")
    return (
        bool(confirmed_ids)
        and entry is not None
        and float(entry) > x24.HOLD_WINDOW_S + x24.EPSILON
        and all(_is_receding_x57_handback(rows[track_id]) for track_id in confirmed_ids)
    )


def apply_receding_handback_horizon_release_episode(core: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(core)
    released_frames = 0
    released_tracks = 0

    for frame in value["frames"]:
        arm = frame["arms"][x85.ARM_X85]
        arm["x86_receding_handback_horizon_release_used"] = False
        arm["x86_released_track_ids"] = []
        if not bool(arm.get("route_risk")):
            continue

        rows = {str(row["track_id"]): row for row in frame["tracks"]}
        confirmed_ids = {
            str(track_id) for track_id in arm.get("confirmed_risk_track_ids", [])
        }
        x24.require(confirmed_ids and confirmed_ids.issubset(rows), "x86_carrier_reference")
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
                "x86_receding_handback_horizon_release_used": True,
                "x86_released_track_ids": sorted(confirmed_ids),
            }
        )
        released_frames += 1
        released_tracks += len(confirmed_ids)

    value["arms"][ARM_X86] = value["arms"].pop(x85.ARM_X85)
    value["diagnostics"]["x86_route_mode_counts"] = value["diagnostics"].pop(
        "x85_route_mode_counts"
    )
    value["diagnostics"].update(
        {
            "x86_receding_handback_horizon_release_frames": released_frames,
            "x86_receding_handback_horizon_released_tracks": released_tracks,
        }
    )
    for frame in value["frames"]:
        frame["arms"][ARM_X86] = frame["arms"].pop(x85.ARM_X85)
    return value


def self_check() -> dict[str, Any]:
    handback = {
        "track_id": "x57-handback::metric-1",
        "support_footprint_mode": "X24_CONFIRMED_METRIC_TRACK",
        "disposition": x57.HANDBACK_DISPOSITION,
        "x57_zero_eligible_metric_handback": True,
        "velocity_forward_mps": 1.0,
    }
    rows = {handback["track_id"]: handback}
    late = {"minimum_entry_s": x24.HOLD_WINDOW_S + 0.1}
    x24.require(
        _release_partition(late, rows, set(rows))
        and not _release_partition(
            {"minimum_entry_s": x24.HOLD_WINDOW_S}, rows, set(rows)
        )
        and not _release_partition(
            late,
            {handback["track_id"]: {**handback, "velocity_forward_mps": -1.0}},
            set(rows),
        )
        and not _release_partition(
            late,
            {
                handback["track_id"]: {
                    **handback,
                    "x57_zero_eligible_metric_handback": False,
                }
            },
            set(rows),
        ),
        "x86_receding_handback_horizon_partition",
    )
    return {
        "status": "X86_RECEDING_HANDBACK_HORIZON_FALSIFIER_MET",
        "release_only": True,
        "inherited_x24_hold_window_reused": True,
        "closing_handbacks_retained": True,
        "current_or_immediate_entries_retained": True,
        "direct_or_mixed_carriers_retained": True,
        "new_metric_threshold_added": False,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), sort_keys=True))
