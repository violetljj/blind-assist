"""X87 bounds sole X72 completion authority by the evidence hold horizon.

X72 can complete a fragmented, historically credentialed surface parent from
current X25 rigid footprints. When that completion is the only active route-risk
mechanism, its forecast must not extend beyond the inherited evidence hold
window. Otherwise a historical credential plus a current proxy authorizes a
future entry after the observation authority supporting the association has
expired.

X87 clears only frames whose sole active mechanism is X72 boundary completion,
whose confirmed set consists entirely of X72 completion proxies, and whose
minimum route-entry time exceeds X24's existing hold window. Any independent
mechanism, direct carrier, or current/immediate completion remains unchanged.
No new numeric threshold is added.
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
import dtr_carla_x86_receding_handback_horizon_release as x86  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X87_SOLO_COMPLETION_HORIZON_RELEASE"
ARM_X87 = "X87_ISSUED_PLAN_SOLO_COMPLETION_HORIZON_RELEASE"
X72_FLAG = "x72_credentialed_surface_boundary_completion_used"


def fixed_constants() -> dict[str, Any]:
    return {
        **x86.fixed_constants(),
        "representation": "X86_WITH_SOLO_COMPLETION_HORIZON_RELEASE",
        "retained_core": "X86",
        "release_rule": (
            "SOLE_ACTIVE_MECHANISM_IS_X72_BOUNDARY_COMPLETION_AND_EVERY_"
            "CONFIRMED_CARRIER_IS_AN_X72_COMPLETION_PROXY_AND_MINIMUM_ROUTE_"
            "ENTRY_EXCEEDS_INHERITED_X24_HOLD_WINDOW"
        ),
        "authority_rule": (
            "HISTORICAL_CREDENTIAL_COMPLETION_CANNOT_OUTLIVE_SUPPORTING_"
            "EVIDENCE_AUTHORITY_WITHOUT_AN_INDEPENDENT_MECHANISM"
        ),
        "evidence_authority_horizon_seconds": x24.HOLD_WINDOW_S,
        "independent_mechanism_frames_retained": True,
        "direct_or_mixed_carriers_retained": True,
        "current_or_immediate_entries_retained": True,
        "detector_threshold_change": False,
        "route_threshold_change": False,
        "weather_or_lighting_label_used": False,
        "class_specific_prior_used": False,
        "new_metric_threshold_added": False,
    }


def _active_mechanism_flags(arm: Mapping[str, Any]) -> set[str]:
    return {
        str(key)
        for key, value in arm.items()
        if str(key).endswith("_used") and value is True
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
    entry = arm.get("minimum_entry_s")
    return (
        _active_mechanism_flags(arm) == {X72_FLAG}
        and bool(confirmed_ids)
        and entry is not None
        and float(entry) > x24.HOLD_WINDOW_S + x24.EPSILON
        and all(_is_x72_completion(rows[track_id]) for track_id in confirmed_ids)
    )


def apply_solo_completion_horizon_release_episode(core: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(core)
    released_frames = 0
    released_tracks = 0

    for frame in value["frames"]:
        arm = frame["arms"][x86.ARM_X86]
        arm["x87_solo_completion_horizon_release_used"] = False
        arm["x87_released_track_ids"] = []
        if not bool(arm.get("route_risk")):
            continue

        rows = {str(row["track_id"]): row for row in frame["tracks"]}
        confirmed_ids = {
            str(track_id) for track_id in arm.get("confirmed_risk_track_ids", [])
        }
        x24.require(confirmed_ids and confirmed_ids.issubset(rows), "x87_carrier_reference")
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
                "x87_solo_completion_horizon_release_used": True,
                "x87_released_track_ids": sorted(confirmed_ids),
            }
        )
        released_frames += 1
        released_tracks += len(confirmed_ids)

    value["arms"][ARM_X87] = value["arms"].pop(x86.ARM_X86)
    value["diagnostics"]["x87_route_mode_counts"] = value["diagnostics"].pop(
        "x86_route_mode_counts"
    )
    value["diagnostics"].update(
        {
            "x87_solo_completion_horizon_release_frames": released_frames,
            "x87_solo_completion_horizon_released_tracks": released_tracks,
        }
    )
    for frame in value["frames"]:
        frame["arms"][ARM_X87] = frame["arms"].pop(x86.ARM_X86)
    return value


def self_check() -> dict[str, Any]:
    completion = {
        "track_id": "x72-completion::footprint-1",
        "x72_credentialed_surface_boundary_completion": True,
        "support_footprint_mode": x72.SUPPORT_MODE,
    }
    rows = {completion["track_id"]: completion}
    late = {X72_FLAG: True, "minimum_entry_s": x24.HOLD_WINDOW_S + 0.1}
    x24.require(
        _release_partition(late, rows, set(rows))
        and not _release_partition(
            {**late, "motion_evidence_credit_used": True}, rows, set(rows)
        )
        and not _release_partition(
            {X72_FLAG: True, "minimum_entry_s": x24.HOLD_WINDOW_S}, rows, set(rows)
        )
        and not _release_partition(
            late,
            {
                completion["track_id"]: {
                    **completion,
                    "x72_credentialed_surface_boundary_completion": False,
                }
            },
            set(rows),
        ),
        "x87_solo_completion_horizon_partition",
    )
    return {
        "status": "X87_SOLO_COMPLETION_HORIZON_FALSIFIER_MET",
        "release_only": True,
        "sole_x72_mechanism_required": True,
        "inherited_x24_hold_window_reused": True,
        "independent_mechanisms_retained": True,
        "direct_or_mixed_carriers_retained": True,
        "current_or_immediate_entries_retained": True,
        "new_metric_threshold_added": False,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), sort_keys=True))
