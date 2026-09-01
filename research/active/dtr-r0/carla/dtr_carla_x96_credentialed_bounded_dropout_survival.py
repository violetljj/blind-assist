"""X96 survives bounded full dropout from the last real collision credential.

X94 bridges only the immediately next observation.  X96 instead retains the
last *measured* rigid surface carrier that already owns an X75 collision
credential and transports that anchor directly for at most the inherited X24
hold window.  A transported row is never a new anchor, so survival cannot
reseed itself.

Survival is allowed only while detector candidates and metric footprints are
both absent, the issued-plan receipt is unchanged and valid, and no current
release, route-mode change, parent contradiction, or foreign measured parent
is present.  Absence cannot create a risk because a prior measured,
credentialed route-risk carrier is mandatory.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Mapping


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import dtr_carla_x24_plan_adherent_predictor as x24  # noqa: E402
import dtr_carla_x27_occupancy_authority_predictor as x27  # noqa: E402
import dtr_carla_x73_credentialed_parent_hull_reconstruction as x73  # noqa: E402
import dtr_carla_x93_conflicted_nonclosing_future_release as x93  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X96_CREDENTIALED_BOUNDED_DROPOUT_SURVIVAL"
ARM_X96 = "X96_ISSUED_PLAN_CREDENTIALED_BOUNDED_DROPOUT_SURVIVAL"
SUPPORT_MODE = "LAST_REAL_MEASURED_CREDENTIAL_BOUNDED_DROPOUT_SURVIVAL"


def fixed_constants() -> dict[str, Any]:
    return {
        **x93.fixed_constants(),
        "representation": "X93_WITH_CREDENTIALED_BOUNDED_DROPOUT_SURVIVAL",
        "retained_core": "X93",
        "anchor_rule": (
            "LAST_REAL_MEASURED_RIGID_SURFACE_ROUTE_RISK_CARRIER_WITH_"
            "PRIOR_X75_COLLISION_CREDENTIAL"
        ),
        "full_dropout_rule": "ZERO_DETECTOR_CANDIDATES_AND_ZERO_METRIC_FOOTPRINTS",
        "route_continuity_rule": "UNCHANGED_VALID_ISSUED_PLAN_RECEIPT",
        "release_precedence": (
            "ZERO_ACTIVE_RELEASE_ROUTE_MODE_CHANGE_PARENT_CONTRADICTION_OR_"
            "FOREIGN_MEASURED_PARENT"
        ),
        "survival_window_s": x24.HOLD_WINDOW_S,
        "survival_can_reseed": False,
        "fill_rows_are_anchors": False,
        "detector_threshold_change": False,
        "route_threshold_change": False,
        "new_numeric_threshold_added": False,
        "weather_or_lighting_label_used": False,
    }


def _parent_id(row: Mapping[str, Any]) -> str:
    return str(row.get("parent_track_id") or row["track_id"])


def _active_release(arm: Mapping[str, Any]) -> bool:
    return any(
        key.endswith("_release_used") and bool(value)
        for key, value in arm.items()
    )


def _is_real_measured_surface(row: Mapping[str, Any]) -> bool:
    return (
        row.get("class_name") == x73.SURFACE_CLASS
        and row.get("disposition") == "MEASURED"
        and float(row.get("evidence_age_s", float("inf"))) <= x24.EPSILON
        and row.get("motion_authority") == x27.RIGID_DYNAMIC
        and bool(row.get("risk_eligible"))
        and bool(row.get("footprint_xy"))
        and not bool(row.get("x94_one_frame_full_dropout_continuity"))
        and not bool(row.get("x96_bounded_dropout_survival"))
    )


def _current_conflict(frame: Mapping[str, Any], parent_id: str) -> bool:
    for row in frame["tracks"]:
        row_parent = _parent_id(row)
        if row_parent == parent_id and int(row.get("transport_contradictions", 0)) > 0:
            return True
        if (
            row_parent != parent_id
            and row.get("disposition") == "MEASURED"
            and row.get("class_name") == x73.SURFACE_CLASS
            and bool(row.get("risk_eligible"))
        ):
            return True
    return False


def _transport_anchor(
    row: Mapping[str, Any], now_s: float, anchor_time_s: float
) -> dict[str, Any]:
    value = copy.deepcopy(dict(row))
    elapsed = float(now_s) - float(anchor_time_s)
    parent_id = _parent_id(row)
    value.update(
        {
            "track_id": f"x96-bounded-survival::{parent_id}::{now_s:.6f}",
            "parent_track_id": parent_id,
            "position_forward_m": float(row["position_forward_m"])
            + float(row["velocity_forward_mps"]) * elapsed,
            "position_right_m": float(row["position_right_m"])
            + float(row["velocity_right_mps"]) * elapsed,
            "disposition": "HOLD",
            "evidence_age_s": elapsed,
            "risk_eligible": True,
            "motion_authority": x27.RIGID_DYNAMIC,
            "support_footprint_mode": SUPPORT_MODE,
            "x96_bounded_dropout_survival": True,
            "x96_anchor_track_id": str(row["track_id"]),
            "x96_anchor_time_s": float(anchor_time_s),
        }
    )
    return value


def apply_credentialed_bounded_dropout_survival_episode(
    core: dict[str, Any]
) -> dict[str, Any]:
    value = copy.deepcopy(core)
    credentialed_parent_ids: set[str] = set()
    anchors: dict[str, tuple[dict[str, Any], float, float, str]] = {}
    survival_frames = 0
    survival_tracks = 0
    expired_rejections = 0
    conflict_rejections = 0
    no_anchor_rejections = 0

    for frame in value["frames"]:
        arm = frame["arms"][x93.ARM_X93]
        credentialed_parent_ids.update(
            str(parent_id)
            for parent_id in arm.get("x75_collision_credential_birth_parent_ids", [])
        )
        arm["x96_bounded_dropout_survival_used"] = False
        arm["x96_survival_parent_ids"] = []
        arm["x96_survival_track_ids"] = []
        arm["x96_anchor_track_ids"] = []

        rows = {str(row["track_id"]): row for row in frame["tracks"]}
        if bool(arm.get("route_risk")):
            for track_id in map(str, arm.get("confirmed_risk_track_ids", [])):
                row = rows.get(track_id)
                if row is None or not _is_real_measured_surface(row):
                    continue
                parent_id = _parent_id(row)
                if parent_id not in credentialed_parent_ids:
                    continue
                entry = arm.get("minimum_entry_s")
                if entry is None:
                    continue
                anchors[parent_id] = (
                    copy.deepcopy(dict(row)),
                    float(frame["time_s"]),
                    float(entry),
                    str(arm.get("plan_receipt_sha256") or ""),
                )
            continue

        if int(frame.get("raw_candidates", 0)) != 0 or int(
            frame.get("metric_footprint_measurements", 0)
        ) != 0:
            continue
        if (
            arm.get("authority") != "VALID"
            or arm.get("route_mode") != "ISSUED_PLAN"
            or bool(arm.get("route_mode_changed"))
            or _active_release(arm)
        ):
            conflict_rejections += 1
            continue

        now_s = float(frame["time_s"])
        carriers: list[dict[str, Any]] = []
        source_entries: list[float] = []
        anchor_ids: list[str] = []
        for parent_id, (anchor, anchor_time_s, anchor_entry_s, plan_receipt) in sorted(
            anchors.items()
        ):
            elapsed = now_s - anchor_time_s
            if elapsed <= x24.EPSILON or elapsed > x24.HOLD_WINDOW_S + x24.EPSILON:
                if elapsed > x24.HOLD_WINDOW_S + x24.EPSILON:
                    expired_rejections += 1
                continue
            if str(arm.get("plan_receipt_sha256") or "") != plan_receipt:
                conflict_rejections += 1
                continue
            if _current_conflict(frame, parent_id):
                conflict_rejections += 1
                continue
            carriers.append(_transport_anchor(anchor, now_s, anchor_time_s))
            source_entries.append(max(0.0, anchor_entry_s - elapsed))
            anchor_ids.append(str(anchor["track_id"]))

        if not carriers:
            no_anchor_rejections += 1
            continue

        existing_ids = {str(row["track_id"]) for row in frame["tracks"]}
        for carrier in carriers:
            x24.require(
                str(carrier["track_id"]) not in existing_ids,
                "x96_survival_track_id_collision",
            )
            frame["tracks"].append(carrier)
            frame["risk_eligible_tracks"] = int(frame["risk_eligible_tracks"]) + 1
        track_ids = sorted(str(row["track_id"]) for row in carriers)
        parent_ids = sorted({_parent_id(row) for row in carriers})
        arm.update(
            {
                "route_risk": True,
                "minimum_entry_s": min(source_entries),
                "candidate_risk_track_ids": track_ids,
                "confirmed_risk_track_ids": track_ids,
                "candidate_risk_parent_track_ids": parent_ids,
                "confirmed_risk_parent_track_ids": parent_ids,
                "x96_bounded_dropout_survival_used": True,
                "x96_survival_parent_ids": parent_ids,
                "x96_survival_track_ids": track_ids,
                "x96_anchor_track_ids": sorted(anchor_ids),
            }
        )
        survival_frames += 1
        survival_tracks += len(carriers)

    value["arms"][ARM_X96] = value["arms"].pop(x93.ARM_X93)
    value["diagnostics"]["x96_route_mode_counts"] = value["diagnostics"].pop(
        "x93_route_mode_counts"
    )
    value["diagnostics"].update(
        {
            "x96_bounded_dropout_survival_frames": survival_frames,
            "x96_bounded_dropout_survival_tracks": survival_tracks,
            "x96_expired_anchor_rejections": expired_rejections,
            "x96_conflict_rejections": conflict_rejections,
            "x96_no_anchor_rejections": no_anchor_rejections,
        }
    )
    for frame in value["frames"]:
        frame["arms"][ARM_X96] = frame["arms"].pop(x93.ARM_X93)
    return value


def self_check() -> dict[str, Any]:
    row = {
        "track_id": "surface-parent-hull",
        "parent_track_id": "surface-parent",
        "class_name": x73.SURFACE_CLASS,
        "disposition": "MEASURED",
        "evidence_age_s": 0.0,
        "motion_authority": x27.RIGID_DYNAMIC,
        "risk_eligible": True,
        "position_forward_m": 2.0,
        "position_right_m": 0.5,
        "velocity_forward_mps": -1.0,
        "velocity_right_mps": 0.25,
        "footprint_xy": [[1.5, 0.0], [2.5, 0.0], [2.5, 1.0], [1.5, 1.0]],
    }
    carrier = _transport_anchor(row, 1.3, 1.0)
    x24.require(
        _is_real_measured_surface(row)
        and not _is_real_measured_surface({**row, "disposition": "HOLD"})
        and abs(float(carrier["position_forward_m"]) - 1.7) <= x24.EPSILON
        and abs(float(carrier["position_right_m"]) - 0.575) <= x24.EPSILON,
        "x96_real_anchor_transport",
    )
    x24.require(
        _active_release({"x93_conflicted_nonclosing_future_release_used": True})
        and not _active_release({"x75_collision_credential_birth_parent_ids": ["p"]}),
        "x96_release_precedence_partition",
    )
    x24.require(
        _current_conflict(
            {
                "tracks": [
                    {
                        **row,
                        "track_id": "other",
                        "parent_track_id": "other-parent",
                    }
                ]
            },
            "surface-parent",
        ),
        "x96_foreign_measured_parent_conflict",
    )
    return {
        "status": "X96_CREDENTIALED_BOUNDED_DROPOUT_SURVIVAL_FALSIFIER_MET",
        "real_measured_anchor_required": True,
        "prior_collision_credential_required": True,
        "unchanged_issued_plan_receipt_required": True,
        "release_and_conflict_precedence": True,
        "survival_window_s": x24.HOLD_WINDOW_S,
        "fill_can_reseed": False,
        "new_numeric_threshold_added": False,
    }


if __name__ == "__main__":
    print(json.dumps(self_check(), sort_keys=True))
