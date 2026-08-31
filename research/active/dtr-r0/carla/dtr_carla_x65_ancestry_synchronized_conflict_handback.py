"""X65 ancestry-aware synchronized conflict handback around frozen X64.

X62 can birth a conflict credential only after the conflict is already active.
If the metric representation is on HOLD at that first conflict frame, prior
agreement between X64 surface risk and the same X24 metric identity is lost.
X65 records that pre-conflict cross-representation agreement.  A handback still
requires current measured support from the suppressed surface lineage at its
first conflict frame; only then may the same credentialed metric identity
continue on HOLD.  Suppressed child IDs are resolved through stable parent
ancestry so a current measured sibling can satisfy the synchronization join.

No detector, route, duration, distance, speed, weather label, or numeric
threshold is added.  C26-C28 and C32 are consumed synthetic Development only.
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
import dtr_carla_x59_modality_evidence_reliability_router as x59  # noqa: E402
import dtr_carla_x61_conflict_localized_route_entry as x61  # noqa: E402
import dtr_carla_x62_synchronized_conflict_handback as x62  # noqa: E402
import dtr_carla_x64_unanchored_crossing_release as x64  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X65_ANCESTRY_SYNCHRONIZED_CONFLICT_HANDBACK"
ARM_X65 = "X65_ISSUED_PLAN_ANCESTRY_SYNCHRONIZED_CONFLICT_HANDBACK"


def fixed_constants() -> dict[str, Any]:
    return {
        **x64.fixed_constants(),
        "representation": "X64_WITH_ANCESTRY_AWARE_X62_SYNCHRONIZATION",
        "retained_core": "X64",
        "credential_birth_rule": (
            "PRECONFLICT_X64_AND_X24_ROUTE_RISK_AGREEMENT_FOR_SAME_X24_IDENTITY"
        ),
        "conflict_activation_rule": (
            "CURRENT_MEASURED_X44_SUPPRESSED_LINEAGE_SUPPORT_REQUIRED"
        ),
        "credential_hold_rule": "SAME_X24_IDENTITY_CONFIRMED_ON_HOLD",
        "synchronization_identity": (
            "CURRENT_SURFACE_TRACK_ID_OR_STABLE_PARENT_TRACK_ID_MATCHES_"
            "X44_SUPPRESSED_LINEAGE"
        ),
        "x62_motion_support_rule_unchanged": True,
        "x62_credential_birth_rule_extended": True,
        "x64_crossing_release_rule_unchanged": True,
        "weather_or_lighting_label_used": False,
        "new_numeric_threshold_added": False,
    }


def _matches_suppressed_lineage(
    row: Mapping[str, Any], suppressed_lineage_ids: set[str]
) -> bool:
    return (
        str(row.get("track_id")) in suppressed_lineage_ids
        or str(row.get("parent_track_id")) in suppressed_lineage_ids
    )


def _suppressed_lineage_ids(
    tracks: Sequence[Mapping[str, Any]], suppressed_ids: set[str]
) -> set[str]:
    lineage_ids = set(suppressed_ids)
    lineage_ids.update(
        str(row["parent_track_id"])
        for row in tracks
        if str(row.get("track_id")) in suppressed_ids
        and row.get("parent_track_id")
    )
    return lineage_ids


def apply_ancestry_synchronized_conflict_handback_episode(
    core: dict[str, Any], metric: dict[str, Any]
) -> dict[str, Any]:
    baseline = x59.apply_modality_evidence_reliability_router_episode(core, metric)
    value = copy.deepcopy(baseline)
    credentialed_source_ids: set[str] = set()
    active_handoff_source_ids: set[str] = set()
    precredential_frames = 0
    measured_frames = 0
    held_frames = 0
    ancestry_matches = 0
    rejected_unsynchronized_births = 0
    rejected_motion_births = 0

    for base_frame, metric_frame in zip(
        value["frames"], metric["frames"], strict=True
    ):
        x24.require(
            int(base_frame["sample_index"]) == int(metric_frame["sample_index"]),
            "x65_frame_alignment",
        )
        arm = base_frame["arms"][x59.ARM_X59]
        metric_arm = metric_frame["arms"][x24.ARM_X24]
        arm["x61_measured_conflict_localized_handback_used"] = False
        arm["x61_held_conflict_localized_handback_used"] = False
        arm["x61_conflict_localized_track_ids"] = []
        arm["x65_parent_ancestry_synchronization_used"] = False
        arm["x65_preconflict_credentialed_handback_used"] = False

        active_conflict = (
            bool(arm.get("x44_velocity_cycle_suppressed", False))
            and arm.get("x44_velocity_cycle_suppression_reason")
            == x61.X44_EDGE_CONFLICT
            and bool(arm.get("x44_velocity_cycle_suppressed_track_ids", []))
        )
        valid_metric_route = (
            bool(metric_arm.get("route_risk"))
            and metric_arm.get("authority") == "VALID"
            and metric_arm.get("route_mode") == "ISSUED_PLAN"
        )
        metric_tracks = {
            str(row["track_id"]): row for row in metric_frame["tracks"]
        }
        confirmed_ids = {
            str(track_id)
            for track_id in metric_arm.get("confirmed_risk_track_ids", [])
            if str(track_id) in metric_tracks
        }
        credentialed_source_ids = (
            credentialed_source_ids & confirmed_ids
            if valid_metric_route
            else set()
        )
        if bool(arm.get("route_risk")) and valid_metric_route and not active_conflict:
            before = len(credentialed_source_ids)
            credentialed_source_ids.update(confirmed_ids)
            precredential_frames += int(len(credentialed_source_ids) > before)

        if not active_conflict:
            active_handoff_source_ids.clear()
            continue
        suppressed_ids = {
            str(track_id)
            for track_id in arm.get(
                "x44_velocity_cycle_suppressed_track_ids", []
            )
        }
        suppressed_lineage_ids = _suppressed_lineage_ids(
            base_frame["tracks"], suppressed_ids
        )
        current_rows = [
            row
            for row in base_frame["tracks"]
            if row.get("disposition") == "MEASURED"
            and _matches_suppressed_lineage(row, suppressed_lineage_ids)
        ]
        current_measured_surface_support = bool(current_rows)
        current_ancestry_support = any(
            str(row.get("track_id")) not in suppressed_ids
            and str(row.get("parent_track_id")) in suppressed_lineage_ids
            for row in current_rows
        )
        source_rows = [
            metric_tracks[track_id]
            for track_id in sorted(credentialed_source_ids & confirmed_ids)
            if metric_tracks[track_id].get("disposition") in {"MEASURED", "HOLD"}
            and x62._motion_supported(metric_tracks[track_id])
        ]
        source_ids = {str(row["track_id"]) for row in source_rows}
        active_handoff_source_ids.intersection_update(source_ids)
        first_activation = not bool(active_handoff_source_ids)
        accepted = bool(source_rows)
        if accepted and first_activation:
            if not current_measured_surface_support:
                rejected_unsynchronized_births += 1
                accepted = False
            else:
                active_handoff_source_ids.update(source_ids)
        elif accepted and not source_ids.issubset(active_handoff_source_ids):
            accepted = False
        if not source_rows and credentialed_source_ids & confirmed_ids:
            rejected_motion_births += int(first_activation)

        if bool(arm.get("x59_evidence_supported_receding_release_used", False)):
            active_handoff_source_ids.clear()
            continue
        if not accepted:
            continue
        ancestry_matches += int(current_ancestry_support)
        carriers = [
            x61._carrier(row, held=row.get("disposition") == "HOLD")
            for row in source_rows
        ]
        existing_ids = {str(row["track_id"]) for row in base_frame["tracks"]}
        for row in carriers:
            x24.require(
                str(row["track_id"]) not in existing_ids,
                "x65_handback_track_id_collision",
            )
            base_frame["tracks"].append(row)
            base_frame["risk_eligible_tracks"] = int(
                base_frame["risk_eligible_tracks"]
            ) + 1
        track_ids = sorted(str(row["track_id"]) for row in carriers)
        parent_ids = sorted(str(row["parent_track_id"]) for row in carriers)
        measured_sources = [
            row for row in source_rows if row.get("disposition") == "MEASURED"
        ]
        held_sources = [
            row for row in source_rows if row.get("disposition") == "HOLD"
        ]
        arm.update(
            {
                "route_risk": True,
                "minimum_entry_s": metric_arm.get("minimum_entry_s"),
                "candidate_risk_track_ids": track_ids,
                "confirmed_risk_track_ids": track_ids,
                "candidate_risk_parent_track_ids": parent_ids,
                "confirmed_risk_parent_track_ids": parent_ids,
                "x61_measured_conflict_localized_handback_used": bool(
                    measured_sources
                ),
                "x61_held_conflict_localized_handback_used": bool(held_sources),
                "x61_conflict_localized_track_ids": track_ids,
                "x65_parent_ancestry_synchronization_used": bool(
                    current_ancestry_support
                ),
                "x65_preconflict_credentialed_handback_used": True,
            }
        )
        measured_frames += int(bool(measured_sources))
        held_frames += int(bool(held_sources))

    value["arms"][x62.ARM_X62] = value["arms"].pop(x59.ARM_X59)
    value["diagnostics"]["x62_route_mode_counts"] = value["diagnostics"].pop(
        "x59_route_mode_counts"
    )
    value["diagnostics"].update(
        {
            "x62_measured_synchronized_conflict_handback_frames": measured_frames,
            "x62_held_synchronized_conflict_handback_frames": held_frames,
            "x62_rejected_unsynchronized_births": rejected_unsynchronized_births,
            "x62_rejected_motion_births": rejected_motion_births,
            "x65_parent_ancestry_synchronization_frames": ancestry_matches,
            "x65_preconflict_joint_credential_frames": precredential_frames,
        }
    )
    for frame in value["frames"]:
        arm = frame["arms"].pop(x59.ARM_X59)
        arm["x62_synchronized_conflict_handback_used"] = bool(
            arm.get("x61_measured_conflict_localized_handback_used", False)
            or arm.get("x61_held_conflict_localized_handback_used", False)
        )
        frame["arms"][x62.ARM_X62] = arm
    return value


def apply_ancestry_handback_episode(
    core: dict[str, Any], metric: dict[str, Any]
) -> dict[str, Any]:
    original = x64.x62.apply_synchronized_conflict_handback_episode
    x64.x62.apply_synchronized_conflict_handback_episode = (
        apply_ancestry_synchronized_conflict_handback_episode
    )
    try:
        value = x64.apply_unanchored_crossing_release_episode(core, metric)
    finally:
        x64.x62.apply_synchronized_conflict_handback_episode = original

    value["arms"][ARM_X65] = value["arms"].pop(x64.ARM_X64)
    value["diagnostics"]["x65_route_mode_counts"] = value["diagnostics"].pop(
        "x64_route_mode_counts"
    )
    for frame in value["frames"]:
        frame["arms"][ARM_X65] = frame["arms"].pop(x64.ARM_X64)
    return value


def predict_episode(
    episode: Any,
    candidate_values: Sequence[Mapping[str, Any]],
    calibration: Any,
) -> dict[str, Any]:
    return apply_ancestry_handback_episode(
        x64.x62.x59.x54.predict_episode(episode, candidate_values, calibration),
        x24.predict_episode(episode, candidate_values, calibration),
    )


def self_check() -> dict[str, Any]:
    suppressed = {"surface-cone-000020::cone-old"}
    tracks = [
        {
            "track_id": "surface-cone-000020::cone-old",
            "parent_track_id": "surface-cone-000020",
        }
    ]
    lineages = _suppressed_lineage_ids(tracks, suppressed)
    x24.require(
        _matches_suppressed_lineage(
            {
                "track_id": "surface-cone-000020::cone-child",
                "parent_track_id": "surface-cone-000020",
            },
            lineages,
        )
        and _matches_suppressed_lineage(
            {
                "track_id": "surface-cone-000020::cone-old",
                "parent_track_id": "surface-cone-000020",
            },
            lineages,
        )
        and not _matches_suppressed_lineage(
            {
                "track_id": "surface-cone-000021::cone-child",
                "parent_track_id": "surface-cone-000021",
            },
            lineages,
        ),
        "x65_ancestry_identity_partition",
    )
    return {
        "status": "X65_ANCESTRY_SYNCHRONIZATION_FALSIFIER_MET",
        "retained_core": "X64",
        "stable_parent_ancestry_match_enabled": True,
        "x62_motion_support_rule_unchanged": True,
        "x62_credential_birth_rule_extended": True,
        "x64_crossing_release_rule_unchanged": True,
        "weather_or_lighting_label_used": False,
        "new_numeric_threshold_added": False,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), sort_keys=True))
