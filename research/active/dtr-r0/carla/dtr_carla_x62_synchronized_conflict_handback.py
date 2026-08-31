"""X62 synchronized conflict handback around frozen X59 and X61.

X61 showed that an X44 surface conflict must not globally veto an independent
issued-plan metric route entry.  X62 makes that handback evidence-synchronous:
the credential can be born only when both the metric route entry and at least
one suppressed surface component are currently MEASURED.  The metric motion
must also be non-receding in route coordinates: longitudinally closing, or
lateral-dominant when its longitudinal component is receding.  The same metric
identity may then continue on HOLD while X61's conflict-localized evidence
remains active.

These are qualitative evidence and vector-order rules.  No detector, duration,
weather label, absolute speed cutoff, or other numeric threshold is added.
C26-C28 remain consumed synthetic Development only.
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


EXPERIMENT_ID = "DTR_CARLA_X62_SYNCHRONIZED_CONFLICT_HANDBACK"
ARM_X62 = "X62_ISSUED_PLAN_SYNCHRONIZED_CONFLICT_HANDBACK"


def fixed_constants() -> dict[str, Any]:
    return {
        **x61.fixed_constants(),
        "representation": "X61_WITH_SYNCHRONIZED_CONFLICT_CREDENTIAL_BIRTH",
        "retained_core": "X59",
        "credential_birth_rule": (
            "CURRENT_X24_MEASURED_ROUTE_ENTRY_AND_CURRENT_MEASURED_X44_"
            "SUPPRESSED_SURFACE_SUPPORT"
        ),
        "metric_motion_rule": (
            "FORWARD_CLOSING_OR_LATERAL_MAGNITUDE_DOMINATES_RECEDING_FORWARD"
        ),
        "credential_hold_rule": "SAME_X61_METRIC_IDENTITY_ON_HOLD",
        "absolute_speed_threshold_added": False,
        "weather_or_lighting_label_used": False,
        "new_numeric_threshold_added": False,
    }


def _motion_supported(row: Mapping[str, Any]) -> bool:
    forward = float(row["velocity_forward_mps"])
    right = float(row["velocity_right_mps"])
    return forward < 0.0 or abs(right) > abs(forward)


def apply_synchronized_conflict_handback_episode(
    core: dict[str, Any], metric: dict[str, Any]
) -> dict[str, Any]:
    baseline = x59.apply_modality_evidence_reliability_router_episode(core, metric)
    candidate = x61.apply_conflict_localized_route_entry_episode(core, metric)
    value = copy.deepcopy(baseline)
    credentialed_source_ids: set[str] = set()
    measured_frames = 0
    held_frames = 0
    rejected_unsynchronized_births = 0
    rejected_motion_births = 0

    for ordinal, (base_frame, candidate_frame) in enumerate(
        zip(value["frames"], candidate["frames"], strict=True)
    ):
        candidate_arm = candidate_frame["arms"][x61.ARM_X61]
        measured_action = bool(
            candidate_arm.get("x61_measured_conflict_localized_handback_used", False)
        )
        held_action = bool(
            candidate_arm.get("x61_held_conflict_localized_handback_used", False)
        )
        if not (measured_action or held_action):
            continue

        carriers = [
            row
            for row in candidate_frame["tracks"]
            if bool(row.get("x61_conflict_localized_route_entry_handback", False))
        ]
        source_ids = {
            str(row["x60_metric_source_track_id"]) for row in carriers
        }
        suppressed_ids = {
            str(track_id)
            for track_id in candidate_arm.get(
                "x44_velocity_cycle_suppressed_track_ids", []
            )
        }
        current_measured_surface_support = any(
            str(row.get("track_id")) in suppressed_ids
            and row.get("disposition") == "MEASURED"
            for row in candidate_frame["tracks"]
        )
        motion_supported = bool(carriers) and all(
            _motion_supported(row) for row in carriers
        )

        accepted = False
        if measured_action:
            if not current_measured_surface_support:
                rejected_unsynchronized_births += 1
            elif not motion_supported:
                rejected_motion_births += 1
            else:
                credentialed_source_ids.update(source_ids)
                accepted = True
                measured_frames += 1
        elif source_ids and source_ids.issubset(credentialed_source_ids):
            accepted = True
            held_frames += 1

        if accepted:
            value["frames"][ordinal] = copy.deepcopy(candidate_frame)

    value["arms"][ARM_X62] = value["arms"].pop(x59.ARM_X59)
    value["diagnostics"]["x62_route_mode_counts"] = value["diagnostics"].pop(
        "x59_route_mode_counts"
    )
    value["diagnostics"].update(
        {
            "x62_measured_synchronized_conflict_handback_frames": measured_frames,
            "x62_held_synchronized_conflict_handback_frames": held_frames,
            "x62_rejected_unsynchronized_births": rejected_unsynchronized_births,
            "x62_rejected_motion_births": rejected_motion_births,
        }
    )
    for frame in value["frames"]:
        source_arm = (
            x61.ARM_X61 if x61.ARM_X61 in frame["arms"] else x59.ARM_X59
        )
        arm = frame["arms"].pop(source_arm)
        arm["x62_synchronized_conflict_handback_used"] = bool(
            arm.get("x61_measured_conflict_localized_handback_used", False)
            or arm.get("x61_held_conflict_localized_handback_used", False)
        )
        frame["arms"][ARM_X62] = arm
    return value


def predict_episode(
    episode: Any,
    candidate_values: Sequence[Mapping[str, Any]],
    calibration: Any,
) -> dict[str, Any]:
    return apply_synchronized_conflict_handback_episode(
        x59.x54.predict_episode(episode, candidate_values, calibration),
        x24.predict_episode(episode, candidate_values, calibration),
    )


def self_check() -> dict[str, Any]:
    x24.require(
        _motion_supported(
            {"velocity_forward_mps": 0.051, "velocity_right_mps": -1.591}
        )
        and _motion_supported(
            {"velocity_forward_mps": -0.63, "velocity_right_mps": -0.53}
        )
        and not _motion_supported(
            {"velocity_forward_mps": 0.87, "velocity_right_mps": -0.54}
        ),
        "x62_qualitative_motion_partition",
    )
    return {
        "status": "X62_SYNCHRONIZED_CONFLICT_HANDBACK_FALSIFIER_MET",
        "retained_core": "X59",
        "current_surface_measurement_required_at_birth": True,
        "qualitative_vector_order_rule": True,
        "absolute_speed_threshold_added": False,
        "weather_or_lighting_label_used": False,
        "new_numeric_threshold_added": False,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), sort_keys=True))
