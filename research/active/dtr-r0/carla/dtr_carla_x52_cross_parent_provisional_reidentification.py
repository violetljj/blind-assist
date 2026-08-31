"""X52 cross-parent provisional-motion reidentification.

X52 narrows X51 to the tracking-fragmentation case.  A provisional lattice
motion observation may update an existing authorized belief only when the
observed parent differs from the belief parent and their inherited position
and velocity state cycle closes.  Same-parent observations cannot consume this
exception and remain governed by the inherited conflict/release behavior.

C22 is consumed posthoc synthetic Development and cannot confirm X52.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import dtr_carla_x24_plan_adherent_predictor as x24  # noqa: E402
import dtr_carla_x45_causal_state_cycle_credential as x45  # noqa: E402
import dtr_carla_x48_evidence_updated_object_permanence as x48  # noqa: E402
import dtr_carla_x51_provisional_motion_belief_update as x51  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X52_CROSS_PARENT_PROVISIONAL_REIDENTIFICATION"
ARM_X52 = "X52_ISSUED_PLAN_CROSS_PARENT_PROVISIONAL_REIDENTIFICATION"
BASE_CLOSES_STATE_CYCLE = x45.closes_state_cycle
BASE_CONTINUED_ROW = x51.continued_row


def fixed_constants() -> dict[str, Any]:
    return {
        **x51.fixed_constants(),
        "representation": "AUTHORIZED_BELIEF_WITH_CROSS_PARENT_PROVISIONAL_REIDENTIFICATION",
        "provisional_update_scope": "TRACK_PARENT_FRAGMENTATION_ONLY",
        "provisional_parent_change_required": True,
        "same_parent_provisional_update": False,
        "provisional_authority_origination": False,
        "new_numeric_threshold_added": False,
    }


def cross_parent_cycle(
    belief: Mapping[str, Any], observation: Mapping[str, Any]
) -> bool:
    return (
        x48.parent_id(belief) != x48.parent_id(observation)
        and BASE_CLOSES_STATE_CYCLE(belief, observation)
    )


def cross_parent_continued_row(
    witness: Mapping[str, Any], belief: Mapping[str, Any], now_s: float
) -> dict[str, Any]:
    value = BASE_CONTINUED_ROW(witness, belief, now_s)
    value["track_id"] = f"x52-reidentified::{witness['track_id']}"
    return value


def apply_cross_parent_episode(
    episode: Any,
    value: dict[str, Any],
    calibration: Any,
) -> dict[str, Any]:
    original_cycle = x51.x45.closes_state_cycle
    original_continued_row = x51.continued_row
    x51.x45.closes_state_cycle = cross_parent_cycle
    x51.continued_row = cross_parent_continued_row
    try:
        result = x51.apply_provisional_motion_episode(
            episode, value, calibration
        )
    finally:
        x51.x45.closes_state_cycle = original_cycle
        x51.continued_row = original_continued_row

    result["arms"][ARM_X52] = result["arms"].pop(x51.ARM_X51)
    result["diagnostics"]["x52_route_mode_counts"] = result["diagnostics"].pop(
        "x51_route_mode_counts"
    )
    for key in list(result["diagnostics"]):
        if key.startswith("x51_"):
            result["diagnostics"]["x52_" + key[4:]] = result["diagnostics"].pop(key)
    for frame in result["frames"]:
        arm = frame["arms"].pop(x51.ARM_X51)
        for key in list(arm):
            if key.startswith("x51_"):
                arm["x52_" + key[4:]] = arm.pop(key)
        frame["arms"][ARM_X52] = arm
    return result


def predict_episode(
    episode: Any,
    candidate_values: Sequence[Mapping[str, Any]],
    calibration: Any,
) -> dict[str, Any]:
    return apply_cross_parent_episode(
        episode,
        x45.predict_episode(episode, candidate_values, calibration),
        calibration,
    )


def self_check() -> dict[str, Any]:
    inherited = x51.self_check()
    belief = {
        "parent_track_id": "prior",
        "position_forward_m": 4.0,
        "position_right_m": 0.0,
        "velocity_forward_mps": -2.0,
        "velocity_right_mps": 0.0,
    }
    changed = {
        "parent_track_id": "current",
        "position_forward_m": 4.5,
        "position_right_m": 0.1,
        "velocity_forward_mps": -1.0,
        "velocity_right_mps": 0.0,
    }
    same = {**changed, "parent_track_id": "prior"}
    x24.require(cross_parent_cycle(belief, changed), "x52_cross_parent_cycle")
    x24.require(not cross_parent_cycle(belief, same), "x52_same_parent_rejected")
    return {
        "status": "X52_CROSS_PARENT_PROVISIONAL_REIDENTIFICATION_FALSIFIER_MET",
        "x51_structural_status": inherited["status"],
        "parent_change_required": True,
        "authority_origination_allowed": False,
        "position_threshold_inherited_from_x24": True,
        "new_numeric_threshold_added": False,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), sort_keys=True))
