"""X55 parent-sibling state-cycle consensus for fragmented occupancy.

X55 replaces X52's unexercised cross-parent exception with a same-parent
distributed state cycle.  A parent that already carries causal motion evidence
(or whose previously confirmed X45 row was suppressed) may recover route risk
from a same-frame pair of distinct measured children.  Both children must
retain rigid-dynamic authority, depth support, risk eligibility, the inherited
X24 position association, and a positive velocity dot product.  At least one
member of the closing pair must enter the current issued route.

The rule treats fragmented components as distributed observations of one
parent rather than requiring one component to close the entire state cycle.
It adds no detector, score, speed, duration, association, or route threshold.

C24 is consumed synthetic Development and cannot confirm X55.
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
import dtr_carla_x27_occupancy_authority_predictor as x27  # noqa: E402
import dtr_carla_x45_causal_state_cycle_credential as x45  # noqa: E402
import dtr_carla_x51_provisional_motion_belief_update as x51  # noqa: E402
import dtr_carla_x52_cross_parent_provisional_reidentification as x52  # noqa: E402
import dtr_carla_x53_anchor_redundant_parent_continuation as x53  # noqa: E402
import dtr_carla_x54_metric_bootstrap_dropout_continuation as x54  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X55_PARENT_SIBLING_STATE_CYCLE_CONSENSUS"
ARM_X55 = "X55_ISSUED_PLAN_PARENT_SIBLING_STATE_CYCLE_CONSENSUS"


def fixed_constants() -> dict[str, Any]:
    return {
        **x54.fixed_constants(),
        "representation": "PARENT_DISTRIBUTED_SIBLING_STATE_CYCLE",
        "parent_authority_rule": (
            "PARENT_ALREADY_HAS_CAUSAL_MOTION_CREDIT_OR_X45_SUPPRESSED_CONFIRMATION"
        ),
        "sibling_cycle_rule": (
            "DISTINCT_CURRENT_MEASURED_DEPTH_SUPPORTED_RIGID_DYNAMIC_CHILDREN_"
            "CLOSE_INHERITED_X24_POSITION_AND_POSITIVE_VELOCITY_EDGES"
        ),
        "route_rule": "TWO_DISTINCT_CLOSING_SIBLINGS_ENTER_CURRENT_ISSUED_ROUTE",
        "sibling_count_rule": "ANOTHER_DISTINCT_CHILD_REQUIRED",
        "provisional_update_scope": "SAME_PARENT_AFTER_DISTRIBUTED_CYCLE_AUTHORITY",
        "x52_cross_parent_exception_replaced": True,
        "inherited_x53_anchor_redundancy": True,
        "inherited_x54_metric_bootstrap_dropout": True,
        "position_association_distance_m": x24.ASSOCIATION_DISTANCE_M,
        "position_threshold_source": "INHERITED_X24_ASSOCIATION_DISTANCE",
        "class_rule": "CLASS_INDEPENDENT",
        "new_numeric_threshold_added": False,
    }


def qualified_child(row: Mapping[str, Any]) -> bool:
    return (
        row.get("disposition") == "MEASURED"
        and row.get("motion_authority") == x27.RIGID_DYNAMIC
        and bool(row.get("risk_eligible", False))
        and row.get("depth_grid_support") is not None
    )


def closing_sibling_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    closing_ids: set[str] = set()
    for ordinal, left in enumerate(rows):
        if not qualified_child(left):
            continue
        for right in rows[ordinal + 1 :]:
            if (
                not qualified_child(right)
                or str(left["track_id"]) == str(right["track_id"])
                or str(left["parent_track_id"]) != str(right["parent_track_id"])
                or not x45.closes_state_cycle(left, right)
            ):
                continue
            closing_ids.update((str(left["track_id"]), str(right["track_id"])))
    return [dict(row) for row in rows if str(row["track_id"]) in closing_ids]


def apply_parent_sibling_seed_episode(
    episode: Any,
    value: dict[str, Any],
    calibration: Any,
) -> dict[str, Any]:
    del calibration
    value = copy.deepcopy(value)
    previous_mode: str | None = None
    receipt_cache: dict[Path, dict[str, Any]] = {}
    consensus_frames = 0
    x45_recovery_frames = 0
    motion_credit_recovery_frames = 0

    for observation, frame in zip(
        episode.observations, value["frames"], strict=True
    ):
        arm = frame["arms"][x45.ARM_X45]
        arm["x55_parent_sibling_consensus_used"] = False
        arm["x55_parent_sibling_consensus_track_ids"] = []
        arm["x55_parent_sibling_consensus_parent_ids"] = []
        arm["x55_x45_suppression_adjudicated"] = False
        wearer_position, wearer_velocity = x24.wearer_anchor_state(
            observation, episode.route_frame
        )
        receipt = x24.load_receipt(observation, receipt_cache)
        selection = x24.route.select_route(
            receipt,
            session_id=observation.navigation_session_id,
            now_s=observation.time_s,
            wearer_position_xy=wearer_position,
            wearer_velocity_xy=wearer_velocity,
            previous_mode=previous_mode,
        )
        previous_mode = selection.mode
        tracks = {str(row["track_id"]): row for row in frame["tracks"]}
        suppressed_ids = {
            str(track_id)
            for track_id in arm.get("x45_state_cycle_suppressed_track_ids", [])
        }
        x45_parents = {
            str(tracks[track_id]["parent_track_id"])
            for track_id in suppressed_ids
            if track_id in tracks
        }
        motion_credit_parents = {
            str(parent_id)
            for parent_id in arm.get(
                "motion_evidence_credit_parent_track_ids", []
            )
        }
        eligible_parents = x45_parents | motion_credit_parents
        if not eligible_parents:
            continue

        closing_by_parent: dict[str, list[dict[str, Any]]] = {}
        for parent_id in sorted(eligible_parents):
            siblings = [
                row
                for row in frame["tracks"]
                if str(row.get("parent_track_id")) == parent_id
            ]
            closing = closing_sibling_rows(siblings)
            if closing:
                closing_by_parent[parent_id] = closing
        if not closing_by_parent:
            continue

        # A distributed same-parent cycle resolves X45's single-child conflict
        # even when the current child centroids do not themselves enter route.
        resolved_x45_parents = set(closing_by_parent) & x45_parents
        if bool(resolved_x45_parents):
            arm["x45_state_cycle_suppressed"] = False
            arm["x55_x45_suppression_adjudicated"] = True
            x45_recovery_frames += 1
        else:
            arm["x55_x45_suppression_adjudicated"] = False

        if bool(arm.get("route_risk")):
            continue

        carriers: list[dict[str, Any]] = []
        entries: list[float] = []
        carrier_parent_ids: set[str] = set()
        for parent_id, closing in closing_by_parent.items():
            for row in closing:
                entry = x24.route.first_selected_route_entry_s(
                    selection,
                    receipt=receipt,
                    now_s=observation.time_s,
                    wearer_position_xy=wearer_position,
                    wearer_velocity_xy=wearer_velocity,
                    target_position_xy=(
                        float(row["position_forward_m"]),
                        float(row["position_right_m"]),
                    ),
                    target_velocity_xy=(
                        float(row["velocity_forward_mps"]),
                        float(row["velocity_right_mps"]),
                    ),
                )
                if entry is None:
                    continue
                carriers.append(row)
                entries.append(float(entry))
                carrier_parent_ids.add(parent_id)
        track_ids = sorted({str(row["track_id"]) for row in carriers})
        if len(track_ids) <= 1:
            continue

        parent_ids = sorted(carrier_parent_ids)
        for track_id in track_ids:
            tracks[track_id]["x55_parent_sibling_consensus_seed"] = True
        arm.update(
            {
                "route_risk": True,
                "minimum_entry_s": min(entries),
                "candidate_risk_track_ids": track_ids,
                "confirmed_risk_track_ids": track_ids,
                "candidate_risk_parent_track_ids": parent_ids,
                "confirmed_risk_parent_track_ids": parent_ids,
                "x55_parent_sibling_consensus_used": True,
                "x55_parent_sibling_consensus_track_ids": track_ids,
                "x55_parent_sibling_consensus_parent_ids": parent_ids,
            }
        )
        consensus_frames += 1
        motion_credit_recovery_frames += int(
            bool(carrier_parent_ids & motion_credit_parents)
        )

    value["diagnostics"].update(
        {
            "x55_parent_sibling_consensus_frames": consensus_frames,
            "x55_x45_suppression_recovery_frames": x45_recovery_frames,
            "x55_motion_credit_recovery_frames": motion_credit_recovery_frames,
        }
    )
    return value


def promote_x51_to_x52_contract(value: dict[str, Any]) -> dict[str, Any]:
    """Expose X51 same-parent updates through X53's inherited input contract."""
    value["arms"][x52.ARM_X52] = value["arms"].pop(x51.ARM_X51)
    value["diagnostics"]["x52_route_mode_counts"] = value["diagnostics"].pop(
        "x51_route_mode_counts"
    )
    for key in list(value["diagnostics"]):
        if key.startswith("x51_"):
            value["diagnostics"]["x52_" + key[4:]] = value["diagnostics"].pop(key)
    for frame in value["frames"]:
        arm = frame["arms"].pop(x51.ARM_X51)
        for key in list(arm):
            if key.startswith("x51_"):
                arm["x52_" + key[4:]] = arm.pop(key)
        frame["arms"][x52.ARM_X52] = arm
    return value


def credential_scoped_cycle(
    belief: Mapping[str, Any], observation: Mapping[str, Any]
) -> bool:
    return x52.BASE_CLOSES_STATE_CYCLE(belief, observation) and (
        x51.x48.parent_id(belief) != x51.x48.parent_id(observation)
        or bool(belief.get("x55_parent_sibling_consensus_seed", False))
    )


def credential_scoped_continued_row(
    witness: Mapping[str, Any], belief: Mapping[str, Any], now_s: float
) -> dict[str, Any]:
    value = x52.BASE_CONTINUED_ROW(witness, belief, now_s)
    parent_changed = x51.x48.parent_id(belief) != x51.x48.parent_id(witness)
    if parent_changed:
        value["track_id"] = f"x52-reidentified::{witness['track_id']}"
    value["x55_parent_sibling_consensus_seed"] = bool(
        belief.get("x55_parent_sibling_consensus_seed", False)
    )
    return value


def apply_credential_scoped_provisional_episode(
    episode: Any, value: dict[str, Any], calibration: Any
) -> dict[str, Any]:
    original_cycle = x51.x45.closes_state_cycle
    original_continued_row = x51.continued_row
    x51.x45.closes_state_cycle = credential_scoped_cycle
    x51.continued_row = credential_scoped_continued_row
    try:
        return x51.apply_provisional_motion_episode(episode, value, calibration)
    finally:
        x51.x45.closes_state_cycle = original_cycle
        x51.continued_row = original_continued_row


def finalize_x55(value: dict[str, Any]) -> dict[str, Any]:
    value["arms"][ARM_X55] = value["arms"].pop(x54.ARM_X54)
    value["diagnostics"]["x55_route_mode_counts"] = value["diagnostics"].pop(
        "x54_route_mode_counts"
    )
    for frame in value["frames"]:
        frame["arms"][ARM_X55] = frame["arms"].pop(x54.ARM_X54)
    return value


def predict_episode(
    episode: Any,
    candidate_values: Sequence[Mapping[str, Any]],
    calibration: Any,
) -> dict[str, Any]:
    seeded = apply_parent_sibling_seed_episode(
        episode,
        x45.predict_episode(episode, candidate_values, calibration),
        calibration,
    )
    same_parent = apply_credential_scoped_provisional_episode(
        episode, seeded, calibration
    )
    promoted = promote_x51_to_x52_contract(same_parent)
    guarded = x53.apply_anchor_redundancy_episode(
        episode, promoted, calibration
    )
    continued = x54.apply_metric_bootstrap_dropout_episode(
        episode, guarded, calibration
    )
    return finalize_x55(continued)


def self_check() -> dict[str, Any]:
    inherited = x54.self_check()
    first = {
        "track_id": "parent::a",
        "parent_track_id": "parent",
        "disposition": "MEASURED",
        "motion_authority": x27.RIGID_DYNAMIC,
        "risk_eligible": True,
        "depth_grid_support": 10,
        "position_forward_m": 4.0,
        "position_right_m": 0.0,
        "velocity_forward_mps": -2.0,
        "velocity_right_mps": 0.0,
    }
    sibling = {
        **first,
        "track_id": "parent::b",
        "position_forward_m": 4.5,
        "velocity_forward_mps": -1.0,
    }
    isolated = {**first, "track_id": "other::a", "parent_track_id": "other"}
    x24.require(
        len(closing_sibling_rows([first, sibling])) == 2,
        "x55_sibling_cycle_closed",
    )
    x24.require(
        not closing_sibling_rows([first, isolated]),
        "x55_cross_parent_not_consensus",
    )
    return {
        "status": "X55_PARENT_SIBLING_STATE_CYCLE_FALSIFIER_MET",
        "x54_structural_status": inherited["status"],
        "same_parent_distinct_children_required": True,
        "two_route_entering_children_required": True,
        "same_parent_update_requires_x55_seed": True,
        "current_measurement_and_depth_support_required": True,
        "rigid_dynamic_authority_preserved": True,
        "class_independent": True,
        "new_numeric_threshold_added": False,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), sort_keys=True))
