"""Freeze the first trajectory-disjoint DTR-IVCA four-arm source panel."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import dtr_carla_c2_rich_scene as c2


HERE = Path(__file__).resolve().parent
PARENT_PROTOCOL = HERE / "dtr_carla_c41_x82_fresh_confirmation_protocol.json"
PARENT_PROTOCOL_SHA256 = (
    "67B806C47B9AA3B038C9CFD84E3BFF89C30D5944BBE84005CE269D2040BA08BE"
)
COHORT_ID = "DTR_CARLA_IVCA_C1_FOUR_ARM_V1"
CAPTURE_SEED = 451101
WEATHERS = {
    "c8_l01": "ClearNight",
    "c8_l02": "HardRainSunset",
    "c8_l03": "WetNoon",
    "c8_l04": "CloudySunset",
}
ARM_HASHES = {
    "X73": "8722FAB54E441459EDE6E1EBE61CE1BE0FD7E8956BB2C9B139BF67E3BF51BBD2",
    "X93": "69E7390801DD7FB1F65A06E7103C2D95B23706FB595E10AEE31BFC334622D85A",
    "X94": "8A58C1387513AD80B4E4AC474B8D4B02E4E1E5183EF13B78894D15BF46C8E5F3",
    "IVCA": "45E836511375EA31FF7267795BBB28FE68C3F8CA197E50691F9DD09C94F8F6C3",
}
EVALUATOR_SHA256 = (
    "C4B5DFD459860FED0D78657D596E020A1943BB72BC14EE626046D31375D567CD"
)


def trajectory(
    start_forward_m: float,
    start_right_m: float,
    segments: list[tuple[float, float, float]],
) -> dict:
    return {
        "start_forward_m": start_forward_m,
        "start_right_m": start_right_m,
        "segments": [
            {
                "start_s": start_s,
                "velocity_forward_mps": forward_mps,
                "velocity_right_mps": right_mps,
            }
            for start_s, forward_mps, right_mps in segments
        ],
    }


def issued_plan(episode_number: int) -> dict:
    session_id = f"ivca_c1_session_{episode_number:02d}"
    return {
        "issued_at_s": 0.0,
        "expires_at_s": 9.0,
        "plan_id": f"ivca_c1_plan_{episode_number:02d}",
        "session_id": session_id,
        "time_parameterized_waypoints": [
            {"time_s": 0.0, "forward_m": -7.0, "right_m": 0.0},
            {"time_s": 1.0, "forward_m": -5.0, "right_m": 0.0},
            {"time_s": 5.0, "forward_m": 3.0, "right_m": 0.0},
            {"time_s": 9.0, "forward_m": 11.0, "right_m": 0.0},
        ],
    }


def scenario(
    episode_number: int,
    layout_id: str,
    role: str,
    target_trajectory: str,
    expected_outcome: str,
) -> dict:
    episode_id = f"ep_{episode_number:02d}"
    target_key = f"{layout_id}_target"
    alias_key = f"{layout_id}_alias"
    occluder_key = f"{layout_id}_occluder"
    return {
        "episode_id": episode_id,
        "layout_id": layout_id,
        "scenario_role": role,
        "twin_role": "ivca_information_role",
        "expected_outcome": expected_outcome,
        "expected_responsible_assets": [target_key] if expected_outcome == "CONTACT" else [],
        "wearer_trajectory": "ivca_c1_wearer_route",
        "asset_trajectories": {
            target_key: target_trajectory,
            alias_key: f"ivca_c1_{layout_id}_alias_far",
            occluder_key: f"ivca_c1_{layout_id}_occluder_cross",
        },
        "navigation_session_id": f"ivca_c1_session_{episode_number:02d}",
        "issued_plan": issued_plan(episode_number),
    }


def materialize(parent: dict) -> dict:
    protocol = copy.deepcopy(parent)
    protocol["schema_version"] = 45
    protocol["cohort_id"] = COHORT_ID
    protocol["capture"]["seed"] = CAPTURE_SEED
    protocol["objective"] = (
        "Evaluate byte-frozen X73, X93, X94, and IVCA on one new trajectory, "
        "seed, render, plan-receipt, and pixel panel that separates interval "
        "birth from one-observation transport persistence."
    )
    for layout_id, weather in WEATHERS.items():
        protocol["layouts"][layout_id]["weather"] = weather

    library = protocol["trajectory_library"]
    library["ivca_c1_wearer_route"] = trajectory(
        -7.0, 0.0, [(0.0, 2.0, 0.0)]
    )
    roles = [
        ("c8_l01", "sustained_true_conflict_and_positive_dropout", trajectory(7.0, 0.0, [(0.0, -1.5, 0.0)]), "CONTACT"),
        ("c8_l01", "near_miss_without_contact", trajectory(7.0, 2.2, [(0.0, -1.5, 0.0)]), "SAFE"),
        ("c8_l02", "future_only_lateral_true_conflict", trajectory(1.0, -8.0, [(0.0, 0.0, 2.0)]), "CONTACT"),
        ("c8_l02", "crossing_exits_before_arrival_and_negative_dropout", trajectory(3.0, -4.0, [(0.0, 0.0, 4.0), (2.0, 0.0, 0.0)]), "SAFE"),
        ("c8_l03", "current_overlap_with_closing", trajectory(-6.4, 0.0, [(0.0, -0.5, 0.0), (1.0, 0.0, 0.0)]), "CONTACT"),
        ("c8_l03", "receding_non_conflict", trajectory(-1.5, 0.0, [(0.0, 3.6, 0.0)]), "SAFE"),
        ("c8_l04", "lateral_only_non_conflict", trajectory(5.0, -8.0, [(0.0, 0.0, 0.8)]), "SAFE"),
        ("c8_l04", "stationary_future_conflict_and_positive_dropout", trajectory(1.0, 0.0, [(0.0, 0.0, 0.0)]), "CONTACT"),
    ]
    scenarios = []
    for episode_number, (layout_id, role, target_value, outcome) in enumerate(roles, start=1):
        target_name = f"ivca_c1_ep_{episode_number:02d}_target"
        library[target_name] = target_value
        library[f"ivca_c1_{layout_id}_alias_far"] = trajectory(
            8.0, -5.5, [(0.0, -0.4, 0.0)]
        )
        library[f"ivca_c1_{layout_id}_occluder_cross"] = trajectory(
            3.0, 8.0, [(0.0, 0.0, 0.0), (1.0, 0.0, -9.0), (2.2, 0.0, 0.0)]
        )
        scenarios.append(
            scenario(episode_number, layout_id, role, target_name, outcome)
        )
    protocol["scenarios"] = scenarios
    protocol["twin_contracts"] = []

    contact_episodes = ["ep_01", "ep_03", "ep_05", "ep_08"]
    safe_episodes = ["ep_02", "ep_04", "ep_06", "ep_07"]
    protocol["evaluation_contract"] = {
        "future_truth_rule": "SOURCE_NATIVE_REALIZED_CONTACT_WITH_FULL_HORIZON",
        "truth_tail_seconds": 3.0,
        "contact_episodes": contact_episodes,
        "safe_episodes": safe_episodes,
        "score_window_end_seconds": {f"ep_{value:02d}": 6.0 for value in range(1, 9)},
        "ivca_role_by_episode": {
            f"ep_{index:02d}": role for index, (_layout, role, _target, _outcome) in enumerate(roles, start=1)
        },
        "controlled_full_dropout": {
            "ep_01": {"sample_index": 20, "length_frames": 1, "required_truth": "FUTURE_CONTACT_POSITIVE"},
            "ep_04": {"sample_index": 25, "length_frames": 1, "required_truth": "FUTURE_CONTACT_NEGATIVE"},
            "ep_08": {"sample_index": 20, "length_frames": 1, "required_truth": "FUTURE_CONTACT_POSITIVE"},
        },
        "primary_comparisons": ["X93_MINUS_X73", "X94_MINUS_X93", "IVCA_MINUS_X94"],
        "primary_metrics": [
            "event_precision_recall_f1",
            "false_onset_births",
            "event_fragmentation",
            "first_alert_lead_seconds",
            "midpoint_interval_iou_with_temporal_resolution",
            "entry_exit_censored_error_seconds",
            "minimum_clearance_error_m",
            "x94_continuity_rescue_tp",
            "false_persistence_duration_seconds",
        ],
        "ivca_gate": {
            "critical_event_loss_maximum": 0,
            "false_onset_births_must_strictly_decrease": True,
            "event_fragmentation_must_not_increase": True,
            "lead_regression_margin_seconds": 0.15,
            "x94_positive_dropout_rescue_must_be_retained": True,
            "hold_window_or_lifecycle_change_allowed": False,
        },
    }
    # The generic C2 capture schema requires one physical visibility contract.
    # It is source admission only and is not the controlled dropout intervention.
    first = copy.deepcopy(parent["occlusion_contracts"][0])
    first.update(
        {
            "contract_id": "ivca_c1_physical_visibility_ep_01",
            "episodes": ["ep_01"],
            "required_outcomes": {"ep_01": "CONTACT"},
            "target_asset": "c8_l01_target",
            "alias_asset": "c8_l01_alias",
            "occluder_asset": "c8_l01_occluder",
            "planned_occlusion_window_s": [1.2, 2.3],
        }
    )
    protocol["occlusion_contracts"] = [first]
    protocol["admission"]["c16_delayed_release_shell_source_gates"]["required_episode_count"] = 8
    protocol["ivca_c1_preregistration"] = {
        "schema": "dtr-carla-ivca-c1-preregistration-v1",
        "created_before_pixels_predictions_or_truth": True,
        "parent_protocol_sha256": PARENT_PROTOCOL_SHA256,
        "arm_sha256": ARM_HASHES,
        "evaluator_sha256": EVALUATOR_SHA256,
        "trajectory_disjoint_from_c41": True,
        "capture_seed": CAPTURE_SEED,
        "weather_by_layout": WEATHERS,
        "dropout_is_controlled_candidate_and_metric_removal_not_natural_prevalence": True,
        "role_failure_rule": "SOURCE_NATIVE_TRUTH_MISMATCH_IS_NOT_EVALUABLE_NO_REPLACEMENT",
        "mechanism_not_exercised_rule": "REPORT_WITHOUT_CONFIRMATION_OR_NEGATIVE_INCREMENTAL_CLAIM",
        "single_scored_invocation": True,
    }
    protocol["source_disjoint_contract"] = {
        "schema": "dtr-carla-ivca-c1-source-contract-v1",
        "parent_cohort_id": parent["cohort_id"],
        "parent_protocol_sha256": PARENT_PROTOCOL_SHA256,
        "new_selected_target_alias_occluder_and_wearer_trajectories": True,
        "new_plan_receipts": True,
        "new_seed_weather_and_pixels": True,
        "future_truth_visible_to_predictor": False,
        "completed_shard_rerun_allowed": False,
        "nonzero_partial_shard_is_source_not_evaluable": True,
    }
    protocol["claim_boundary"] = [
        "IVCA-C1 is scripted synthetic Development evidence only.",
        "Role-balanced construction is not natural-distribution prevalence evidence.",
        "Controlled dropout is not natural detector-dropout prevalence evidence.",
        "A source-native role mismatch is NOT_EVALUABLE and cannot be repaired after truth opens.",
        "No real-sensor, deployment, reliability, user-benefit, or safety claim follows.",
    ]
    c2.validate_protocol(protocol)
    return protocol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if c2.sha256_file(PARENT_PROTOCOL) != PARENT_PROTOCOL_SHA256:
        raise RuntimeError("C41 parent protocol drift")
    protocol = materialize(json.loads(PARENT_PROTOCOL.read_text(encoding="utf-8")))
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(
        (json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )
    )
    print(
        json.dumps(
            {
                "status": "IVCA_C1_PROTOCOL_STATIC_VALID_PREREGISTERED",
                "cohort_id": COHORT_ID,
                "episode_count": len(protocol["scenarios"]),
                "role_count": len(protocol["evaluation_contract"]["ivca_role_by_episode"]),
                "output": str(output),
                "sha256": c2.sha256_file(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
