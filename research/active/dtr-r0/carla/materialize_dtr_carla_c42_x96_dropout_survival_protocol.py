"""Freeze fresh C42 source and the X96 2/3/6-frame dropout stress."""

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
PARENT_SOURCE_RESULT_SHA256 = (
    "342F1DAB5D6E6D03B81E638A271B5907CD2985BD8F84D6B9DC277888BB9B4311"
)
PARENT_SOURCE_STATUS = "DTR_CARLA_C2_RICH_MULTILAYOUT_SOURCE_COMPLETE"
COHORT_ID = "DTR_CARLA_C42_X96_DROPOUT_SURVIVAL_STRESS_V1"
CAPTURE_SEED = 421096
WEATHERS = {
    "c8_l01": "CloudyNoon",
    "c8_l02": "SoftRainSunset",
    "c8_l03": "WetCloudyNoon",
    "c8_l04": "ClearSunset",
}
NEW_COMPONENTS = {
    "dtr_carla_x83_rigid_risk_reference_projection.py": "BEC2E8251CF019AD1AED3C7CC2C4BBE4865B0621D42049DE17718C61DEB2289D",
    "dtr_carla_x84_branch_overloaded_closing_continuation_release.py": "56D4FE10987B932130D4EB9E7C0CCD58A9F53D2F43D02581E29404E2E2F539BD",
    "dtr_carla_x85_dequantization_completion_precedence_release.py": "05DD9B81E940295A16A7D5445C7E5B17F9DC2DEC7D67CE3C3E4FC8CE3431B323",
    "dtr_carla_x86_receding_handback_horizon_release.py": "12E4BC2F6EA196703A2EF6F6CEB61A3F4CE31111843BF52525AE1F2B7B3DF1FB",
    "dtr_carla_x87_solo_completion_horizon_release.py": "8B741E1B58EDC2BE7319A206FFCDBD217218CA7BC8613FA289BDF360D85408F9",
    "dtr_carla_x88_motion_epoch_contradiction_release.py": "C6119C61291FA5667C15B10396D41701BA88D3708AABFE9CCC234C0E7F54887D",
    "dtr_carla_x89_branch_overloaded_receding_release.py": "957E279AAA5F446E8219E2F8E64EEE337DA680521479B1DA9D03FF04C726C30F",
    "dtr_carla_x90_collision_credentialed_lateral_dominant_release.py": "35F34F872B87C339402CDF35331C44691F43D511BA0E2EC0F1D0EA41D30470BC",
    "dtr_carla_x91_held_risk_birth_horizon_release.py": "9AB89C111762E0C8DA45E22E100D456D08354E0BFAC721C01A470CD891ECB430",
    "dtr_carla_x92_held_risk_birth_horizon_latch.py": "D81811C32DF29BA485FFC34D682439A1A6FD9BA012FCB3187F9855ADEFF51B92",
    "dtr_carla_x93_conflicted_nonclosing_future_release.py": "69E7390801DD7FB1F65A06E7103C2D95B23706FB595E10AEE31BFC334622D85A",
    "dtr_carla_x94_one_frame_full_dropout_continuity.py": "8A58C1387513AD80B4E4AC474B8D4B02E4E1E5183EF13B78894D15BF46C8E5F3",
    "dtr_carla_x96_credentialed_bounded_dropout_survival.py": "C155911FDADA84EDAE31F417F0B68C8D2747304A86C5D85F04360D471E0FB0D2",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _c17_trajectory(parent: dict, old_name: str) -> str:
    prefix = f"c17_{old_name}_"
    matches = sorted(
        name for name in parent["trajectory_library"] if name.startswith(prefix)
    )
    if len(matches) != 1:
        raise RuntimeError(f"expected one unused c17 trajectory for {old_name}: {matches}")
    return matches[0]


def materialize(parent: dict) -> dict:
    protocol = copy.deepcopy(parent)
    protocol["schema_version"] = 42
    protocol["cohort_id"] = COHORT_ID
    protocol["capture"]["seed"] = CAPTURE_SEED
    protocol["objective"] = (
        "Test byte-frozen X96 against X94, recursive forward-fill, and 0.60 s "
        "hysteresis on preregistered 2/3/6-frame full detector-plus-metric "
        "dropouts over new actor trajectories, seed, render assignment, and pixels."
    )
    for layout_id, weather in WEATHERS.items():
        protocol["layouts"][layout_id]["weather"] = weather

    for scenario in protocol["scenarios"]:
        scenario["wearer_trajectory"] = "c17_wearer_route"
        scenario["asset_trajectories"] = {
            asset_key: _c17_trajectory(parent, old_name)
            for asset_key, old_name in scenario["asset_trajectories"].items()
        }
        episode_number = int(str(scenario["episode_id"]).split("_")[-1])
        session_id = f"c42_session_dropout_{episode_number:02d}"
        scenario["navigation_session_id"] = session_id
        scenario["issued_plan"] = {
            "plan_id": f"c42_plan_dropout_{episode_number:02d}",
            "session_id": session_id,
            "issued_at_s": 0.0,
            "expires_at_s": 9.0,
            "time_parameterized_waypoints": [
                {"time_s": 0.0, "forward_m": -6.48, "right_m": 0.0},
                {"time_s": 1.0, "forward_m": -4.428, "right_m": 0.0},
                {"time_s": 5.0, "forward_m": 4.428, "right_m": 0.0},
                {"time_s": 9.0, "forward_m": 13.068, "right_m": 0.0},
            ],
        }

    parent_prereg = protocol.pop("c41_x82_preregistration")
    frozen_components = {
        **parent_prereg["frozen_component_sha256"],
        **NEW_COMPONENTS,
    }
    protocol["c42_x96_preregistration"] = {
        "schema": "dtr-carla-c42-x96-dropout-survival-preregistration-v1",
        "arms": [
            "X94_ONE_FRAME_FULL_DROPOUT_CONTINUITY",
            "RECURSIVE_FORWARD_FILL",
            "HYSTERESIS_0_60_S",
            "X96_CREDENTIALED_BOUNDED_DROPOUT_SURVIVAL",
        ],
        "dropout_lengths_frames": [2, 3, 6],
        "sensor_tick_seconds": 0.1,
        "dropout_operation": (
            "SET_FROZEN_YOLO_CANDIDATE_LIST_EMPTY_BEFORE_X24_X25_AND_SURFACE_"
            "PIPELINE_SO_DETECTOR_AND_METRIC_FOOTPRINTS_ARE_BOTH_CAUSALLY_ABSENT"
        ),
        "contact_episodes": ["ep_01", "ep_03", "ep_05", "ep_07"],
        "placements": {
            "A_ACTIVE_MIDDLE": {
                "start_sample_index": 22,
                "expected_authority": "MAINTAIN_IF_PRIOR_MEASURED_CREDENTIAL_EXISTS",
            },
            "B_PRE_ONSET": {
                "start_sample_index": 2,
                "expected_authority": "MUST_NOT_BIRTH_FROM_ABSENCE",
            },
            "C_RELEASE_BOUNDARY": {
                "start_sample_index": 44,
                "expected_authority": "MUST_NOT_PERSIST_AFTER_RELEASE_EVIDENCE",
            },
            "D_PLAN_CONFLICT": {
                "start_sample_index": 22,
                "controlled_intervention": "ROUTE_MODE_CHANGED_TRUE_DURING_DROPOUT",
                "expected_authority": "MUST_TERMINATE_IMMEDIATELY",
            },
        },
        "placement_indices_are_frozen_before_capture": True,
        "placement_may_not_move_after_truth_or_predictions_are_opened": True,
        "single_joint_score_after_all_arm_predictions_are_sealed": True,
        "no_post_capture_algorithm_threshold_or_scenario_tuning": True,
        "score_window_seconds": [0.0, 6.0],
        "frozen_component_sha256": frozen_components,
        "primary_gate": {
            "event_recall_not_below_x94": True,
            "frame_f1_not_below_x94": True,
            "minimum_2_3_frame_positive_dropout_recovery": 0.80,
            "maximum_negative_dropout_persistence": 0.10,
            "maximum_false_births": 0,
            "maximum_cross_plan_or_conflict_carries": 0,
            "minimum_6_frame_tp_minus_fp": 1,
            "maximum_median_release_overshoot_seconds": 0.20,
        },
        "partition_evaluability": (
            "A_REQUIRES_POSITIVE_TRUTH_IN_DROPOUT_B_REQUIRES_NO_PRE_DROPOUT_"
            "POSITIVE_TRUTH_C_REQUIRES_POSITIVE_TO_NEGATIVE_TRUTH_TRANSITION_"
            "WITHIN_ONE_SAMPLE_OF_START_D_IS_CONTROLLED_AUTHORITY_NEGATIVE_"
            "INDEPENDENT_OF_CONTACT_TRUTH"
        ),
        "outcomes": [
            "GATE_MET",
            "GATE_NOT_MET",
            "SOURCE_NOT_EVALUABLE",
            "PARTITION_NOT_EVALUABLE",
            "MECHANISM_NOT_EXERCISED",
        ],
    }
    protocol["source_disjoint_contract"] = {
        "schema": "dtr-carla-c42-x96-fresh-source-contract-v1",
        "parent_cohort_id": parent["cohort_id"],
        "parent_protocol_sha256": PARENT_PROTOCOL_SHA256,
        "parent_source_result_sha256": PARENT_SOURCE_RESULT_SHA256,
        "parent_source_status": PARENT_SOURCE_STATUS,
        "c42_is_new_cohort_not_c41_retry": True,
        "source_change": (
            "NEW_PREEXISTING_C17_ACTOR_TRAJECTORY_BINDINGS_CAPTURE_SEED_"
            "RENDER_DOMAIN_ASSIGNMENT_AND_PIXELS"
        ),
        "new_capture_seed": CAPTURE_SEED,
        "changed_weather_by_layout": WEATHERS,
        "wearer_trajectory": "c17_wearer_route",
        "actor_trajectory_namespace": "c17_",
        "c17_trajectories_existed_unused_in_frozen_c41_parent": True,
        "camera_geometry_unchanged_from_c41": True,
        "route_geometry_unchanged_from_c41": True,
        "model_sensor_replay_projection": "ACTUAL_ALL_ACTORS",
        "critical_contact_outcome_truth_must_match_across_all_sensors": True,
        "fresh_pixels": True,
        "prior_pixels_reused": False,
        "future_truth_visible_to_predictor": False,
        "capture_retry_policy": (
            "ONE_FAILED_SERVER_SHARD_WITH_ZERO_DURABLE_FRAMES_MAY_BE_RETRIED_"
            "ONCE_AFTER_ATTEMPT_LOGS_ARE_PRESERVED"
        ),
        "capture_in_doubt_policy": (
            "ANY_NONZERO_PARTIAL_SHARD_IS_SOURCE_NOT_EVALUABLE_AND_MUST_NOT_RETRY"
        ),
        "completed_shard_rerun_allowed": False,
    }
    protocol["claim_boundary"] = [
        "C42 uses new c17 actor trajectory bindings, seed, render assignment, and pixels; no C41 payload is reused.",
        "Dropout placements and lengths are fixed before capture and may not move after predictions or truth are opened.",
        "Candidate removal is a controlled sensor-intervention stress layered on fresh CARLA pixels, not naturally occurring detector dropout prevalence evidence.",
        "D plan conflict is a controlled authority intervention, not a captured route replanning event.",
        "A partition that does not realize its preregistered truth semantics is NOT_EVALUABLE and cannot be repaired on C42.",
        "C42 remains source-disjoint scripted synthetic Development, not open-world, real-sensor, deployment, reliability, user-benefit, or safety evidence.",
    ]
    c2.validate_protocol(protocol)
    return protocol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c41-source-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if c2.sha256_file(PARENT_PROTOCOL) != PARENT_PROTOCOL_SHA256:
        raise RuntimeError("frozen C41 parent protocol drift")
    source_result_path = args.c41_source_result.resolve(strict=True)
    if c2.sha256_file(source_result_path) != PARENT_SOURCE_RESULT_SHA256:
        raise RuntimeError("frozen C41 source result drift")
    source_result = read_json(source_result_path)
    if source_result.get("status") != PARENT_SOURCE_STATUS:
        raise RuntimeError("C41 source result status drift")
    for file_name, expected in NEW_COMPONENTS.items():
        path = HERE / file_name
        if not path.is_file() or c2.sha256_file(path) != expected:
            raise RuntimeError(f"frozen C42 component drift: {file_name}")

    protocol = materialize(read_json(PARENT_PROTOCOL))
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "C42_X96_DROPOUT_SURVIVAL_PROTOCOL_STATIC_VALID_PREREGISTERED",
                "output": str(output),
                "sha256": c2.sha256_file(output),
                "cohort_id": COHORT_ID,
                "capture_seed": CAPTURE_SEED,
                "weather_domains": WEATHERS,
                "dropout_lengths_frames": [2, 3, 6],
                "placements": [
                    "A_ACTIVE_MIDDLE",
                    "B_PRE_ONSET",
                    "C_RELEASE_BOUNDARY",
                    "D_PLAN_CONFLICT",
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
