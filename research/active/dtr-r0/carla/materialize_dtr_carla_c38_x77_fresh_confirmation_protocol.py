"""Freeze a new C38 pixel source for unchanged X77 confirmation."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import dtr_carla_c2_rich_scene as c2
import materialize_dtr_carla_c37_x75_fresh_confirmation_protocol as c37


HERE = Path(__file__).resolve().parent
PARENT_PROTOCOL = HERE / "dtr_carla_c37_x75_fresh_confirmation_protocol.json"
PARENT_PROTOCOL_SHA256 = (
    "81AE71541B2DB5C6CD7DD29A97A360D0B4D9F02E9E19E5F759D428D8EDF9BF80"
)
PARENT_SOURCE_RESULT_SHA256 = (
    "581509BCDF6AEC07CFA932449414558F020AEB25F2349A621BBF2688810BD134"
)
PARENT_SOURCE_STATUS = "DTR_CARLA_C2_RICH_MULTILAYOUT_SOURCE_COMPLETE"
COHORT_ID = "DTR_CARLA_C38_X77_FRESH_CONFIRMATION_V1"
CAPTURE_SEED = 381077
WEATHERS = {
    "c8_l01": "CloudySunset",
    "c8_l02": "HardRainNight",
    "c8_l03": "WetCloudySunset",
    "c8_l04": "DustStorm",
}
FROZEN_COMPONENTS = {
    **c37.FROZEN_COMPONENTS,
    "dtr_carla_x76_zero_shift_parent_hull_motion_rejection.py": (
        "517E50196CC85EEA55A494EA505DAB6D48F1F26235DA4B67553A8836905E7865"
    ),
    "dtr_carla_x77_receding_metric_temporal_handoff_rejection.py": (
        "1F7BF820AB3048C394923E3CA7A23F10BBDB9C8813AE8C78A86D201318AEC167"
    ),
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def materialize(parent: dict) -> dict:
    protocol = copy.deepcopy(parent)
    protocol["schema_version"] = 38
    protocol["cohort_id"] = COHORT_ID
    protocol["capture"]["seed"] = CAPTURE_SEED
    protocol["objective"] = (
        "Confirm byte-frozen X77 once on genuinely new seed- and render-domain-"
        "disjoint CARLA pixels after its eight-cohort zero-TP-loss precision "
        "gain, while preserving camera, route, geometry, and trajectories."
    )
    for layout_id, weather in WEATHERS.items():
        protocol["layouts"][layout_id]["weather"] = weather

    protocol.pop("c37_x75_preregistration")
    protocol["c38_x77_preregistration"] = {
        "schema": "dtr-carla-c38-x77-preregistration-v1",
        "baselines": ["X24", "X76", "X77"],
        "single_x77_scored_invocation": True,
        "no_post_capture_algorithm_threshold_or_scenario_tuning": True,
        "score_window_seconds": [0.0, 6.0],
        "frozen_component_sha256": FROZEN_COMPONENTS,
        "frozen_capture_script_sha256": c37.c36.CAPTURE_SCRIPT_SHA256,
        "frozen_join_script_sha256": c37.c36.JOIN_SCRIPT_SHA256,
        "primary_transfer_gate": {
            "minimum_precision": 0.85,
            "minimum_recall": 0.70,
            "minimum_f1": 0.78,
            "minimum_each_contact_recall": 0.55,
            "maximum_safe_false_alert_segments_per_episode": 4,
            "maximum_total_safe_false_alert_segments": 10,
            "minimum_x77_receding_handoff_release_frames": 1,
            "minimum_x77_tp_delta_vs_x76": 0,
            "maximum_x77_fp_delta_vs_x76": -1,
            "required_zero_authority_invariants": [
                "confirmed_missing_track_references",
                "confirmed_non_rigid_risk_track_references",
                "confirmed_parent_identity_mismatches",
                "route_risk_without_confirmed_eligible_track_frames",
                "route_risk_without_confirmed_rigid_dynamic_frames",
            ],
        },
        "stretch_target": {"precision": 0.92, "recall": 0.78, "f1": 0.85},
        "outcomes": [
            "GATE_MET",
            "GATE_NOT_MET",
            "SOURCE_NOT_EVALUABLE",
            "MECHANISM_NOT_EXERCISED",
        ],
    }
    protocol["source_disjoint_contract"] = {
        "schema": "dtr-carla-c38-x77-fresh-source-contract-v1",
        "parent_cohort_id": parent["cohort_id"],
        "parent_protocol_sha256": PARENT_PROTOCOL_SHA256,
        "parent_source_result_sha256": PARENT_SOURCE_RESULT_SHA256,
        "parent_source_status": PARENT_SOURCE_STATUS,
        "c38_is_new_cohort_not_c37_retry": True,
        "source_change": "NEW_CAPTURE_SEED_AND_RENDER_DOMAIN_ASSIGNMENT_ONLY",
        "new_capture_seed": CAPTURE_SEED,
        "changed_weather_by_layout": WEATHERS,
        "camera_geometry_unchanged_from_c37": True,
        "target_and_occluder_trajectories_unchanged_from_c37": True,
        "route_and_truth_contract_unchanged_from_c37": True,
        "model_sensor_replay_projection": "ACTUAL_ALL_ACTORS",
        "critical_contact_outcome_truth_must_match_across_all_sensors": True,
        "fresh_pixels": True,
        "prior_pixels_reused": False,
        "future_truth_visible_to_predictor": False,
        "single_scored_invocation": True,
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
        "C38 uses a new seed, render-domain assignment, and pixels; it does not reuse C37 payloads.",
        "X77 and every imported algorithm component are byte-frozen before C38 capture.",
        "The sole C38 score confirms X77 only if receding-handoff release is exercised, loses no TP versus X76, and removes at least one FP.",
        "A mechanism-not-exercised result is not positive or negative incremental confirmation.",
        "C38 is source-disjoint synthetic Development, not unseen-map, open-world traffic, real-sensor, deployment, reliability, user-benefit, or safety evidence.",
        "A failed source gate is NOT_EVALUABLE; frozen X77 cannot be changed or rerun on C38 after pixels are opened.",
    ]
    c2.validate_protocol(protocol)
    return protocol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c37-source-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if c2.sha256_file(PARENT_PROTOCOL) != PARENT_PROTOCOL_SHA256:
        raise RuntimeError("frozen C37 parent protocol drift")
    source_result_path = args.c37_source_result.resolve(strict=True)
    if c2.sha256_file(source_result_path) != PARENT_SOURCE_RESULT_SHA256:
        raise RuntimeError("frozen C37 source result drift")
    source_result = read_json(source_result_path)
    if source_result.get("status") != PARENT_SOURCE_STATUS:
        raise RuntimeError("C37 source result status drift")
    for file_name, expected in FROZEN_COMPONENTS.items():
        path = HERE / file_name
        if not path.is_file() or c2.sha256_file(path) != expected:
            raise RuntimeError(f"frozen C38 component drift: {file_name}")
    for path, expected in {
        HERE / "capture_dtr_carla_c2_rich_scene.py": c37.c36.CAPTURE_SCRIPT_SHA256,
        HERE / "join_dtr_carla_c2_rich_scene.py": c37.c36.JOIN_SCRIPT_SHA256,
    }.items():
        if c2.sha256_file(path) != expected:
            raise RuntimeError(f"frozen C38 source script drift: {path.name}")

    protocol = materialize(read_json(PARENT_PROTOCOL))
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "C38_X77_PROTOCOL_STATIC_VALID_PREREGISTERED",
        "output": str(output),
        "sha256": c2.sha256_file(output),
        "cohort_id": COHORT_ID,
        "capture_seed": CAPTURE_SEED,
        "weather_domains": WEATHERS,
        "x77_sha256": FROZEN_COMPONENTS[
            "dtr_carla_x77_receding_metric_temporal_handoff_rejection.py"
        ],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
