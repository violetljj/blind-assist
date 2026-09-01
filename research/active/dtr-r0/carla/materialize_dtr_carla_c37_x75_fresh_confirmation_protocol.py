"""Freeze a new C37 pixel source for unchanged X75 confirmation."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import dtr_carla_c2_rich_scene as c2
import materialize_dtr_carla_c36_x74_fresh_confirmation_protocol as c36


HERE = Path(__file__).resolve().parent
PARENT_PROTOCOL = HERE / "dtr_carla_c36_x74_fresh_confirmation_protocol.json"
PARENT_PROTOCOL_SHA256 = (
    "4EF6BE6B4E1625556A4F78E4DFBC8F981E5C89DC1EB73EDA0C991F7DFECD83EA"
)
PARENT_SOURCE_RESULT_SHA256 = (
    "3877F53C33D65FB86C430DD7E927AD991045B9FDD7F4558FDE5CDD2342CB236C"
)
PARENT_SOURCE_STATUS = "DTR_CARLA_C2_RICH_MULTILAYOUT_SOURCE_COMPLETE"
COHORT_ID = "DTR_CARLA_C37_X75_FRESH_CONFIRMATION_V1"
CAPTURE_SEED = 371075
WEATHERS = {
    "c8_l01": "MidRainSunset",
    "c8_l02": "WetNight",
    "c8_l03": "ClearNight",
    "c8_l04": "SoftRainSunset",
}
FROZEN_COMPONENTS = {
    **c36.FROZEN_COMPONENTS,
    "dtr_carla_x75_collision_credentialed_object_permanence.py": (
        "2A21A794EC52ED30D15D45BE88FC5E0846735FA06B65839BEBC365ED5E992808"
    ),
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def materialize(parent: dict) -> dict:
    protocol = copy.deepcopy(parent)
    protocol["schema_version"] = 37
    protocol["cohort_id"] = COHORT_ID
    protocol["capture"]["seed"] = CAPTURE_SEED
    protocol["objective"] = (
        "Confirm byte-frozen X75 once on genuinely new seed- and render-domain-"
        "disjoint CARLA pixels after its seven-cohort consumed precision gain, "
        "while preserving camera, route, geometry, and trajectories."
    )
    for layout_id, weather in WEATHERS.items():
        protocol["layouts"][layout_id]["weather"] = weather

    protocol.pop("c36_x74_preregistration")
    protocol["c37_x75_preregistration"] = {
        "schema": "dtr-carla-c37-x75-preregistration-v1",
        "baselines": ["X24", "X74", "X75"],
        "single_x75_scored_invocation": True,
        "no_post_capture_algorithm_threshold_or_scenario_tuning": True,
        "score_window_seconds": [0.0, 6.0],
        "frozen_component_sha256": FROZEN_COMPONENTS,
        "frozen_capture_script_sha256": c36.CAPTURE_SCRIPT_SHA256,
        "frozen_join_script_sha256": c36.JOIN_SCRIPT_SHA256,
        "primary_transfer_gate": {
            "minimum_precision": 0.85,
            "minimum_recall": 0.70,
            "minimum_f1": 0.78,
            "minimum_each_contact_recall": 0.55,
            "maximum_safe_false_alert_segments_per_episode": 4,
            "maximum_total_safe_false_alert_segments": 10,
            "minimum_x75_permanence_release_frames": 1,
            "minimum_x75_tp_delta_vs_x74": 0,
            "maximum_x75_fp_delta_vs_x74": -1,
            "required_zero_authority_invariants": [
                "confirmed_missing_track_references",
                "confirmed_non_rigid_risk_track_references",
                "confirmed_parent_identity_mismatches",
                "route_risk_without_confirmed_eligible_track_frames",
                "route_risk_without_confirmed_rigid_dynamic_frames",
            ],
        },
        "stretch_target": {"precision": 0.92, "recall": 0.77, "f1": 0.84},
        "outcomes": [
            "GATE_MET",
            "GATE_NOT_MET",
            "SOURCE_NOT_EVALUABLE",
            "MECHANISM_NOT_EXERCISED",
        ],
    }
    protocol["source_disjoint_contract"] = {
        "schema": "dtr-carla-c37-x75-fresh-source-contract-v1",
        "parent_cohort_id": parent["cohort_id"],
        "parent_protocol_sha256": PARENT_PROTOCOL_SHA256,
        "parent_source_result_sha256": PARENT_SOURCE_RESULT_SHA256,
        "parent_source_status": PARENT_SOURCE_STATUS,
        "c37_is_new_cohort_not_c36_retry": True,
        "source_change": "NEW_CAPTURE_SEED_AND_RENDER_DOMAIN_ASSIGNMENT_ONLY",
        "new_capture_seed": CAPTURE_SEED,
        "changed_weather_by_layout": WEATHERS,
        "camera_geometry_unchanged_from_c36": True,
        "target_and_occluder_trajectories_unchanged_from_c36": True,
        "route_and_truth_contract_unchanged_from_c36": True,
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
        "C37 uses a new seed, render-domain assignment, and pixels; it does not reuse C36 payloads.",
        "X75 and every imported algorithm component are byte-frozen before C37 capture.",
        "The sole C37 score confirms X75 only if permanence release is exercised, loses no TP versus X74, and removes at least one FP.",
        "A mechanism-not-exercised result is not positive or negative incremental confirmation.",
        "C37 is source-disjoint synthetic Development, not unseen-map, open-world traffic, real-sensor, deployment, reliability, user-benefit, or safety evidence.",
        "A failed source gate is NOT_EVALUABLE; frozen X75 cannot be changed or rerun on C37 after pixels are opened.",
    ]
    c2.validate_protocol(protocol)
    return protocol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c36-source-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if c2.sha256_file(PARENT_PROTOCOL) != PARENT_PROTOCOL_SHA256:
        raise RuntimeError("frozen C36 parent protocol drift")
    source_result_path = args.c36_source_result.resolve(strict=True)
    if c2.sha256_file(source_result_path) != PARENT_SOURCE_RESULT_SHA256:
        raise RuntimeError("frozen C36 source result drift")
    source_result = read_json(source_result_path)
    if source_result.get("status") != PARENT_SOURCE_STATUS:
        raise RuntimeError("C36 source result status drift")
    for file_name, expected in FROZEN_COMPONENTS.items():
        path = HERE / file_name
        if not path.is_file() or c2.sha256_file(path) != expected:
            raise RuntimeError(f"frozen C37 component drift: {file_name}")
    for path, expected in {
        HERE / "capture_dtr_carla_c2_rich_scene.py": c36.CAPTURE_SCRIPT_SHA256,
        HERE / "join_dtr_carla_c2_rich_scene.py": c36.JOIN_SCRIPT_SHA256,
    }.items():
        if c2.sha256_file(path) != expected:
            raise RuntimeError(f"frozen C37 source script drift: {path.name}")

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
        "status": "C37_X75_PROTOCOL_STATIC_VALID_PREREGISTERED",
        "output": str(output),
        "sha256": c2.sha256_file(output),
        "cohort_id": COHORT_ID,
        "capture_seed": CAPTURE_SEED,
        "weather_domains": WEATHERS,
        "x75_sha256": FROZEN_COMPONENTS["dtr_carla_x75_collision_credentialed_object_permanence.py"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
