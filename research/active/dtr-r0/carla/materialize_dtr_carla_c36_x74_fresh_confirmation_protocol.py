"""Freeze a new C36 pixel source for unchanged X74 confirmation."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import dtr_carla_c2_rich_scene as c2


HERE = Path(__file__).resolve().parent
PARENT_PROTOCOL = HERE / "dtr_carla_c35_x73_fresh_confirmation_protocol.json"
PARENT_PROTOCOL_SHA256 = (
    "53E52FC4318E0ECD4F60870E3999B878DF18CF6C84C7890D8434C043B6718A7E"
)
PARENT_SOURCE_RESULT_SHA256 = (
    "8F302B7D58FA800FE1883D3E8AE3DAC9E3DE9305A92A197636530D19192B660E"
)
PARENT_SOURCE_STATUS = "DTR_CARLA_C2_RICH_MULTILAYOUT_SOURCE_COMPLETE"
COHORT_ID = "DTR_CARLA_C36_X74_FRESH_CONFIRMATION_V1"
CAPTURE_SEED = 361074
WEATHERS = {
    "c8_l01": "ClearSunset",
    "c8_l02": "HardRainSunset",
    "c8_l03": "WetCloudyNoon",
    "c8_l04": "SoftRainNoon",
}
FROZEN_COMPONENTS = {
    "dtr_carla_x24_plan_adherent_predictor.py": "C37F2667EC119A855F69DE01132ABDD32D37C387B8CD7AFF5879046498631997",
    "dtr_carla_x25_rigid_footprint_predictor.py": "EE9A37ED6978C9E187D70E607288A129710B2C3C823F7D3E37CE6B1ECC088895",
    "dtr_carla_x54_metric_bootstrap_dropout_continuation.py": "F253AF7AF77202B5E38FC0D176CFAE2AEF786A90D741FF6ED76A78EB4609B1F3",
    "dtr_carla_x65_ancestry_synchronized_conflict_handback.py": "B87E444384CF6BE4A2B69A4B8536F9EA4CD10FE8A46DD9B5D0499A60AB94E4F1",
    "dtr_carla_x67_measurement_horizon_receding_release.py": "4E9719E605501F1DE078B71A93A19DDA065CDBF4B9DF059F9290BB89E06380B2",
    "dtr_carla_x68_object_local_lateral_dequantization.py": "48B354246BEEF6287AFB961880B81FF54BD2423059385FA485FC599FDC4A9D1E",
    "dtr_carla_x69_mature_cross_route_rigid_contradiction.py": "4DA147DCF99CB45BEFA79AAF63D33D3C2798A73DCA7A9B2B91A01912DFE95E60",
    "dtr_carla_x70_triple_credential_surface_dropout_handback.py": "7D438E9F852CDA2380CC8E95D95E89C6FB341D9F30582042B8B8DF55539D5245",
    "dtr_carla_x71_entry_cotransport_occupancy_birth.py": "D67A9A4722A2AD212C2137A0E8C666FF750DFA3556F486D788B25ED9D15E1FFA",
    "dtr_carla_x72_credentialed_surface_boundary_completion.py": "30F2E21893D34EF71FC4D8D74DFA80C3C38A6047F3B0E3BC64FB7D818C23CE82",
    "dtr_carla_x73_credentialed_parent_hull_reconstruction.py": "8722FAB54E441459EDE6E1EBE61CE1BE0FD7E8956BB2C9B139BF67E3BF51BBD2",
    "dtr_carla_x74_metric_handback_class_contradiction.py": "52558F7999258B4966C43A6473793E364D170111C90BD41BAC3FDE55F033289E",
}
CAPTURE_SCRIPT_SHA256 = (
    "3E292F66B066E215B701E3595642EC6510248C0E0C140002339D56A5554A279B"
)
JOIN_SCRIPT_SHA256 = (
    "D41012C0E016AC1149427FBC35ACF4FD2810F7E15B50650792718EDEC6D28636"
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def materialize(parent: dict) -> dict:
    protocol = copy.deepcopy(parent)
    protocol["schema_version"] = 36
    protocol["cohort_id"] = COHORT_ID
    protocol["capture"]["seed"] = CAPTURE_SEED
    protocol["objective"] = (
        "Confirm byte-frozen X74 once on genuinely new seed- and render-domain-"
        "disjoint CARLA pixels after its six-cohort consumed precision gain, "
        "while preserving camera, route, geometry, and trajectories."
    )
    for layout_id, weather in WEATHERS.items():
        protocol["layouts"][layout_id]["weather"] = weather

    protocol.pop("c35_x73_preregistration")
    protocol["c36_x74_preregistration"] = {
        "schema": "dtr-carla-c36-x74-preregistration-v1",
        "baselines": ["X24", "X73", "X74"],
        "single_x74_scored_invocation": True,
        "no_post_capture_algorithm_threshold_or_scenario_tuning": True,
        "score_window_seconds": [0.0, 6.0],
        "frozen_component_sha256": FROZEN_COMPONENTS,
        "frozen_capture_script_sha256": CAPTURE_SCRIPT_SHA256,
        "frozen_join_script_sha256": JOIN_SCRIPT_SHA256,
        "primary_transfer_gate": {
            "minimum_precision": 0.85,
            "minimum_recall": 0.70,
            "minimum_f1": 0.78,
            "minimum_each_contact_recall": 0.55,
            "maximum_safe_false_alert_segments_per_episode": 4,
            "maximum_total_safe_false_alert_segments": 10,
            "minimum_x74_class_contradiction_release_frames": 1,
            "minimum_x74_tp_delta_vs_x73": 0,
            "maximum_x74_fp_delta_vs_x73": -1,
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
        "schema": "dtr-carla-c36-x74-fresh-source-contract-v1",
        "parent_cohort_id": parent["cohort_id"],
        "parent_protocol_sha256": PARENT_PROTOCOL_SHA256,
        "parent_source_result_sha256": PARENT_SOURCE_RESULT_SHA256,
        "parent_source_status": PARENT_SOURCE_STATUS,
        "c36_is_new_cohort_not_c35_retry": True,
        "source_change": "NEW_CAPTURE_SEED_AND_RENDER_DOMAIN_ASSIGNMENT_ONLY",
        "new_capture_seed": CAPTURE_SEED,
        "changed_weather_by_layout": WEATHERS,
        "camera_geometry_unchanged_from_c35": True,
        "target_and_occluder_trajectories_unchanged_from_c35": True,
        "route_and_truth_contract_unchanged_from_c35": True,
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
        "C36 uses a new seed, render-domain assignment, and pixels; it does not reuse C35 payloads.",
        "X74 and every imported algorithm component are byte-frozen before C36 capture.",
        "The sole C36 score confirms X74 only if class contradiction is exercised, loses no TP versus X73, and removes at least one FP.",
        "A mechanism-not-exercised result is not positive or negative incremental confirmation.",
        "C36 is source-disjoint synthetic Development, not unseen-map, open-world traffic, real-sensor, deployment, reliability, user-benefit, or safety evidence.",
        "A failed source gate is NOT_EVALUABLE; frozen X74 cannot be changed or rerun on C36 after pixels are opened.",
    ]
    c2.validate_protocol(protocol)
    return protocol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c35-source-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if c2.sha256_file(PARENT_PROTOCOL) != PARENT_PROTOCOL_SHA256:
        raise RuntimeError("frozen C35 parent protocol drift")
    source_result_path = args.c35_source_result.resolve(strict=True)
    if c2.sha256_file(source_result_path) != PARENT_SOURCE_RESULT_SHA256:
        raise RuntimeError("frozen C35 source result drift")
    source_result = read_json(source_result_path)
    if source_result.get("status") != PARENT_SOURCE_STATUS:
        raise RuntimeError("C35 source result status drift")
    for file_name, expected in FROZEN_COMPONENTS.items():
        path = HERE / file_name
        if not path.is_file() or c2.sha256_file(path) != expected:
            raise RuntimeError(f"frozen C36 component drift: {file_name}")
    for path, expected in {
        HERE / "capture_dtr_carla_c2_rich_scene.py": CAPTURE_SCRIPT_SHA256,
        HERE / "join_dtr_carla_c2_rich_scene.py": JOIN_SCRIPT_SHA256,
    }.items():
        if c2.sha256_file(path) != expected:
            raise RuntimeError(f"frozen C36 source script drift: {path.name}")

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
        "status": "C36_X74_PROTOCOL_STATIC_VALID_PREREGISTERED",
        "output": str(output),
        "sha256": c2.sha256_file(output),
        "cohort_id": COHORT_ID,
        "capture_seed": CAPTURE_SEED,
        "weather_domains": WEATHERS,
        "x74_sha256": FROZEN_COMPONENTS["dtr_carla_x74_metric_handback_class_contradiction.py"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
