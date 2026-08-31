"""Materialize the source-corrected fresh C24 confirmation protocol for X54."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import dtr_carla_c2_rich_scene as c2
import materialize_dtr_carla_c17_x42_counterfactual_protocol as c17
import materialize_dtr_carla_c22_x44_kinematic_transfer_protocol as c22
import materialize_dtr_carla_c23_x54_fresh_generalization_protocol as c23


COHORT_ID = "DTR_CARLA_C24_X54_SOURCE_CORRECTED_GENERALIZATION_CONFIRMATION_V1"
CAPTURE_SEED = 240964
WEATHERS = {
    "c8_l01": "WetCloudySunset",
    "c8_l02": "MidRainyNoon",
    "c8_l03": "HardRainNoon",
    "c8_l04": "WetCloudyNoon",
}
COMPONENT_SHA256 = dict(c23.COMPONENT_SHA256)
CAPTURE_SCRIPT_SHA256 = c23.CAPTURE_SCRIPT_SHA256
JOIN_SCRIPT_SHA256 = c23.JOIN_SCRIPT_SHA256
C23_PROTOCOL_SHA256 = "741018AEF69CF00779FFA689928FE5D2F979DF0E665C8F53FCB752D8A8920C0B"
SOURCE_FIX_CANARY_RECEIPT_SHA256 = (
    "19EE5C914E45070619958D799D284F1A4D47BBC1A60D9A8394D7642E7D65A118"
)


def materialize(base: dict) -> dict:
    protocol = c23.materialize(base)
    protocol["schema_version"] = 24
    protocol["cohort_id"] = COHORT_ID
    protocol["capture"]["seed"] = CAPTURE_SEED
    protocol["objective"] = (
        "Confirm frozen X54 once on fresh pixels and four new weather/render "
        "domains, retaining three unseen C23 target-kinematic profiles while "
        "repairing only the two source contracts that made C23 not evaluable."
    )
    for layout_id, weather in WEATHERS.items():
        protocol["layouts"][layout_id]["weather"] = weather

    # C23 ep_03 had no post-reappearance samples. Restore only c8_l02 to the
    # already source-valid C22 profile; the other three unseen profiles remain.
    for twin_role in ("contact", "safe"):
        protocol["trajectory_library"][f"c8_l02_target_{twin_role}"]["segments"] = (
            c22.MOTION_PROFILES["c8_l02"]
        )

    # The C23 failure was exact actual-state divergence for this otherwise
    # irrelevant dynamic walker across fresh servers. A 728-frame, unscored
    # instance/depth canary established this minimal pose-authority correction.
    corrected_assets = 0
    for layout in protocol["layouts"].values():
        for asset in layout["assets"]:
            if asset["asset_key"] == "walker_cross_1":
                asset["collisions_enabled"] = False
                asset["scripted_pose_authority"] = True
                corrected_assets += 1
    if corrected_assets != 1:
        raise RuntimeError(
            f"expected exactly one walker_cross_1 source correction, found {corrected_assets}"
        )

    protocol.pop("c23_x54_preregistration", None)
    protocol["c24_x54_preregistration"] = {
        "schema": "dtr-carla-c24-x54-preregistration-v1",
        "frozen_component_sha256": COMPONENT_SHA256,
        "frozen_capture_script_sha256": CAPTURE_SCRIPT_SHA256,
        "frozen_join_script_sha256": JOIN_SCRIPT_SHA256,
        "baselines": ["X24", "X54"],
        "score_window_seconds": [0.0, 6.0],
        "single_x54_scored_invocation": True,
        "no_post_capture_algorithm_threshold_or_scenario_tuning": True,
        "outcomes": [
            "GATE_MET",
            "GATE_NOT_MET",
            "SOURCE_NOT_EVALUABLE",
            "MECHANISM_NOT_EXERCISED",
        ],
        "primary_transfer_gate": {
            "minimum_precision": 0.80,
            "minimum_recall": 0.70,
            "minimum_f1": 0.76,
            "minimum_each_contact_recall": 0.55,
            "maximum_safe_false_alert_segments_per_episode": 4,
            "maximum_total_safe_false_alert_segments": 10,
            "minimum_x52_provisional_parent_changes": 1,
            "minimum_x53_anchor_redundancy_suppressions": 1,
            "minimum_x54_dropout_continuations": 1,
            "required_continuous_contact_episodes": 3,
            "required_parent_ancestry_episodes": 3,
            "required_zero_authority_invariants": [
                "confirmed_missing_track_references",
                "confirmed_non_rigid_risk_track_references",
                "confirmed_parent_identity_mismatches",
                "route_risk_without_confirmed_eligible_track_frames",
                "route_risk_without_confirmed_rigid_dynamic_frames",
            ],
        },
        "stretch_target": {"precision": 0.86, "recall": 0.75, "f1": 0.80},
    }
    protocol["source_disjoint_contract"] = {
        "schema": "dtr-carla-c24-x54-source-corrected-generalization-contract-v1",
        "failed_parent_cohort_id": c23.COHORT_ID,
        "failed_parent_protocol_sha256": C23_PROTOCOL_SHA256,
        "failed_parent_status": "SOURCE_NOT_EVALUABLE_NO_ALGORITHM_SCORE",
        "new_capture_seed": CAPTURE_SEED,
        "fresh_pixels": True,
        "changed_weather_by_layout": WEATHERS,
        "retained_unseen_c23_motion_profiles": ["c8_l01", "c8_l03", "c8_l04"],
        "restored_source_valid_c22_motion_profiles": ["c8_l02"],
        "source_corrections": {
            "walker_cross_1": {
                "collisions_enabled": False,
                "scripted_pose_authority": True,
                "canary_receipt_sha256": SOURCE_FIX_CANARY_RECEIPT_SHA256,
                "canary_compared_frames": 728,
                "canary_actual_replay_mismatches": 0,
                "canary_algorithm_scoring_performed": False,
            },
            "c8_l02_target_motion": "restore C22 profile to recover post-reappearance window",
        },
        "preserved_route_occlusion_blueprint_and_lateral_twin_geometry": True,
        "model_sensor_replay_projection": "ACTUAL_ALL_ACTORS",
        "critical_contact_outcome_truth_must_match_across_all_sensors": True,
        "prior_pixels_reused": False,
        "future_truth_visible_to_predictor": False,
        "single_scored_invocation": True,
    }
    protocol["claim_boundary"] = [
        "C24 is frozen before capture and reuses no C22 or C23 pixels.",
        "C24 changes seed and all four weather/render domains; three C23-unseen target-motion profiles remain, while c8_l02 restores the C22 profile solely to recover a valid reappearance window.",
        "The only actor-policy change is the canary-supported collision-free scripted pose authority for walker_cross_1; X54 is unchanged.",
        "A C24 gate confirms X54 only as source-disjoint synthetic Development and only when all three successor mechanisms are exercised.",
        "C24 cannot establish unseen-map, open-world traffic, real-sensor, deployment, reliability, user-benefit, or safety authority.",
        "A failed source gate is NOT_EVALUABLE; frozen X54 cannot be changed or rerun on C24 after pixels are opened.",
    ]
    c2.validate_protocol(protocol)
    return protocol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    frozen_files = {
        here / "capture_dtr_carla_c2_rich_scene.py": CAPTURE_SCRIPT_SHA256,
        here / "join_dtr_carla_c2_rich_scene.py": JOIN_SCRIPT_SHA256,
        here / "dtr_carla_c23_x54_fresh_generalization_protocol.json": C23_PROTOCOL_SHA256,
        **{here / name: digest for name, digest in COMPONENT_SHA256.items()},
    }
    for path, expected in frozen_files.items():
        if c2.sha256_file(path) != expected:
            raise RuntimeError(f"frozen C24 component drift: {path.name}")
    protocol = materialize(c17._read_json(c17.BASE_PROTOCOL))
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
                "status": "C24_X54_PROTOCOL_STATIC_VALID_PREREGISTERED",
                "output": str(output),
                "sha256": c2.sha256_file(output),
                "cohort_id": COHORT_ID,
                "capture_seed": CAPTURE_SEED,
                "x54_sha256": COMPONENT_SHA256[
                    "dtr_carla_x54_metric_bootstrap_dropout_continuation.py"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
