"""Materialize the fresh C22 kinematic-transfer confirmation protocol for X44."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import dtr_carla_c2_rich_scene as c2
import materialize_dtr_carla_c17_x42_counterfactual_protocol as c17
import materialize_dtr_carla_c20_x42_fresh_visual_holdout_protocol as c20
import materialize_dtr_carla_c21_x43_weather_transfer_protocol as c21


COHORT_ID = "DTR_CARLA_C22_X44_KINEMATIC_TRANSFER_CONFIRMATION_V1"
CAPTURE_SEED = 220944
WEATHERS = {
    "c8_l01": "SoftRainSunset",
    "c8_l02": "ClearNoon",
    "c8_l03": "WetSunset",
    "c8_l04": "HardRainSunset",
}
MOTION_PROFILES = {
    "c8_l01": [
        {"start_s": 0.0, "velocity_forward_mps": -1.85, "velocity_right_mps": 0.0},
        {"start_s": 1.3, "velocity_forward_mps": -1.75, "velocity_right_mps": 0.12},
        {"start_s": 2.4, "velocity_forward_mps": -1.15, "velocity_right_mps": -0.12},
        {"start_s": 4.6, "velocity_forward_mps": 0.0, "velocity_right_mps": 0.0},
    ],
    "c8_l02": [
        {"start_s": 0.0, "velocity_forward_mps": -2.0, "velocity_right_mps": 0.0},
        {"start_s": 1.4, "velocity_forward_mps": -2.1, "velocity_right_mps": -0.18},
        {"start_s": 2.5, "velocity_forward_mps": -2.25, "velocity_right_mps": 0.18},
        {"start_s": 4.4, "velocity_forward_mps": 0.0, "velocity_right_mps": 0.0},
    ],
    "c8_l03": [
        {"start_s": 0.0, "velocity_forward_mps": -2.35, "velocity_right_mps": 0.0},
        {"start_s": 1.2, "velocity_forward_mps": -2.3, "velocity_right_mps": 0.25},
        {"start_s": 2.2, "velocity_forward_mps": -1.775, "velocity_right_mps": -0.25},
        {"start_s": 4.5, "velocity_forward_mps": 0.0, "velocity_right_mps": 0.0},
    ],
    "c8_l04": [
        {"start_s": 0.0, "velocity_forward_mps": -2.4, "velocity_right_mps": 0.0},
        {"start_s": 1.5, "velocity_forward_mps": -2.1, "velocity_right_mps": -0.3},
        {"start_s": 2.5, "velocity_forward_mps": -2.2, "velocity_right_mps": 0.3},
        {"start_s": 4.6, "velocity_forward_mps": 0.0, "velocity_right_mps": 0.0},
    ],
}
X44_PREDICTOR_SHA256 = (
    "1F56857E2A2E680DEF0F1AC842393DC1B67729FC037B71FD7A7E9F9BC7D0A5C6"
)
CAPTURE_SCRIPT_SHA256 = c21.CAPTURE_SCRIPT_SHA256
JOIN_SCRIPT_SHA256 = c21.JOIN_SCRIPT_SHA256
C21_PROTOCOL_SHA256 = (
    "05EAE555B6D0CF9F061EDB3B255733AAEE8D4E640A29A7CF3F0D51EF6CDFE649"
)
C21_X44_SUMMARY_SHA256 = (
    "BFD782DAD642C7EBE8EC3C47414C00EF548EF992B8152E9A5B984DDF64243D80"
)


def materialize(base: dict) -> dict:
    protocol = c21.materialize(base)
    protocol["schema_version"] = 22
    protocol["cohort_id"] = COHORT_ID
    protocol["capture"]["seed"] = CAPTURE_SEED
    protocol["objective"] = (
        "Confirm frozen X44 once on fresh target kinematics and four fresh "
        "weather/render domains while preserving admitted route, occlusion, "
        "target blueprint, and contact/safe lateral geometry."
    )
    for layout_id, weather in WEATHERS.items():
        protocol["layouts"][layout_id]["weather"] = weather
        for twin_role in ("contact", "safe"):
            key = f"{layout_id}_target_{twin_role}"
            protocol["trajectory_library"][key]["segments"] = MOTION_PROFILES[
                layout_id
            ]

    protocol.pop("c21_x43_preregistration", None)
    protocol["c22_x44_preregistration"] = {
        "schema": "dtr-carla-c22-x44-preregistration-v1",
        "frozen_x44_predictor_sha256": X44_PREDICTOR_SHA256,
        "frozen_capture_script_sha256": CAPTURE_SCRIPT_SHA256,
        "frozen_join_script_sha256": JOIN_SCRIPT_SHA256,
        "baselines": ["X24", "X43", "X44"],
        "score_window_seconds": [0.0, 6.0],
        "single_x43_x44_scored_invocation": True,
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
            "minimum_x44_tp_delta_vs_x43": 0,
            "maximum_x44_fp_delta_vs_x43": -1,
            "minimum_x44_f1_delta_vs_x43": 0.0,
            "minimum_velocity_cycle_suppressions": 1,
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
        "stretch_target": {
            "precision": 0.85,
            "recall": 0.75,
            "f1": 0.80,
        },
    }
    protocol["source_disjoint_contract"] = {
        "schema": "dtr-carla-c22-x44-kinematic-transfer-contract-v1",
        "parent_confirmation_cohort_id": c21.COHORT_ID,
        "parent_protocol_sha256": C21_PROTOCOL_SHA256,
        "new_capture_seed": CAPTURE_SEED,
        "fresh_pixels": True,
        "changed_weather_by_layout": WEATHERS,
        "changed_target_motion_profiles_by_layout": MOTION_PROFILES,
        "target_blueprints_by_layout": c20.TARGET_TEMPLATES,
        "preserved_route_occlusion_and_lateral_twin_geometry": True,
        "model_sensor_replay_projection": "ACTUAL_ALL_ACTORS",
        "witness_replay_projection": c21.WITNESS_REPLAY_PROJECTION,
        "critical_contact_outcome_truth_must_match_across_all_sensors": True,
        "prior_pixels_reused": False,
        "future_truth_visible_to_predictor": False,
        "single_scored_invocation": True,
        "development_evidence_only": {
            "c21_x44_summary_sha256": C21_X44_SUMMARY_SHA256,
            "status": "C21_POSTHOC_DEVELOPMENT_ONLY",
        },
    }
    protocol["claim_boundary"] = [
        "C22 is frozen before capture and uses no C21 pixels.",
        "C22 tests fresh target kinematics plus fresh weather/render transfer; map, route, occlusion layout, target blueprints, and lateral contact/safe geometry remain inherited.",
        "Wearable RGB and metric depth retain exact all-actor actual-state replay against the instance evaluator.",
        "A C22 gate can confirm X44 only if the frozen velocity-cycle mechanism is exercised and removes at least one X43 false positive without losing a true positive.",
        "C22 cannot establish unseen-map, open-world traffic, real-sensor, deployment, reliability, user-benefit, or safety authority.",
        "A failed source gate is NOT_EVALUABLE; X44 cannot be changed or rerun on C22 after pixels are opened.",
    ]
    c2.validate_protocol(protocol)
    return protocol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if c2.sha256_file(c17.BASE_PROTOCOL) != c17.BASE_PROTOCOL_SHA256:
        raise RuntimeError("C16 base protocol drift")
    here = Path(__file__).resolve().parent
    frozen_files = {
        here / "capture_dtr_carla_c2_rich_scene.py": CAPTURE_SCRIPT_SHA256,
        here / "join_dtr_carla_c2_rich_scene.py": JOIN_SCRIPT_SHA256,
        here / "dtr_carla_x44_causal_velocity_cycle_credential.py": (
            X44_PREDICTOR_SHA256
        ),
    }
    for path, expected in frozen_files.items():
        if c2.sha256_file(path) != expected:
            raise RuntimeError(f"frozen C22 component drift: {path.name}")
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
                "status": "C22_X44_PROTOCOL_STATIC_VALID_PREREGISTERED",
                "output": str(output),
                "sha256": c2.sha256_file(output),
                "cohort_id": COHORT_ID,
                "capture_seed": CAPTURE_SEED,
                "x44_sha256": X44_PREDICTOR_SHA256,
                "join_sha256": JOIN_SCRIPT_SHA256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
