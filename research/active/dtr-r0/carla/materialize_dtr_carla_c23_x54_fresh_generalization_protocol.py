"""Materialize the fresh C23 generalization confirmation protocol for X54."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import dtr_carla_c2_rich_scene as c2
import materialize_dtr_carla_c17_x42_counterfactual_protocol as c17
import materialize_dtr_carla_c22_x44_kinematic_transfer_protocol as c22


COHORT_ID = "DTR_CARLA_C23_X54_FRESH_GENERALIZATION_CONFIRMATION_V1"
CAPTURE_SEED = 230954
WEATHERS = {
    "c8_l01": "ClearSunset",
    "c8_l02": "WetNoon",
    "c8_l03": "SoftRainNoon",
    "c8_l04": "CloudySunset",
}
MOTION_PROFILES = {
    "c8_l01": [
        {"start_s": 0.0, "velocity_forward_mps": -2.05, "velocity_right_mps": 0.0},
        {"start_s": 1.1, "velocity_forward_mps": -1.9, "velocity_right_mps": 0.15},
        {"start_s": 2.5, "velocity_forward_mps": -1.45, "velocity_right_mps": -0.15},
        {"start_s": 4.5, "velocity_forward_mps": 0.0, "velocity_right_mps": 0.0},
    ],
    "c8_l02": [
        {"start_s": 0.0, "velocity_forward_mps": -1.65, "velocity_right_mps": 0.0},
        {"start_s": 1.25, "velocity_forward_mps": -2.3, "velocity_right_mps": -0.22},
        {"start_s": 2.7, "velocity_forward_mps": -1.95, "velocity_right_mps": 0.22},
        {"start_s": 4.55, "velocity_forward_mps": 0.0, "velocity_right_mps": 0.0},
    ],
    "c8_l03": [
        {"start_s": 0.0, "velocity_forward_mps": -2.55, "velocity_right_mps": 0.0},
        {"start_s": 1.35, "velocity_forward_mps": -2.05, "velocity_right_mps": 0.3},
        {"start_s": 2.35, "velocity_forward_mps": -1.55, "velocity_right_mps": -0.3},
        {"start_s": 4.4, "velocity_forward_mps": 0.0, "velocity_right_mps": 0.0},
    ],
    "c8_l04": [
        {"start_s": 0.0, "velocity_forward_mps": -2.15, "velocity_right_mps": 0.0},
        {"start_s": 1.2, "velocity_forward_mps": -2.45, "velocity_right_mps": -0.35},
        {"start_s": 2.6, "velocity_forward_mps": -1.85, "velocity_right_mps": 0.35},
        {"start_s": 4.5, "velocity_forward_mps": 0.0, "velocity_right_mps": 0.0},
    ],
}
COMPONENT_SHA256 = {
    "dtr_carla_x45_causal_state_cycle_credential.py": "7B58417F3195844C631ABCA0A2A6432200F993B6C13EC590AB71255CC341CFAB",
    "dtr_carla_x46_evidence_terminated_object_permanence.py": "9319E9D4A15B2B7A0A269D9918B29DCD90E2EE909CBDF962D0BAE0D88B9C669D",
    "dtr_carla_x47_depth_free_space_permanence_release.py": "9D6FF67A7ACD536B4591071B63EC6FDB525999E2133C47E42614152479EDAB8A",
    "dtr_carla_x51_provisional_motion_belief_update.py": "E28BB465DA77BA69F618B225FF4F30FA37C700E5288E03499E349EC6B2804BBA",
    "dtr_carla_x52_cross_parent_provisional_reidentification.py": "A6E6C35377535180AAE17C11EDD575E251F6AC67D1738C8EA00D1B93D734DE33",
    "dtr_carla_x53_anchor_redundant_parent_continuation.py": "44D8D18752ABD640790D5ED6985A26FFB20B6F61116E039B3EE77CC6F3218B2E",
    "dtr_carla_x54_metric_bootstrap_dropout_continuation.py": "F253AF7AF77202B5E38FC0D176CFAE2AEF786A90D741FF6ED76A78EB4609B1F3",
}
CAPTURE_SCRIPT_SHA256 = "3E292F66B066E215B701E3595642EC6510248C0E0C140002339D56A5554A279B"
JOIN_SCRIPT_SHA256 = "D41012C0E016AC1149427FBC35ACF4FD2810F7E15B50650792718EDEC6D28636"
C22_PROTOCOL_SHA256 = "B6E1632C83525D3298683B913623DD5598DB9415943A6968E5C9F5C5F149E8D2"
C22_X54_SUMMARY_SHA256 = "FD31ECE5173567566E57E4257D90B1866A6FCC2182F26E3FF2D63CF0BB289AD6"


def materialize(base: dict) -> dict:
    protocol = c22.materialize(base)
    protocol["schema_version"] = 23
    protocol["cohort_id"] = COHORT_ID
    protocol["capture"]["seed"] = CAPTURE_SEED
    protocol["objective"] = (
        "Confirm frozen X54 once on fresh pixels, four changed weather/render "
        "domains, and four unseen piecewise target kinematics while preserving "
        "the admitted route, occlusion, blueprint, and lateral twin geometry."
    )
    for layout_id, weather in WEATHERS.items():
        protocol["layouts"][layout_id]["weather"] = weather
        for twin_role in ("contact", "safe"):
            protocol["trajectory_library"][f"{layout_id}_target_{twin_role}"][
                "segments"
            ] = MOTION_PROFILES[layout_id]

    protocol.pop("c22_x44_preregistration", None)
    protocol["c23_x54_preregistration"] = {
        "schema": "dtr-carla-c23-x54-preregistration-v1",
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
        "schema": "dtr-carla-c23-x54-fresh-generalization-contract-v1",
        "parent_confirmation_cohort_id": c22.COHORT_ID,
        "parent_protocol_sha256": C22_PROTOCOL_SHA256,
        "new_capture_seed": CAPTURE_SEED,
        "fresh_pixels": True,
        "changed_weather_by_layout": WEATHERS,
        "changed_target_motion_profiles_by_layout": MOTION_PROFILES,
        "preserved_route_occlusion_blueprint_and_lateral_twin_geometry": True,
        "model_sensor_replay_projection": "ACTUAL_ALL_ACTORS",
        "critical_contact_outcome_truth_must_match_across_all_sensors": True,
        "prior_pixels_reused": False,
        "future_truth_visible_to_predictor": False,
        "single_scored_invocation": True,
        "development_evidence_only": {
            "c22_x54_summary_sha256": C22_X54_SUMMARY_SHA256,
            "status": "C22_POSTHOC_DEVELOPMENT_ONLY",
        },
    }
    protocol["claim_boundary"] = [
        "C23 is frozen before capture and reuses no C22 pixels.",
        "C23 changes seed, weather/render domain, and target kinematics; route, occlusion layout, target blueprint, and lateral contact/safe geometry remain inherited.",
        "Wearable RGB and metric depth retain exact all-actor actual-state replay against the instance evaluator.",
        "A C23 gate confirms X54 only as source-disjoint synthetic Development and only when all three successor mechanisms are exercised.",
        "C23 cannot establish unseen-map, open-world traffic, real-sensor, deployment, reliability, user-benefit, or safety authority.",
        "A failed source gate is NOT_EVALUABLE; frozen X54 cannot be changed or rerun on C23 after pixels are opened.",
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
        **{here / name: digest for name, digest in COMPONENT_SHA256.items()},
    }
    for path, expected in frozen_files.items():
        if c2.sha256_file(path) != expected:
            raise RuntimeError(f"frozen C23 component drift: {path.name}")
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
                "status": "C23_X54_PROTOCOL_STATIC_VALID_PREREGISTERED",
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
