"""Materialize fresh daylight/adverse-weather C27 confirmation for frozen X57."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import dtr_carla_c2_rich_scene as c2
import materialize_dtr_carla_c17_x42_counterfactual_protocol as c17
import materialize_dtr_carla_c26_x56_source_corrected_protocol as c26


COHORT_ID = "DTR_CARLA_C27_X57_DAYLIGHT_WEATHER_TRANSFER_CONFIRMATION_V1"
CAPTURE_SEED = 270986
C26_PROTOCOL_SHA256 = (
    "37335C08F91A6A3CC360094BBC672E7350484618D4E91C0E40E052F830D005FA"
)
C26_SOURCE_RESULT_SHA256 = (
    "BF9EECAA9D12E4CFD8B6AF9AEF2E4988B4BEDF5510B8CE5E73D7DAA321B52013"
)
C26_X56_SUMMARY_SHA256 = (
    "D2EC2E45F76010B5C1B00211E229449E85CDAF4854881A9629C9EA88E3558FEF4"
)
X57_SHA256 = "A5CC95CDB3E4CD0608458DE4FB6EB32FEBF6F1388A705DAAC4FEF0C96671AE9E"
WEATHERS = {
    "c8_l01": "ClearSunset",
    "c8_l02": "WetCloudyNoon",
    "c8_l03": "MidRainSunset",
    "c8_l04": "HardRainNoon",
}
COMPONENT_SHA256 = {
    name: digest
    for name, digest in c26.COMPONENT_SHA256.items()
    if name
    not in {
        "dtr_carla_x55_parent_sibling_state_cycle_consensus.py",
        "dtr_carla_x56_zero_eligible_fusion_metric_handback.py",
    }
}
COMPONENT_SHA256["dtr_carla_x57_retained_core_metric_handback.py"] = X57_SHA256
CAPTURE_SCRIPT_SHA256 = c26.CAPTURE_SCRIPT_SHA256
JOIN_SCRIPT_SHA256 = c26.JOIN_SCRIPT_SHA256


def materialize(base: dict) -> dict:
    protocol = c26.materialize(base)
    protocol["schema_version"] = 27
    protocol["cohort_id"] = COHORT_ID
    protocol["capture"]["seed"] = CAPTURE_SEED
    protocol["objective"] = (
        "Confirm frozen X57 retained-core reliability routing once on fresh "
        "daylight, sunset, wet, and rain pixels."
    )
    for layout_id, weather in WEATHERS.items():
        protocol["layouts"][layout_id]["weather"] = weather

    old_prereg = protocol.pop("c26_x56_preregistration")
    gates = old_prereg["primary_transfer_gate"]
    protocol["c27_x57_preregistration"] = {
        "schema": "dtr-carla-c27-x57-preregistration-v1",
        "baselines": ["X24", "X54", "X57"],
        "frozen_capture_script_sha256": CAPTURE_SCRIPT_SHA256,
        "frozen_join_script_sha256": JOIN_SCRIPT_SHA256,
        "frozen_component_sha256": COMPONENT_SHA256,
        "single_x57_scored_invocation": True,
        "no_post_capture_algorithm_threshold_or_scenario_tuning": True,
        "score_window_seconds": old_prereg["score_window_seconds"],
        "outcomes": [
            "GATE_MET",
            "GATE_NOT_MET",
            "SOURCE_NOT_EVALUABLE",
            "MECHANISM_NOT_EXERCISED",
        ],
        "primary_transfer_gate": {
            "minimum_precision": 0.85,
            "minimum_recall": 0.70,
            "minimum_f1": 0.78,
            "minimum_each_contact_recall": 0.55,
            "maximum_safe_false_alert_segments_per_episode": gates[
                "maximum_safe_false_alert_segments_per_episode"
            ],
            "maximum_total_safe_false_alert_segments": gates[
                "maximum_total_safe_false_alert_segments"
            ],
            "minimum_x57_zero_eligible_metric_handback_frames": 1,
            "required_zero_authority_invariants": gates[
                "required_zero_authority_invariants"
            ],
        },
        "stretch_target": old_prereg["stretch_target"],
        "inheritance_roles": {
            "X54_ROUTE_RISK_CORE": "RETAINED_CORE",
            "X56_ZERO_ELIGIBLE_METRIC_HANDBACK": "COMPONENT",
            "X55_PARENT_SIBLING_ROUTE_PROMOTION": "NEGATIVE_CONTROL",
            "X57_RETAINED_CORE_METRIC_HANDBACK": "CHALLENGER",
        },
    }
    protocol["source_disjoint_contract"] = {
        "schema": "dtr-carla-c27-x57-daylight-transfer-contract-v1",
        "development_parent_cohort_id": c26.COHORT_ID,
        "development_parent_protocol_sha256": C26_PROTOCOL_SHA256,
        "development_parent_source_result_sha256": C26_SOURCE_RESULT_SHA256,
        "development_parent_x56_summary_sha256": C26_X56_SUMMARY_SHA256,
        "development_parent_status": "CONSUMED_X57_POSTHOC_DEVELOPMENT_ONLY",
        "new_capture_seed": CAPTURE_SEED,
        "fresh_pixels": True,
        "changed_weather_by_layout": WEATHERS,
        "source_valid_c26_target_motion_profiles_preserved": True,
        "preserved_route_occlusion_blueprint_and_lateral_twin_geometry": True,
        "model_sensor_replay_projection": "ACTUAL_ALL_ACTORS",
        "critical_contact_outcome_truth_must_match_across_all_sensors": True,
        "prior_pixels_reused": False,
        "future_truth_visible_to_predictor": False,
        "single_scored_invocation": True,
    }
    protocol["claim_boundary"] = [
        "C27 is frozen before capture and reuses no C22-C26 pixels.",
        "C27 preserves the source-valid C26 motion profiles and changes seed plus all four weather and lighting domains.",
        "X57 is byte-frozen before capture; X54 remains the retained core, the X56 handback is a component, and X55 route promotion is a negative control for this role.",
        "A C27 gate confirms X57 only as source-disjoint synthetic Development when its handback mechanism is exercised.",
        "C27 cannot establish unseen-map, open-world traffic, real-sensor, deployment, reliability, user-benefit, or safety authority.",
        "A failed source gate is NOT_EVALUABLE; frozen X57 cannot be changed or rerun on C27 after pixels are opened.",
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
        here / "dtr_carla_c26_x56_source_corrected_protocol.json": (
            C26_PROTOCOL_SHA256
        ),
        **{here / name: digest for name, digest in COMPONENT_SHA256.items()},
    }
    for path, expected in frozen_files.items():
        if c2.sha256_file(path) != expected:
            raise RuntimeError(f"frozen C27 component drift: {path.name}")
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
                "status": "C27_X57_PROTOCOL_STATIC_VALID_PREREGISTERED",
                "output": str(output),
                "sha256": c2.sha256_file(output),
                "cohort_id": COHORT_ID,
                "capture_seed": CAPTURE_SEED,
                "weather_domains": WEATHERS,
                "x57_sha256": X57_SHA256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
