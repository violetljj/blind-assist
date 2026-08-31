"""Materialize fresh mixed-lighting C28 confirmation for frozen X59."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import dtr_carla_c2_rich_scene as c2
import materialize_dtr_carla_c17_x42_counterfactual_protocol as c17
import materialize_dtr_carla_c27_x57_daylight_transfer_protocol as c27


COHORT_ID = "DTR_CARLA_C28_X59_MIXED_LIGHTING_TRANSFER_CONFIRMATION_V1"
CAPTURE_SEED = 281006
C27_PROTOCOL_SHA256 = (
    "263BAED96E23CD009979412A71A4396728A898A740094C3431E9942135CACDEB"
)
C27_SOURCE_RESULT_SHA256 = (
    "9C9D020642B34D611EE00C33E586195FFD5A79DE20A4A0B6FD7321071DABE8DB"
)
C27_X57_SUMMARY_SHA256 = (
    "51429E5E41D534ACA950D4C2B44169EC6A513782B4E57793A1DE7DBF26B2457D"
)
C27_X59_DEVELOPMENT_SHA256 = (
    "A1F696AF1554BBE95FD47ED44482A511BB986A77DF5ACE0B6D8F4914CD44F0C8"
)
C26_X59_DEVELOPMENT_SHA256 = (
    "68BF257BFA42863C24ABAF2C773A5C3CE356CA6E59476D0DFEEA9CC7FC299106"
)
X59_SHA256 = "A81C3FFE2C4942FCFB45B5CD3D7452DC705F47ADF40B57FE7902E10AE7455C47"
WEATHERS = {
    "c8_l01": "ClearNight",
    "c8_l02": "WetSunset",
    "c8_l03": "SoftRainNoon",
    "c8_l04": "HardRainSunset",
}
COMPONENT_SHA256 = dict(c27.COMPONENT_SHA256)
COMPONENT_SHA256["dtr_carla_x59_modality_evidence_reliability_router.py"] = (
    X59_SHA256
)
CAPTURE_SCRIPT_SHA256 = c27.CAPTURE_SCRIPT_SHA256
JOIN_SCRIPT_SHA256 = c27.JOIN_SCRIPT_SHA256


def materialize(base: dict) -> dict:
    protocol = c27.materialize(base)
    protocol["schema_version"] = 28
    protocol["cohort_id"] = COHORT_ID
    protocol["capture"]["seed"] = CAPTURE_SEED
    protocol["objective"] = (
        "Confirm frozen X59 modality-evidence reliability routing once on "
        "fresh cross-combinations of night, sunset, noon, wet, and rain."
    )
    for layout_id, weather in WEATHERS.items():
        protocol["layouts"][layout_id]["weather"] = weather

    old_prereg = protocol.pop("c27_x57_preregistration")
    gates = old_prereg["primary_transfer_gate"]
    protocol["c28_x59_preregistration"] = {
        "schema": "dtr-carla-c28-x59-preregistration-v1",
        "baselines": ["X24", "X54", "X59"],
        "frozen_capture_script_sha256": CAPTURE_SCRIPT_SHA256,
        "frozen_join_script_sha256": JOIN_SCRIPT_SHA256,
        "frozen_component_sha256": COMPONENT_SHA256,
        "single_x59_scored_invocation": True,
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
            "minimum_x59_modality_reliability_actions": 1,
            "required_zero_authority_invariants": gates[
                "required_zero_authority_invariants"
            ],
        },
        "stretch_target": old_prereg["stretch_target"],
        "inheritance_roles": {
            "X54_ROUTE_RISK_CORE": "RETAINED_CORE",
            "X57_ZERO_ELIGIBLE_METRIC_HANDBACK": "COMPONENT",
            "X58_UNGATED_BIDIRECTIONAL_ROUTER": "NEGATIVE_CONTROL",
            "X59_MODALITY_EVIDENCE_RELIABILITY_ROUTER": "CHALLENGER",
        },
    }
    protocol["source_disjoint_contract"] = {
        "schema": "dtr-carla-c28-x59-mixed-lighting-contract-v1",
        "development_parent_cohort_id": c27.COHORT_ID,
        "development_parent_protocol_sha256": C27_PROTOCOL_SHA256,
        "development_parent_source_result_sha256": C27_SOURCE_RESULT_SHA256,
        "development_parent_x57_summary_sha256": C27_X57_SUMMARY_SHA256,
        "development_parent_c27_x59_summary_sha256": C27_X59_DEVELOPMENT_SHA256,
        "development_parent_c26_x59_summary_sha256": C26_X59_DEVELOPMENT_SHA256,
        "development_parent_status": "TWO_CONSUMED_COHORTS_DEVELOPMENT_ONLY",
        "new_capture_seed": CAPTURE_SEED,
        "fresh_pixels": True,
        "changed_weather_by_layout": WEATHERS,
        "source_valid_c27_target_motion_profiles_preserved": True,
        "preserved_route_occlusion_blueprint_and_lateral_twin_geometry": True,
        "model_sensor_replay_projection": "ACTUAL_ALL_ACTORS",
        "critical_contact_outcome_truth_must_match_across_all_sensors": True,
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
        "C28 is frozen before capture and reuses no C22-C27 pixels.",
        "C28 preserves source-valid C27 motion profiles and changes seed plus all four weather-lighting combinations.",
        "X59 is byte-frozen before capture and consumes no weather or lighting label.",
        "A C28 gate confirms X59 only as source-disjoint synthetic Development when at least one X59 modality-reliability action is exercised.",
        "C28 cannot establish unseen-map, open-world traffic, real-sensor, deployment, reliability, user-benefit, or safety authority.",
        "A failed source gate is NOT_EVALUABLE; frozen X59 cannot be changed or rerun on C28 after pixels are opened.",
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
        here / "dtr_carla_c27_x57_daylight_transfer_protocol.json": (
            C27_PROTOCOL_SHA256
        ),
        **{here / name: digest for name, digest in COMPONENT_SHA256.items()},
    }
    for path, expected in frozen_files.items():
        if c2.sha256_file(path) != expected:
            raise RuntimeError(f"frozen C28 component drift: {path.name}")
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
                "status": "C28_X59_PROTOCOL_STATIC_VALID_PREREGISTERED",
                "output": str(output),
                "sha256": c2.sha256_file(output),
                "cohort_id": COHORT_ID,
                "capture_seed": CAPTURE_SEED,
                "weather_domains": WEATHERS,
                "x59_sha256": X59_SHA256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
