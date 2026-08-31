"""Materialize fresh C25 night/adverse-weather confirmation for frozen X56."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import dtr_carla_c2_rich_scene as c2
import materialize_dtr_carla_c17_x42_counterfactual_protocol as c17
import materialize_dtr_carla_c24_x54_source_corrected_generalization_protocol as c24


COHORT_ID = "DTR_CARLA_C25_X56_GRACEFUL_DEGRADATION_CONFIRMATION_V1"
CAPTURE_SEED = 250966
WEATHERS = {
    "c8_l01": "CloudyNight",
    "c8_l02": "WetNight",
    "c8_l03": "SoftRainNight",
    "c8_l04": "DustStorm",
}
C24_PROTOCOL_SHA256 = (
    "7767473E7EF9EEE7445E915EC2EF095F496BF1E1CB3524D957FC131555CD260B"
)
X55_SHA256 = "9D299C1CD8D890B881576CC950DD6ADB9D0D40576BA66ACA45C8BEDBA4A06717"
X56_SHA256 = "06AF954FADD5C82B433CB431F5A25F698AE6880A8D99284948CDF362AFE72F17"
X56_DEVELOPMENT_SUMMARY_SHA256 = (
    "787112A23103FE70B509232E33221D248144E2AD8149E32A045FF1D359FB851B"
)
X56_DEVELOPMENT_PREDICTIONS_SHA256 = (
    "0C6D065BE4C042FEED13CE7E4AB3916C54BC482E89AA32C23C01E0701547D747"
)
COMPONENT_SHA256 = {
    **c24.COMPONENT_SHA256,
    "dtr_carla_x55_parent_sibling_state_cycle_consensus.py": X55_SHA256,
    "dtr_carla_x56_zero_eligible_fusion_metric_handback.py": X56_SHA256,
}
CAPTURE_SCRIPT_SHA256 = c24.CAPTURE_SCRIPT_SHA256
JOIN_SCRIPT_SHA256 = c24.JOIN_SCRIPT_SHA256


def mirror_lateral_motion(segments: list[dict]) -> list[dict]:
    return [
        {
            **segment,
            "velocity_right_mps": -float(segment["velocity_right_mps"]),
        }
        for segment in segments
    ]


def materialize(base: dict) -> dict:
    protocol = c24.materialize(base)
    protocol["schema_version"] = 25
    protocol["cohort_id"] = COHORT_ID
    protocol["capture"]["seed"] = CAPTURE_SEED
    protocol["objective"] = (
        "Confirm frozen X56 once on fresh pixels under four unseen night or "
        "dust render domains and three mirrored lateral-motion profiles, "
        "testing graceful degradation when the fused representation becomes empty."
    )
    for layout_id, weather in WEATHERS.items():
        protocol["layouts"][layout_id]["weather"] = weather

    mirrored_layouts = ["c8_l01", "c8_l03", "c8_l04"]
    for layout_id in mirrored_layouts:
        for twin_role in ("contact", "safe"):
            key = f"{layout_id}_target_{twin_role}"
            protocol["trajectory_library"][key]["segments"] = mirror_lateral_motion(
                protocol["trajectory_library"][key]["segments"]
            )

    protocol.pop("c24_x54_preregistration", None)
    protocol["c25_x56_preregistration"] = {
        "schema": "dtr-carla-c25-x56-preregistration-v1",
        "frozen_component_sha256": COMPONENT_SHA256,
        "frozen_capture_script_sha256": CAPTURE_SCRIPT_SHA256,
        "frozen_join_script_sha256": JOIN_SCRIPT_SHA256,
        "baselines": ["X24", "X54", "X56"],
        "score_window_seconds": [0.0, 6.0],
        "single_x56_scored_invocation": True,
        "no_post_capture_algorithm_threshold_or_scenario_tuning": True,
        "retired_mechanism_requirements": [
            "X52_CROSS_PARENT_PROVISIONAL_REIDENTIFICATION"
        ],
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
            "maximum_safe_false_alert_segments_per_episode": 4,
            "maximum_total_safe_false_alert_segments": 10,
            "minimum_x53_anchor_redundancy_suppressions": 1,
            "minimum_x54_dropout_continuations": 1,
            "minimum_x55_parent_sibling_consensus_frames": 1,
            "minimum_x56_zero_eligible_metric_handback_frames": 1,
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
        "stretch_target": {"precision": 0.90, "recall": 0.75, "f1": 0.82},
    }
    protocol["source_disjoint_contract"] = {
        "schema": "dtr-carla-c25-x56-graceful-degradation-contract-v1",
        "development_parent_cohort_id": c24.COHORT_ID,
        "development_parent_protocol_sha256": C24_PROTOCOL_SHA256,
        "development_parent_x56_summary_sha256": X56_DEVELOPMENT_SUMMARY_SHA256,
        "development_parent_x56_predictions_sha256": (
            X56_DEVELOPMENT_PREDICTIONS_SHA256
        ),
        "development_parent_status": (
            "CONSUMED_POSTHOC_DEVELOPMENT_REFERENCE_TARGET_MET_NOT_CONFIRMATION"
        ),
        "new_capture_seed": CAPTURE_SEED,
        "fresh_pixels": True,
        "changed_weather_by_layout": WEATHERS,
        "unseen_render_domains": list(WEATHERS.values()),
        "mirrored_lateral_motion_layouts": mirrored_layouts,
        "source_corrections_inherited_from_c24": True,
        "preserved_route_occlusion_blueprint_and_lateral_twin_geometry": True,
        "model_sensor_replay_projection": "ACTUAL_ALL_ACTORS",
        "critical_contact_outcome_truth_must_match_across_all_sensors": True,
        "prior_pixels_reused": False,
        "future_truth_visible_to_predictor": False,
        "single_scored_invocation": True,
    }
    protocol["external_design_evidence"] = {
        "grace_bev": {
            "url": "https://arxiv.org/html/2605.30983",
            "design_link": (
                "active reliability routing can preserve an available modality "
                "when static fusion collapses"
            ),
        },
        "unibev": {
            "url": "https://arxiv.org/html/2309.14516v2",
            "design_link": (
                "aligned modality representations should remain independently operable"
            ),
        },
    }
    protocol["claim_boundary"] = [
        "C25 is frozen before capture and reuses no C22-C24 pixels.",
        "C25 changes the capture seed, all four render domains, and mirrors lateral target motion in three layouts; layouts, route, occluders, detector, and numeric thresholds remain fixed.",
        "X55 and X56 are frozen from consumed C24 Development before C25 pixels exist.",
        "A C25 gate confirms X56 only as source-disjoint synthetic Development and only when X53-X56 successor mechanisms are exercised.",
        "C25 cannot establish unseen-map, open-world traffic, real-sensor, deployment, reliability, user-benefit, or safety authority.",
        "A failed source gate is NOT_EVALUABLE; frozen X56 cannot be changed or rerun on C25 after pixels are opened.",
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
        here / "dtr_carla_c24_x54_source_corrected_generalization_protocol.json": (
            C24_PROTOCOL_SHA256
        ),
        **{here / name: digest for name, digest in COMPONENT_SHA256.items()},
    }
    for path, expected in frozen_files.items():
        if c2.sha256_file(path) != expected:
            raise RuntimeError(f"frozen C25 component drift: {path.name}")
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
                "status": "C25_X56_PROTOCOL_STATIC_VALID_PREREGISTERED",
                "output": str(output),
                "sha256": c2.sha256_file(output),
                "cohort_id": COHORT_ID,
                "capture_seed": CAPTURE_SEED,
                "x55_sha256": X55_SHA256,
                "x56_sha256": X56_SHA256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
