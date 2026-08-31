"""Materialize the fresh C21 weather-transfer confirmation protocol for X43."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import dtr_carla_c2_rich_scene as c2
import materialize_dtr_carla_c17_x42_counterfactual_protocol as c17
import materialize_dtr_carla_c20_x42_fresh_visual_holdout_protocol as c20


COHORT_ID = "DTR_CARLA_C21_X43_WEATHER_TRANSFER_CONFIRMATION_V1"
CAPTURE_SEED = 210943
WEATHERS = {
    "c8_l01": "ClearSunset",
    "c8_l02": "WetNoon",
    "c8_l03": "HardRainNoon",
    "c8_l04": "CloudySunset",
}
WITNESS_REPLAY_PROJECTION = (
    "AUTHORITATIVE_ACTUAL_PLUS_NONAUTHORITATIVE_SCRIPTED_COMMAND"
)
X43_PREDICTOR_SHA256 = (
    "9BC23F349AF5FA5898B704D421341686E20EAB2270FD9F4070F1BE80F6B81BD8"
)
CAPTURE_SCRIPT_SHA256 = (
    "3E292F66B066E215B701E3595642EC6510248C0E0C140002339D56A5554A279B"
)
JOIN_SCRIPT_SHA256 = (
    "D41012C0E016AC1149427FBC35ACF4FD2810F7E15B50650792718EDEC6D28636"
)
C20_PROTOCOL_SHA256 = (
    "71DDCB32C89AC2D8480153FCCADF8360237BF34D5800BDA215DA71B0A19D2A6E"
)
C20_X43_SUMMARY_SHA256 = (
    "E6E6F6DC329C4C15864026DB3765123D82BE1C5389AC03B6EC9D2183E9E1A2BC"
)
C20_TRANSPORT_ADJUDICATION_SHA256 = (
    "5BE82B523C2490B9DDDD0B8AC3EF4A2E1DC71854E58B559B96BAD2CD6E563266"
)


def materialize(base: dict) -> dict:
    protocol = c20.materialize(base)
    protocol["schema_version"] = 21
    protocol["cohort_id"] = COHORT_ID
    protocol["capture"]["seed"] = CAPTURE_SEED
    protocol["capture"]["witness_replay_projection"] = WITNESS_REPLAY_PROJECTION
    protocol["objective"] = (
        "Confirm frozen X43 once on a fresh four-weather visual transfer while "
        "preserving the C20-admitted counterfactual motion, route, occlusion, "
        "and target geometry.  Model sensors retain exact actual-state replay; "
        "the diagnostic witness admits only preregistered non-authoritative "
        "scripted-command equivalence."
    )
    for layout_id, weather in WEATHERS.items():
        protocol["layouts"][layout_id]["weather"] = weather

    protocol.pop("c17_x42_preregistration", None)
    protocol["c21_x43_preregistration"] = {
        "schema": "dtr-carla-c21-x43-preregistration-v1",
        "frozen_x43_predictor_sha256": X43_PREDICTOR_SHA256,
        "frozen_capture_script_sha256": CAPTURE_SCRIPT_SHA256,
        "frozen_join_script_sha256": JOIN_SCRIPT_SHA256,
        "baselines": ["X24", "X43"],
        "score_window_seconds": [0.0, 6.0],
        "single_x43_scored_invocation": True,
        "no_post_capture_algorithm_threshold_or_scenario_tuning": True,
        "outcomes": ["GATE_MET", "GATE_NOT_MET", "SOURCE_NOT_EVALUABLE"],
        "primary_transfer_gate": {
            "minimum_precision": 0.68,
            "minimum_recall": 0.68,
            "minimum_f1": 0.68,
            "minimum_f1_gain_over_x24": 0.15,
            "minimum_each_contact_recall": 0.55,
            "maximum_safe_false_alert_segments_per_episode": 4,
            "maximum_total_safe_false_alert_segments": 10,
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
        "schema": "dtr-carla-c21-x43-weather-transfer-contract-v1",
        "parent_development_cohort_id": c20.COHORT_ID,
        "parent_protocol_sha256": C20_PROTOCOL_SHA256,
        "new_capture_seed": CAPTURE_SEED,
        "fresh_pixels": True,
        "changed_weather_by_layout": WEATHERS,
        "target_blueprints_by_layout": c20.TARGET_TEMPLATES,
        "preserved_admitted_motion_route_occlusion_geometry": True,
        "model_sensor_replay_projection": "ACTUAL_ALL_ACTORS",
        "witness_replay_projection": WITNESS_REPLAY_PROJECTION,
        "witness_non_authoritative_actual_pose_is_score_irrelevant": True,
        "critical_contact_outcome_truth_must_match_across_all_sensors": True,
        "prior_pixels_reused": False,
        "future_truth_visible_to_predictor": False,
        "single_scored_invocation": True,
        "development_evidence_only": {
            "c20_x43_summary_sha256": C20_X43_SUMMARY_SHA256,
            "c20_transport_adjudication_sha256": C20_TRANSPORT_ADJUDICATION_SHA256,
            "status": "C20_POSTHOC_DEVELOPMENT_ONLY",
        },
    }
    protocol["claim_boundary"] = [
        "C21 is frozen before capture and uses no C20 pixels.",
        "C21 tests weather and fresh-render transfer only; motion, route, occlusion, map, and target geometry remain inherited.",
        "Wearable RGB and metric depth retain exact all-actor actual-state replay against the instance evaluator.",
        "Witness is a diagnostic-only view: actual pose equality is required for scripted-authority actors, while non-authoritative actors are compared by their exact scripted commands and critical contact outcomes.",
        "C21 cannot establish geometric or map transfer, deployment, product, user-benefit, reliability, or safety authority.",
        "A failed source gate is NOT_EVALUABLE; X43 cannot be changed or rerun on C21 after pixels are opened.",
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
        here / "dtr_carla_x43_authority_preserving_credential_belief.py": (
            X43_PREDICTOR_SHA256
        ),
    }
    for path, expected in frozen_files.items():
        if c2.sha256_file(path) != expected:
            raise RuntimeError(f"frozen C21 component drift: {path.name}")
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
                "status": "C21_X43_PROTOCOL_STATIC_VALID_PREREGISTERED",
                "output": str(output),
                "sha256": c2.sha256_file(output),
                "cohort_id": COHORT_ID,
                "capture_seed": CAPTURE_SEED,
                "x43_sha256": X43_PREDICTOR_SHA256,
                "join_sha256": JOIN_SCRIPT_SHA256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
