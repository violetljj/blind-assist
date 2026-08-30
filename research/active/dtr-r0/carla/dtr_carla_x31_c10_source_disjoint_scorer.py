"""Score the frozen one-shot C10 opaque single-pass Development cohort.

Static validation opens only frozen code and protocol files.  It does not open
fresh source, predictions, evaluator truth, or a model result.  C9 instance
source outcome metadata was consumed only to design reachable C10 geometry;
no C9 model/evaluator/scorer/X24/X31 outcome existed or is used.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any, Mapping


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import dtr_carla_x25_rigid_footprint_scorer as base  # noqa: E402
import dtr_carla_x31_source_disjoint_scorer as shared  # noqa: E402
import materialize_dtr_carla_c10_x31_opaque_single_pass_protocol as c10  # noqa: E402


EXPECTED_PROTOCOL_SCHEMA = 10
EXPECTED_COHORT_ID = "DTR_CARLA_C10_X31_OPAQUE_SINGLE_PASS_SOURCE_DISJOINT_V1"
EXPECTED_PROTOCOL_SHA256 = (
    "255402227C86133B95706F7B49957F9AB463A146EDC3D266853381794BC5078B"
)
EXPECTED_CAPTURE_SEED = 130365
EXPECTED_X31_PREDICTOR_SHA256 = (
    "28E777021FC6AA39129480B069B544B562F559FD447CCE164706B6DCAC3A9F18"
)
EXPECTED_SHARED_C8_SCORER_SHA256 = (
    "8EB74AE938DA1E9912A6D91EC6BD8D529A10108CF6F94C429C48BC69948B7B3B"
)
EPISODES = shared.EPISODES
CONTACT_EPISODES = shared.CONTACT_EPISODES
SAFE_EPISODES = shared.SAFE_EPISODES
LAYOUTS = shared.LAYOUTS
SCORE_WINDOW_END_SECONDS = shared.SCORE_WINDOW_END_SECONDS
SAFE_SEGMENT_START_SECONDS = shared.SAFE_SEGMENT_START_SECONDS
FROZEN_GATES = shared.FROZEN_GATES
EPSILON = shared.EPSILON
ARM_X24 = shared.ARM_X24
ARM_X31 = shared.ARM_X31


def protocol_default_path() -> Path:
    return HERE / "dtr_carla_c10_x31_opaque_single_pass_protocol.json"


def validate_protocol(
    protocol: Mapping[str, Any], path: Path
) -> dict[str, Any]:
    base.require(
        base.sha256_file(path) == EXPECTED_PROTOCOL_SHA256,
        "c10_protocol_hash_not_frozen",
    )
    base.require(
        protocol.get("schema_version") == EXPECTED_PROTOCOL_SCHEMA,
        "c10_protocol_schema",
    )
    base.require(protocol.get("cohort_id") == EXPECTED_COHORT_ID, "c10_cohort")
    base.require(
        int(protocol["capture"]["seed"]) == EXPECTED_CAPTURE_SEED,
        "c10_capture_seed",
    )
    base.require(
        protocol.get("experiment_id")
        == "DTR_CARLA_C2_RICH_MULTILAYOUT_OCCLUSION_SOURCE_V2",
        "c10_c2_identity",
    )
    base.require(protocol["environment"]["map"] == shared.EXPECTED_MAP, "c10_map")
    base.require(tuple(protocol["layouts"]) == LAYOUTS, "c10_layout_order")
    base.require(
        {key: value["weather"] for key, value in protocol["layouts"].items()}
        == shared.EXPECTED_WEATHER_BY_LAYOUT,
        "c10_weather",
    )
    scenarios = {value["episode_id"]: value for value in protocol["scenarios"]}
    base.require(tuple(scenarios) == EPISODES, "c10_episode_order")
    base.require(
        tuple(key for key in EPISODES if scenarios[key]["expected_outcome"] == "CONTACT")
        == CONTACT_EPISODES,
        "c10_contacts",
    )
    base.require(
        tuple(key for key in EPISODES if scenarios[key]["expected_outcome"] == "SAFE")
        == SAFE_EPISODES,
        "c10_safe",
    )
    evaluation = protocol["evaluation_contract"]
    base.require(
        float(evaluation["truth_tail_seconds"]) == shared.TRUTH_TAIL_SECONDS,
        "c10_truth_tail",
    )
    base.require(
        evaluation["score_window_end_seconds"] == SCORE_WINDOW_END_SECONDS,
        "c10_score_windows",
    )
    base.require(
        evaluation["safe_segment_start_seconds"] == SAFE_SEGMENT_START_SECONDS,
        "c10_safe_boundaries",
    )
    base.require(
        evaluation["frozen_thresholds"] == shared.EXPECTED_PROTOCOL_THRESHOLDS,
        "c10_frozen_thresholds",
    )
    base.require(
        evaluation["transport_cone_falsification"]
        == shared.EXPECTED_TRANSPORT_FALSIFICATION,
        "c10_transport_falsification",
    )
    base.require(
        evaluation["dynamic_pairs"] == shared.expected_dynamic_pairs(),
        "c10_dynamic_pairs",
    )
    base.require(len(protocol["occlusion_contracts"]) == len(EPISODES), "c10_occlusions")
    base.require(protocol["twin_contracts"] == [], "c10_twin_contracts")
    for episode_id, contract in zip(EPISODES, protocol["occlusion_contracts"]):
        base.require(contract["episodes"] == [episode_id], f"c10_occlusion_singleton:{episode_id}")

    parent_path = c10.DEFAULT_BASE.resolve(strict=True)
    base.require(
        base.sha256_file(parent_path) == c10.PARENT_PROTOCOL_SHA256,
        "c10_parent_protocol_hash",
    )
    static_receipt = c10.validate_c10(protocol, c10.read_json(parent_path))
    source = protocol["source_disjoint_contract"]
    base.require(source["predecessor_c9_disposition"] == c10.PREDECESSOR_DISPOSITION, "c10_c9_disposition")
    base.require(source["predecessor_instance_source_outcome_accessed_for_geometry_reachability"] is True, "c10_c9_source_access_not_recorded")
    base.require(source["predecessor_instance_jsonl_and_manifests_accessed_for_geometry_reachability"] is True, "c10_c9_jsonl_access_not_recorded")
    base.require(source["predecessor_model_evaluator_scorer_x24_or_x31_outcome_existed"] is False, "c10_c9_algorithm_outcome_existed")
    base.require(source["predecessor_model_evaluator_scorer_x24_or_x31_outcome_accessed"] is False, "c10_c9_algorithm_outcome_accessed")
    base.require(source["development_only_never_confirmation"] is True, "c10_confirmation_leak")
    base.require(source["one_fresh_capture_only"] is True and source["resume_replay_reseed_substitution_or_partial_selection_allowed"] is False, "c10_one_shot_policy")
    return static_receipt


def validate_static(protocol_path: Path) -> dict[str, Any]:
    protocol_path = protocol_path.resolve(strict=True)
    protocol = base.read_json(protocol_path)
    static_receipt = validate_protocol(protocol, protocol_path)
    predictor_path = shared.predictor_path().resolve(strict=True)
    shared.validate_predictor_source(predictor_path)
    base.require(
        base.sha256_file(predictor_path) == EXPECTED_X31_PREDICTOR_SHA256,
        "c10_x31_predictor_hash",
    )
    shared_path = Path(shared.__file__).resolve(strict=True)
    base.require(
        base.sha256_file(shared_path) == EXPECTED_SHARED_C8_SCORER_SHA256,
        "c10_shared_scorer_utility_hash",
    )
    base.require(
        FROZEN_GATES["dynamic_contact_risk_authority"] == "RIGID_DYNAMIC_ONLY"
        and {
            key: FROZEN_GATES[key]
            for key in shared.EXPECTED_PROTOCOL_THRESHOLDS
        }
        == shared.EXPECTED_PROTOCOL_THRESHOLDS,
        "c10_scorer_gate_constants",
    )
    return {
        "status": "C10_X31_SCORER_STATIC_VALID",
        "opened_fresh_source_predictions_evaluator_or_model_result": False,
        "protocol_sha256": base.sha256_file(protocol_path),
        "predictor_sha256": base.sha256_file(predictor_path),
        "shared_c8_scorer_utility_sha256": base.sha256_file(shared_path),
        "scorer_sha256": base.sha256_file(Path(__file__).resolve()),
        "episodes": len(EPISODES),
        "singleton_physical_occlusion_contracts": len(protocol["occlusion_contracts"]),
        "predecessor_c9_status": "SOURCE_NOT_EVALUABLE",
        "predecessor_c9_run_id": c10.C9_RUN_ID,
        "predecessor_c9_instance_result_sha256": c10.C9_INSTANCE_RESULT_SHA256,
        "predecessor_c9_instance_source_outcome_accessed_for_geometry_design": True,
        "predecessor_c9_model_evaluator_scorer_x24_or_x31_outcome_existed": False,
        "predecessor_c9_model_evaluator_scorer_x24_or_x31_outcome_accessed": False,
        "development_only_never_confirmation": True,
        "one_fresh_capture_only": True,
        "static_launch_falsifier": static_receipt,
        "frozen_gates": FROZEN_GATES,
    }


def render_svg(result: Mapping[str, Any]) -> str:
    aggregate = result["aggregate"]
    decision = html.escape(str(result["decision"]))
    return "\n".join(
        [
            '<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="280" viewBox="0 0 1120 280">',
            '<rect width="1120" height="280" fill="#f8fafc"/>',
            '<text x="30" y="45" font-size="25" font-family="sans-serif" fill="#0f172a">C10 X31 opaque single-pass Development score</text>',
            f'<text x="30" y="82" font-size="14" font-family="monospace" fill="#334155">{decision}</text>',
            f'<text x="30" y="122" font-size="17" fill="#111827">X31 precision={aggregate[ARM_X31]["precision"]:.3f} F1={aggregate[ARM_X31]["f1"]:.3f}; X24 F1={aggregate[ARM_X24]["f1"]:.3f}</text>',
            f'<text x="30" y="162" font-size="15" fill="#111827">ambiguity selected-loss frames={result["transport_ambiguity"]["selected_contact_loss_ambiguity_frames"]}; standardized opaque single-pass occluders</text>',
            '<text x="30" y="202" font-size="15" fill="#111827">C9 predecessor: SOURCE_NOT_EVALUABLE; source geometry diagnostic only, no algorithm outcome.</text>',
            '<text x="30" y="244" font-size="13" fill="#475569">One fresh scripted-CARLA Development capture only; never confirmation, deployment, product, or safety authority.</text>',
            "</svg>",
        ]
    )


def score(args: argparse.Namespace) -> dict[str, Any]:
    protocol_path = args.protocol.resolve(strict=True)
    source_root = args.source_root.resolve(strict=True)
    run_root = args.run_root.resolve(strict=True)
    result_path = run_root / "result-x31.json"
    svg_path = run_root / "result-x31.svg"
    base.require(not result_path.exists() and not svg_path.exists(), "c10_x31_score_outputs_exist")
    protocol = base.read_json(protocol_path)
    validate_protocol(protocol, protocol_path)
    predictor_path = shared.predictor_path().resolve(strict=True)
    shared.validate_predictor_source(predictor_path)
    base.require(base.sha256_file(predictor_path) == EXPECTED_X31_PREDICTOR_SHA256, "c10_x31_predictor_hash")
    base.require(base.sha256_file(Path(shared.__file__).resolve(strict=True)) == EXPECTED_SHARED_C8_SCORER_SHA256, "c10_shared_scorer_utility_hash")
    x24 = base.read_json(run_root / "predictions-x24.json")
    x31 = base.read_json(run_root / "predictions-x31.json")
    freeze_x24 = base.read_json(run_root / "freeze-x24.json")
    freeze_x31 = base.read_json(run_root / "freeze-x31.json")
    shared.validate_prediction_envelopes(x24, x31, freeze_x24, freeze_x31, run_root)

    # Evaluator-bearing files open only after all frozen code/protocol identities.
    source_result = base.read_json(source_root / "result.json")
    base.require(source_result.get("status") == shared.SOURCE_COMPLETE_STATUS, "source_incomplete")
    base.require(bool(source_result.get("checks")) and all(bool(value) for value in source_result["checks"].values()), "source_gate_failed")
    base.require(source_result["protocol_sha256"] == EXPECTED_PROTOCOL_SHA256, "protocol_source_drift")
    base.require(int(source_result["episode_count"]) == len(EPISODES) and int(source_result["layout_count"]) == len(LAYOUTS), "source_cohort_count")
    model_manifest_path = source_root / "model" / "manifest.json"
    base.require(base.sha256_file(model_manifest_path) == freeze_x24["model_manifest"]["sha256"] == freeze_x31["source"]["model_manifest_sha256"], "source_model_manifest_drift")

    score_end = dict(SCORE_WINDOW_END_SECONDS)
    horizon_s = float(protocol["route_contract"]["future_horizon_seconds"])
    evaluator_full = {
        episode_id: base.read_jsonl(source_root / "evaluator" / "episodes" / episode_id / "frames.jsonl")
        for episode_id in EPISODES
    }
    predictions_full = {
        ARM_X24: {episode_id: shared.c7.arm_frames_full(x24, episode_id, ARM_X24) for episode_id in EPISODES},
        ARM_X31: {episode_id: shared.c7.arm_frames_full(x31, episode_id, ARM_X31) for episode_id in EPISODES},
    }
    for arm, episodes in predictions_full.items():
        for episode_id in EPISODES:
            base.align(evaluator_full[episode_id], episodes[episode_id], f"{arm}:{episode_id}")
    truth_tail_checks = {
        episode_id: float(rows[-1]["time_s"]) + EPSILON >= score_end[episode_id] + horizon_s
        for episode_id, rows in evaluator_full.items()
    }
    base.require(all(truth_tail_checks.values()), "right_censored_score_window")
    evaluator = {episode_id: shared.c7.prefix(rows, score_end[episode_id]) for episode_id, rows in evaluator_full.items()}
    predictions = {
        arm: {episode_id: shared.c7.prefix(rows, score_end[episode_id]) for episode_id, rows in episodes.items()}
        for arm, episodes in predictions_full.items()
    }
    aggregate = {arm: base.confusion(evaluator, episodes) for arm, episodes in predictions.items()}
    contacts = {
        episode_id: {arm: base.contact_metrics(evaluator[episode_id], episodes[episode_id]) for arm, episodes in predictions.items()}
        for episode_id in CONTACT_EPISODES
    }
    safe = {
        episode_id: {arm: base.false_segments(episodes[episode_id], SAFE_SEGMENT_START_SECONDS[episode_id]) for arm, episodes in predictions.items()}
        for episode_id in SAFE_EPISODES
    }
    occlusion_path = source_root / "evaluator" / "physical_occlusion_report.json"
    selected_occlusions = shared.validate_occlusion_reports(protocol, base.read_json(occlusion_path), evaluator_full)
    continuity, ambiguity = shared.contact_transport_continuity(x31, selected_occlusions)
    invariants = shared.authority_invariants(x31, score_end)
    safe_segments = sum(safe[key][ARM_X31]["false_alert_segment_count"] for key in SAFE_EPISODES)
    dynamic_authority = invariants["dynamic_contacts"]
    gate_checks = {
        "all_scored_frames_have_full_realized_future": all(truth_tail_checks.values()),
        "all_eight_singleton_physical_occlusion_contracts_valid": len(selected_occlusions) == len(EPISODES),
        "x31_detects_all_four_dynamic_contacts": all(contacts[key][ARM_X31]["event_detected_before_contact"] for key in CONTACT_EPISODES),
        "each_dynamic_contact_has_at_least_2s_lead": all(contacts[key][ARM_X31]["first_alert_lead_seconds"] is not None and float(contacts[key][ARM_X31]["first_alert_lead_seconds"]) + EPSILON >= FROZEN_GATES["minimum_dynamic_contact_lead_seconds"] for key in CONTACT_EPISODES),
        "each_dynamic_contact_future_positive_recall_at_least_0_80": all(contacts[key][ARM_X31]["future_positive_recall"] + EPSILON >= FROZEN_GATES["minimum_dynamic_contact_future_positive_recall"] for key in CONTACT_EPISODES),
        "x31_has_zero_safe_risk_segments_from_0s": safe_segments == 0,
        "x31_aggregate_precision_at_least_0_95": aggregate[ARM_X31]["precision"] + EPSILON >= FROZEN_GATES["minimum_aggregate_precision"],
        "x31_aggregate_f1_at_least_0_80": aggregate[ARM_X31]["f1"] + EPSILON >= FROZEN_GATES["minimum_aggregate_f1"],
        "x31_frame_f1_strictly_exceeds_x24": aggregate[ARM_X31]["f1"] > aggregate[ARM_X24]["f1"] + EPSILON,
        "x31_route_risk_continuous_through_each_contact_loss_and_first_reappearance": all(value["continuous_route_risk"] for value in continuity.values()),
        "confirmed_parent_ancestry_preserved_when_diagnosable": all(value["parent_ancestry_status"] != "BROKEN" for value in continuity.values()),
        "x31_transport_ambiguity_materially_exercised": ambiguity["materially_exercised"],
        "every_dynamic_contact_risk_is_exclusively_rigid_dynamic": all(dynamic_authority[key]["rigid_dynamic_confirmed_route_risk_frames"] > 0 and dynamic_authority[key]["route_risk_without_confirmed_rigid_dynamic_frames"] == 0 and dynamic_authority[key]["confirmed_non_rigid_risk_track_references"] == 0 and dynamic_authority[key]["confirmed_missing_track_references"] == 0 for key in CONTACT_EPISODES),
        "ego_and_unauthorized_tracks_never_enter_risk": invariants["risk_ineligible_authority_frames"] == 0,
        "non_dynamic_authorities_have_zero_velocity": invariants["non_dynamic_nonzero_velocity_frames"] == 0,
        "hold_never_promotes_motion_authority": invariants["promotion_during_hold_frames"] == 0,
        "all_motion_authorities_are_known": invariants["unknown_authority_frames"] == 0,
        "risk_eligible_track_counts_are_consistent": invariants["risk_eligible_track_count_mismatches"] == 0,
        "route_risk_has_confirmed_eligible_rigid_dynamic_track": invariants["route_risk_without_confirmed_eligible_track_frames"] == 0 and invariants["route_risk_without_confirmed_rigid_dynamic_frames"] == 0,
        "confirmed_parent_identity_matches_track_ancestry": invariants["confirmed_parent_identity_mismatches"] == 0,
    }
    gate_met = all(gate_checks.values())
    result = {
        "schema": "blindassist-dtr-carla-c10-x31-source-disjoint-score-result-v1",
        "status": "COMPLETE",
        "decision": "DTR_CARLA_X31_OPAQUE_SINGLE_PASS_SOURCE_DISJOINT_PRETRUTH_DEVELOPMENT_GATE_MET" if gate_met else "DTR_CARLA_X31_OPAQUE_SINGLE_PASS_SOURCE_DISJOINT_PRETRUTH_DEVELOPMENT_GATE_NOT_MET",
        "gate_met": gate_met,
        "gate_checks": gate_checks,
        "thresholds": FROZEN_GATES,
        "aggregate": aggregate,
        "contacts": contacts,
        "safe": safe,
        "physical_occlusion_selected_indices": selected_occlusions,
        "transport_continuity": continuity,
        "transport_ambiguity": ambiguity,
        "authority_invariants": invariants,
        "truth_tail_checks": truth_tail_checks,
        "score_window_end_seconds": score_end,
        "source": {
            "source_result_sha256": base.sha256_file(source_root / "result.json"),
            "physical_occlusion_report_sha256": base.sha256_file(occlusion_path),
            "protocol_sha256": base.sha256_file(protocol_path),
            "x24_predictions_sha256": base.sha256_file(run_root / "predictions-x24.json"),
            "x31_predictions_sha256": base.sha256_file(run_root / "predictions-x31.json"),
            "x24_freeze_sha256": base.sha256_file(run_root / "freeze-x24.json"),
            "x31_freeze_sha256": base.sha256_file(run_root / "freeze-x31.json"),
            "x31_predictor_sha256": base.sha256_file(predictor_path),
            "shared_c8_scorer_utility_sha256": base.sha256_file(Path(shared.__file__).resolve(strict=True)),
            "scorer_sha256": base.sha256_file(Path(__file__).resolve()),
        },
        "claim_boundary": {
            "fresh_scripted_carla_source_disjoint_pretruth_development": True,
            "development_only_never_confirmation": True,
            "one_fresh_capture_only": True,
            "standardized_opaque_single_pass_physical_occluders": True,
            "scripted_pose_collision_response_decoupled": True,
            "c9_predecessor_status": "SOURCE_NOT_EVALUABLE",
            "c9_instance_source_outcome_accessed_for_geometry_design": True,
            "c9_model_evaluator_scorer_x24_or_x31_outcome_existed": False,
            "c9_model_evaluator_scorer_x24_or_x31_outcome_accessed": False,
            "c9_resume_replay_reseed_substitution_or_partial_selection_authorized": False,
            "full_horizon_truth_tail": True,
            "frozen_x31_predictor_sha256": EXPECTED_X31_PREDICTOR_SHA256,
            "real_world_confirmation": False,
            "product_default_authority": False,
            "deployment_or_safety_authority": False,
        },
    }
    base.write_json_exclusive(result_path, result)
    base.write_exclusive(svg_path, render_svg(result).encode("utf-8"))
    return {**result, "result_sha256": base.sha256_file(result_path), "svg_sha256": base.sha256_file(svg_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-static", action="store_true")
    parser.add_argument("--protocol", type=Path, default=protocol_default_path())
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--run-root", type=Path)
    args = parser.parse_args()
    if not args.validate_static and (args.source_root is None or args.run_root is None):
        parser.error("--source-root and --run-root are required unless --validate-static is used")
    return args


def main() -> int:
    args = parse_args()
    result = validate_static(args.protocol) if args.validate_static else score(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
