"""Score the frozen C8 X24/X31 transport-cone Development cohort.

The static validation path opens only this scorer, the frozen C8 protocol, and
the X31 predictor source.  Normal scoring opens evaluator truth only after the
protocol, predictor, prediction envelopes, and freeze identities validate.
"""

from __future__ import annotations

import argparse
import ast
import html
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import dtr_carla_x25_rigid_footprint_scorer as base  # noqa: E402
import dtr_carla_x30_source_disjoint_scorer as c7  # noqa: E402


X24_SCHEMA = "blindassist-dtr-carla-x23-x24-predictions-v1"
X24_FREEZE_SCHEMA = "blindassist-dtr-carla-x24-freeze-v1"
X31_SCHEMA = (
    "blindassist-dtr-carla-x31-ambiguity-preserving-transport-predictions-v1"
)
X31_FREEZE_SCHEMA = (
    "blindassist-dtr-carla-x31-ambiguity-preserving-transport-freeze-v1"
)
X31_EXPERIMENT_ID = "DTR_CARLA_X31_AMBIGUITY_PRESERVING_SURFACE_TRANSPORT"
ARM_X24 = "X24_ISSUED_PLAN_ADHERENCE"
ARM_X31 = "X31_ISSUED_PLAN_SET_VALUED_SURFACE_TRANSPORT_ANCESTRY"

EXPECTED_PROTOCOL_SCHEMA = 8
EXPECTED_COHORT_ID = "DTR_CARLA_C8_X31_TRANSPORT_CONE_SOURCE_DISJOINT_V1"
EXPECTED_PROTOCOL_SHA256 = (
    "180A9FF243FB6D07999AD936481F77429BE2759B0A9038D13E6B9821BC35ACF0"
)
EXPECTED_X31_PREDICTOR_SHA256 = (
    "28E777021FC6AA39129480B069B544B562F559FD447CCE164706B6DCAC3A9F18"
)
EXPECTED_PARENT_COHORT_ID = "DTR_CARLA_C7_X30_SOURCE_DISJOINT_CONFIRMATION_V1"
EXPECTED_PARENT_PROTOCOL_CANONICAL_SHA256 = (
    "106676620833E9FC45BD14AE6F886781C71D6E8D4E41F1FD8838CEDF37DECB7B"
)
EXPECTED_MAP = "Carla/Maps/Town10HD_Opt"
EXPECTED_CAPTURE_SEED = 130363
SOURCE_COMPLETE_STATUS = "DTR_CARLA_C2_RICH_MULTILAYOUT_SOURCE_COMPLETE"
PREDICTION_STATUS = "SEALED_TRUTH_BLIND_PENDING_SCORE"
FREEZE_STATUS = "FROZEN_TRUTH_BLIND_PENDING_PREDICTION"

EPISODES = tuple(f"ep_{value:02d}" for value in range(1, 9))
CONTACT_EPISODES = ("ep_01", "ep_03", "ep_05", "ep_07")
SAFE_EPISODES = ("ep_02", "ep_04", "ep_06", "ep_08")
DYNAMIC_CONTACT_EPISODES = CONTACT_EPISODES
DYNAMIC_SAFE_EPISODES = SAFE_EPISODES
LAYOUTS = ("c8_l01", "c8_l02", "c8_l03", "c8_l04")
EXPECTED_WEATHER_BY_LAYOUT = {
    "c8_l01": "HardRainSunset",
    "c8_l02": "WetSunset",
    "c8_l03": "MidRainyNoon",
    "c8_l04": "DustStorm",
}
EXPECTED_DYNAMIC_BLUEPRINTS = {
    "adult": "walker.pedestrian.0044",
    "sedan": "vehicle.chevrolet.impala",
    "van": "vehicle.volkswagen.t2",
    "emergency": "vehicle.dodge.charger_police",
}
SCORE_WINDOW_END_SECONDS = {episode_id: 6.0 for episode_id in EPISODES}
SAFE_SEGMENT_START_SECONDS = {episode_id: 0.0 for episode_id in SAFE_EPISODES}
TRUTH_TAIL_SECONDS = 3.0
FROZEN_GATES = {
    "minimum_dynamic_contact_lead_seconds": 2.0,
    "minimum_dynamic_contact_future_positive_recall": 0.80,
    "maximum_safe_risk_segments": 0,
    "minimum_aggregate_precision": 0.95,
    "minimum_aggregate_f1": 0.80,
    "x31_f1_must_strictly_exceed_x24": True,
    "dynamic_contact_risk_authority": "RIGID_DYNAMIC_ONLY",
}
EXPECTED_PROTOCOL_THRESHOLDS = {
    "minimum_dynamic_contact_lead_seconds": 2.0,
    "minimum_dynamic_contact_future_positive_recall": 0.80,
    "maximum_safe_risk_segments": 0,
    "minimum_aggregate_precision": 0.95,
    "minimum_aggregate_f1": 0.80,
    "x31_f1_must_strictly_exceed_x24": True,
}
EXPECTED_TRANSPORT_FALSIFICATION = {
    "contact_pre_loss_future_positive_risk_required": True,
    "zero_future_positive_risk_gaps_during_full_loss": True,
    "first_reappearance_future_positive_risk_required": True,
    "post_loss_risk_branch_must_descend_from_pre_loss_authorized_branch": True,
    "single_conflicting_shift_must_not_revoke_compatible_branch": True,
    "safe_alias_confirmed_risk_frames_maximum": 0,
}
ALLOWED_MOTION_AUTHORITIES = {
    "STATIC_SCENE",
    "RIGID_DYNAMIC",
    "EGO_CARRIED",
    "UNAUTHORIZED_MOTION",
}
RISK_INELIGIBLE_AUTHORITIES = {"EGO_CARRIED", "UNAUTHORIZED_MOTION"}
EPSILON = 1e-9


def expected_dynamic_pairs() -> list[dict[str, Any]]:
    values = [
        ("adult", "ep_01", "ep_02", "c8_l01", "walker.pedestrian.0044", 1.3, 2.4, 4.15),
        ("sedan", "ep_03", "ep_04", "c8_l02", "vehicle.chevrolet.impala", 1.3, 2.4, 3.87),
        ("van", "ep_05", "ep_06", "c8_l03", "vehicle.volkswagen.t2", 1.2, 2.3, 3.70),
        ("emergency", "ep_07", "ep_08", "c8_l04", "vehicle.dodge.charger_police", 1.5, 2.5, 3.97),
    ]
    output: list[dict[str, Any]] = []
    for index, (family, contact, safe, layout, blueprint, start, end, contact_s) in enumerate(values, 1):
        output.append(
            {
                "family": family,
                "contact_episode": contact,
                "safe_episode": safe,
                "layout_id": layout,
                "target_asset": f"c8_l0{index}_target",
                "alias_asset": f"c8_l0{index}_alias",
                "occluder_asset": f"c8_l0{index}_occluder",
                "target_blueprint": blueprint,
                "pair_difference": "TARGET_INITIAL_LATERAL_POSITION_ONLY",
                "target_velocity_segments_identical": True,
                "causally_distinguishable_from_s": 0.0,
                "planned_occlusion_window_s": [start, end],
                "expected_contact_time_s": contact_s,
            }
        )
    return output


def predictor_path() -> Path:
    return HERE / "dtr_carla_x31_ambiguity_preserving_transport_predictor.py"


def protocol_default_path() -> Path:
    return HERE / "dtr_carla_c8_x31_transport_cone_protocol.json"


def _source_assignments(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    output: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        for target in targets:
            if isinstance(target, ast.Name):
                try:
                    output[target.id] = ast.literal_eval(value)
                except (ValueError, TypeError):
                    pass
    return output


def validate_protocol(protocol: Mapping[str, Any], path: Path) -> None:
    base.require(base.sha256_file(path) == EXPECTED_PROTOCOL_SHA256, "c8_protocol_hash_not_frozen")
    base.require(protocol.get("schema_version") == EXPECTED_PROTOCOL_SCHEMA, "c8_protocol_schema")
    base.require(protocol.get("cohort_id") == EXPECTED_COHORT_ID, "c8_cohort")
    base.require(protocol.get("experiment_id") == "DTR_CARLA_C2_RICH_MULTILAYOUT_OCCLUSION_SOURCE_V2", "c8_c2_identity")
    base.require(protocol["environment"]["map"] == EXPECTED_MAP, "c8_map")
    base.require(float(protocol["environment"]["sample_seconds"]) == 0.1, "c8_sample_period")
    base.require(int(protocol["capture"]["seed"]) == EXPECTED_CAPTURE_SEED, "c8_capture_seed")
    layouts = protocol["layouts"]
    base.require(tuple(layouts) == LAYOUTS, "c8_layout_order")
    base.require({key: value["weather"] for key, value in layouts.items()} == EXPECTED_WEATHER_BY_LAYOUT, "c8_weather")
    base.require(all(float(value["duration_seconds"]) == 9.0 for value in layouts.values()), "c8_duration")

    scenarios = {value["episode_id"]: value for value in protocol["scenarios"]}
    base.require(tuple(scenarios) == EPISODES, "c8_episode_order")
    base.require(tuple(key for key in EPISODES if scenarios[key]["expected_outcome"] == "CONTACT") == CONTACT_EPISODES, "c8_contacts")
    base.require(tuple(key for key in EPISODES if scenarios[key]["expected_outcome"] == "SAFE") == SAFE_EPISODES, "c8_safe")

    contract = protocol["evaluation_contract"]
    base.require(float(contract["truth_tail_seconds"]) == TRUTH_TAIL_SECONDS, "c8_truth_tail")
    base.require(contract["future_truth_rule"] == "SCORE_ONLY_FRAMES_WITH_FULL_CAPTURED_HORIZON", "c8_future_truth_rule")
    base.require(contract["score_window_end_seconds"] == SCORE_WINDOW_END_SECONDS, "c8_score_windows")
    base.require(tuple(contract["contact_episodes"]) == CONTACT_EPISODES, "c8_contact_contract")
    base.require(tuple(contract["safe_episodes"]) == SAFE_EPISODES, "c8_safe_contract")
    base.require(tuple(contract["fresh_dynamic_contact_episodes"]) == DYNAMIC_CONTACT_EPISODES, "c8_dynamic_contacts")
    base.require(tuple(contract["fresh_dynamic_safe_episodes"]) == DYNAMIC_SAFE_EPISODES, "c8_dynamic_safe")
    base.require(contract["retained_occlusion_episodes"] == [], "c8_retained_outcomes")
    base.require(contract["safe_segment_start_seconds"] == SAFE_SEGMENT_START_SECONDS, "c8_safe_boundaries")
    base.require(contract["dynamic_pairs"] == expected_dynamic_pairs(), "c8_dynamic_pairs")
    base.require(contract["frozen_thresholds"] == EXPECTED_PROTOCOL_THRESHOLDS, "c8_frozen_thresholds")
    base.require(contract["transport_cone_falsification"] == EXPECTED_TRANSPORT_FALSIFICATION, "c8_transport_falsification")
    base.require(contract.get("all_physical_obstacles_are_truth_relevant") is True, "c8_truth_relevance")

    source_disjoint = protocol["source_disjoint_contract"]
    base.require(source_disjoint["parent_cohort_id"] == EXPECTED_PARENT_COHORT_ID, "c8_parent_cohort")
    base.require(source_disjoint["parent_protocol_canonical_sha256"] == EXPECTED_PARENT_PROTOCOL_CANONICAL_SHA256, "c8_parent_protocol_hash")
    base.require(tuple(source_disjoint["confirmation_episodes"]) == EPISODES, "c8_confirmation_stratum")
    base.require(source_disjoint["retained_regression_episodes"] == [], "c8_regression_stratum")
    base.require(source_disjoint["map"] == EXPECTED_MAP and source_disjoint["map_disjoint_from_prior_protocols"] is False, "c8_same_map_boundary")
    base.require(source_disjoint["same_map_numeric_anchor_and_witness_reuse"] is True, "c8_anchor_boundary")
    base.require(int(source_disjoint["new_capture_seed"]) == EXPECTED_CAPTURE_SEED, "c8_disjoint_seed")
    base.require(source_disjoint["weather_by_layout"] == EXPECTED_WEATHER_BY_LAYOUT, "c8_disjoint_weather")
    base.require(source_disjoint["new_dynamic_target_blueprints"] == EXPECTED_DYNAMIC_BLUEPRINTS, "c8_disjoint_blueprints")
    base.require(source_disjoint["dynamic_pairs"] == expected_dynamic_pairs(), "c8_disjoint_pairs")
    base.require(source_disjoint["prior_outcomes_or_evidence_referenced"] is False, "c8_prior_outcomes")

    compatibility = protocol["capture_compatibility_contract"]
    base.require(compatibility["capture_ready_with_current_generic_runner"] is True, "c8_capture_ready")
    base.require(compatibility["requested_server_map"] == EXPECTED_MAP, "c8_capture_map")
    base.require(protocol["twin_contracts"] == [], "c8_twin_contracts")
    occlusions = protocol["occlusion_contracts"]
    base.require(len(occlusions) == len(EPISODES), "c8_occlusion_contract_count")
    for episode_id, value in zip(EPISODES, occlusions):
        base.require(value["contract_id"] == f"c8_transport_cone_loss_{episode_id}", f"c8_occlusion_id:{episode_id}")
        base.require(value["episodes"] == [episode_id], f"c8_occlusion_singleton:{episode_id}")
        base.require(value["required_outcomes"] == {episode_id: scenarios[episode_id]["expected_outcome"]}, f"c8_occlusion_outcome:{episode_id}")

    horizon_s = float(protocol["route_contract"]["future_horizon_seconds"])
    base.require(TRUTH_TAIL_SECONDS + EPSILON >= horizon_s, "c8_truth_tail_shorter_than_horizon")
    for episode_id, scenario in scenarios.items():
        duration_s = float(layouts[scenario["layout_id"]]["duration_seconds"])
        base.require(duration_s + EPSILON >= SCORE_WINDOW_END_SECONDS[episode_id] + TRUTH_TAIL_SECONDS, f"c8_declared_tail:{episode_id}")


def validate_predictor_source(path: Path) -> None:
    base.require(base.sha256_file(path) == EXPECTED_X31_PREDICTOR_SHA256, "x31_predictor_source_drift")
    constants = _source_assignments(path)
    base.require(constants.get("EXPERIMENT_ID") == X31_EXPERIMENT_ID, "x31_experiment_id")
    base.require(constants.get("FREEZE_SCHEMA") == X31_FREEZE_SCHEMA, "x31_freeze_schema_constant")
    base.require(constants.get("PREDICTION_SCHEMA") == X31_SCHEMA, "x31_prediction_schema_constant")
    base.require(constants.get("ARM_X31") == ARM_X31, "x31_arm_constant")


def validate_static(protocol_path: Path) -> dict[str, Any]:
    protocol_path = protocol_path.resolve(strict=True)
    protocol = base.read_json(protocol_path)
    validate_protocol(protocol, protocol_path)
    x31_path = predictor_path().resolve(strict=True)
    validate_predictor_source(x31_path)
    base.require(
        {key: FROZEN_GATES[key] for key in EXPECTED_PROTOCOL_THRESHOLDS}
        == EXPECTED_PROTOCOL_THRESHOLDS
        and FROZEN_GATES["dynamic_contact_risk_authority"]
        == "RIGID_DYNAMIC_ONLY",
        "x31_scorer_gate_constants",
    )
    return {
        "status": "C8_X31_SCORER_STATIC_VALID",
        "opened_source_or_predictions": False,
        "protocol_sha256": base.sha256_file(protocol_path),
        "predictor_sha256": base.sha256_file(x31_path),
        "scorer_sha256": base.sha256_file(Path(__file__).resolve()),
        "episodes": len(EPISODES),
        "singleton_physical_occlusion_contracts": len(protocol["occlusion_contracts"]),
        "frozen_gates": FROZEN_GATES,
    }


def validate_prediction_envelopes(
    x24: Mapping[str, Any],
    x31: Mapping[str, Any],
    freeze_x24: Mapping[str, Any],
    freeze_x31: Mapping[str, Any],
    run_root: Path,
) -> None:
    base.require(x24.get("schema") == X24_SCHEMA and x31.get("schema") == X31_SCHEMA, "prediction_schema")
    base.require(x31.get("experiment_id") == X31_EXPERIMENT_ID, "x31_prediction_experiment")
    base.require(x24.get("status") == PREDICTION_STATUS and x31.get("status") == PREDICTION_STATUS, "prediction_status")
    base.require(x24.get("truth_blind") is True and x31.get("truth_blind") is True, "prediction_truth_blind")
    base.require(x24.get("claim_boundary", {}).get("evaluator_opened") is False and x31.get("claim_boundary", {}).get("evaluator_opened") is False, "prediction_evaluator_opened")
    base.require(x24.get("claim_boundary", {}).get("current_actor_oracle_used") is False and x31.get("claim_boundary", {}).get("current_actor_oracle_used") is False, "prediction_actor_oracle")
    base.require(ARM_X24 in x24["arms"] and x31["arms"] == [ARM_X31], "prediction_arms")
    base.require(tuple(x24["episodes"]) == EPISODES and tuple(x31["episodes"]) == EPISODES, "prediction_episodes")
    base.require(freeze_x24.get("schema") == X24_FREEZE_SCHEMA and freeze_x31.get("schema") == X31_FREEZE_SCHEMA, "prediction_freeze_schema")
    base.require(freeze_x31.get("experiment_id") == X31_EXPERIMENT_ID and freeze_x31.get("arm") == ARM_X31, "x31_freeze_identity")
    base.require(freeze_x24.get("status") == FREEZE_STATUS and freeze_x31.get("status") == FREEZE_STATUS, "prediction_freeze_status")
    base.require(freeze_x24.get("truth_blind") is True and freeze_x31.get("truth_blind") is True, "prediction_freeze_truth_blind")
    x24_predictions_hash = base.sha256_file(run_root / "predictions-x24.json")
    x24_freeze_hash = base.sha256_file(run_root / "freeze-x24.json")
    x31_freeze_hash = base.sha256_file(run_root / "freeze-x31.json")
    base.require(x24["source"]["freeze_sha256"] == x24_freeze_hash, "x24_prediction_freeze_drift")
    base.require(x31["source"]["freeze_sha256"] == x31_freeze_hash, "x31_prediction_freeze_drift")
    base.require(freeze_x31["source"]["x24_freeze_sha256"] == x24_freeze_hash, "x31_x24_freeze_drift")
    base.require(freeze_x31["source"]["x24_predictions_sha256"] == x24_predictions_hash and x31["source"]["x24_predictions_sha256"] == x24_predictions_hash, "x31_x24_prediction_drift")
    base.require(freeze_x31["algorithm_files"]["x31_predictor"]["sha256"] == EXPECTED_X31_PREDICTOR_SHA256, "x31_predictor_not_frozen")
    base.require(freeze_x31["fixed_constants"] == x31["fixed_constants"], "x31_constants_drift")
    base.require(freeze_x31["fixed_constants"].get("representation") == "AMBIGUITY_PRESERVING_SET_VALUED_SURFACE_TRANSPORT_ANCESTRY", "x31_representation")
    base.require(freeze_x31["fixed_constants"].get("detector_threshold_change") is False and freeze_x31["fixed_constants"].get("association_threshold_change") is False and freeze_x31["fixed_constants"].get("route_threshold_change") is False, "x31_threshold_change")
    base.require(int(freeze_x24["episodes"]) == len(EPISODES) and int(freeze_x31["episodes"]) == len(EPISODES), "prediction_freeze_episode_count")


def authority_invariants(predictions: Mapping[str, Any], score_end: Mapping[str, float]) -> dict[str, Any]:
    counts = {
        "risk_ineligible_authority_frames": 0,
        "non_dynamic_nonzero_velocity_frames": 0,
        "promotion_during_hold_frames": 0,
        "unknown_authority_frames": 0,
        "risk_eligible_track_count_mismatches": 0,
        "route_risk_without_confirmed_eligible_track_frames": 0,
        "route_risk_without_confirmed_rigid_dynamic_frames": 0,
        "confirmed_non_rigid_risk_track_references": 0,
        "confirmed_missing_track_references": 0,
        "confirmed_parent_identity_mismatches": 0,
    }
    dynamic = {
        key: {"route_risk_frames": 0, "rigid_dynamic_confirmed_route_risk_frames": 0, "route_risk_without_confirmed_rigid_dynamic_frames": 0, "confirmed_non_rigid_risk_track_references": 0, "confirmed_missing_track_references": 0}
        for key in DYNAMIC_CONTACT_EPISODES
    }
    for episode_id, episode in predictions["episodes"].items():
        previous_parent_authorities: dict[str, set[str]] = {}
        for frame in episode["frames"]:
            if float(frame["time_s"]) > float(score_end[episode_id]) + EPSILON:
                break
            tracks = {str(value["track_id"]): value for value in frame["tracks"]}
            eligible = {track_id for track_id, value in tracks.items() if bool(value["risk_eligible"])}
            counts["risk_eligible_track_count_mismatches"] += int(int(frame["risk_eligible_tracks"]) != len(eligible))
            current_parent_authorities: dict[str, set[str]] = {}
            for track_id, track in tracks.items():
                authority = str(track["motion_authority"])
                parent_id = str(track.get("parent_track_id") or track_id)
                current_parent_authorities.setdefault(parent_id, set()).add(authority)
                speed = abs(float(track["velocity_forward_mps"])) + abs(float(track["velocity_right_mps"]))
                counts["unknown_authority_frames"] += int(authority not in ALLOWED_MOTION_AUTHORITIES)
                counts["risk_ineligible_authority_frames"] += int(authority in RISK_INELIGIBLE_AUTHORITIES and bool(track["risk_eligible"]))
                counts["non_dynamic_nonzero_velocity_frames"] += int(authority != "RIGID_DYNAMIC" and speed > EPSILON)
                counts["promotion_during_hold_frames"] += int(
                    str(track["disposition"]) == "HOLD"
                    and authority == "RIGID_DYNAMIC"
                    and "RIGID_DYNAMIC"
                    not in previous_parent_authorities.get(parent_id, set())
                )
            previous_parent_authorities = current_parent_authorities
            arm = frame["arms"][ARM_X31]
            if not bool(arm["route_risk"]):
                continue
            confirmed = {str(value) for value in arm["confirmed_risk_track_ids"]}
            confirmed_eligible = confirmed & eligible
            missing = sum(track_id not in tracks for track_id in confirmed)
            non_rigid = sum(track_id in tracks and bool(tracks[track_id]["risk_eligible"]) and str(tracks[track_id]["motion_authority"]) != "RIGID_DYNAMIC" for track_id in confirmed)
            rigid = {track_id for track_id in confirmed_eligible if str(tracks[track_id]["motion_authority"]) == "RIGID_DYNAMIC"}
            derived_parents = {str(tracks[track_id].get("parent_track_id") or track_id) for track_id in rigid}
            declared_parents = {str(value) for value in arm.get("confirmed_risk_parent_track_ids", [])}
            counts["route_risk_without_confirmed_eligible_track_frames"] += int(not confirmed_eligible)
            counts["route_risk_without_confirmed_rigid_dynamic_frames"] += int(not rigid)
            counts["confirmed_non_rigid_risk_track_references"] += non_rigid
            counts["confirmed_missing_track_references"] += missing
            counts["confirmed_parent_identity_mismatches"] += int(derived_parents != declared_parents)
            if episode_id in dynamic:
                item = dynamic[episode_id]
                item["route_risk_frames"] += 1
                item["rigid_dynamic_confirmed_route_risk_frames"] += int(bool(rigid))
                item["route_risk_without_confirmed_rigid_dynamic_frames"] += int(not rigid)
                item["confirmed_non_rigid_risk_track_references"] += non_rigid
                item["confirmed_missing_track_references"] += missing
    return {**counts, "dynamic_contacts": dynamic}


def validate_occlusion_reports(
    protocol: Mapping[str, Any],
    reports: Sequence[Mapping[str, Any]],
    evaluator_full: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, list[int]]:
    expected = {value["contract_id"]: value for value in protocol["occlusion_contracts"]}
    actual = {str(value.get("contract_id")): value for value in reports}
    base.require(len(reports) == len(expected) and set(actual) == set(expected), "c8_occlusion_report_set")
    output: dict[str, list[int]] = {}
    for contract_id, contract in expected.items():
        episode_id = str(contract["episodes"][0])
        report = actual[contract_id]
        base.require(report.get("passed") is True, f"occlusion_contract_not_passed:{episode_id}")
        base.require(report.get("pair_occlusion_indices_identical") is True, f"occlusion_singleton_identity:{episode_id}")
        episode_report = report.get("episodes", {}).get(episode_id)
        base.require(isinstance(episode_report, Mapping) and episode_report.get("passed") is True, f"occlusion_episode_failed:{episode_id}")
        selected = episode_report.get("selected")
        base.require(isinstance(selected, Mapping) and selected.get("passed") is True, f"occlusion_selected:{episode_id}")
        indices = selected.get("sample_indices")
        base.require(isinstance(indices, list) and indices and all(type(value) is int for value in indices), f"occlusion_indices:{episode_id}")
        base.require(indices == report.get("selected_indices", {}).get(episode_id), f"occlusion_selected_identity:{episode_id}")
        base.require(indices == list(range(indices[0], indices[0] + len(indices))), f"occlusion_not_contiguous:{episode_id}")
        pre_track = selected.get("pre_track_sample_indices")
        post_track = selected.get("post_reappearance_sample_indices")
        base.require(
            isinstance(pre_track, list)
            and pre_track
            and int(pre_track[-1]) == indices[0] - 1,
            f"occlusion_pre_loss_boundary:{episode_id}",
        )
        base.require(
            isinstance(post_track, list)
            and post_track
            and int(post_track[0]) == indices[-1] + 1,
            f"occlusion_first_reappearance:{episode_id}",
        )
        available = {int(value["sample_index"]) for value in evaluator_full[episode_id]}
        base.require(set(indices).issubset(available), f"occlusion_index_missing:{episode_id}")
        output[episode_id] = list(indices)
    return output


def contact_transport_continuity(
    predictions: Mapping[str, Any], selected: Mapping[str, Sequence[int]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    continuity: dict[str, Any] = {}
    ambiguity_frames_total = 0
    ambiguity_selected_contact_frames = 0
    for episode_id in EPISODES:
        frames = predictions["episodes"][episode_id]["frames"]
        for frame in frames:
            if float(frame["time_s"]) <= SCORE_WINDOW_END_SECONDS[episode_id] + EPSILON:
                ambiguous = any(int(track.get("surface_transport_branch_count", 0)) > 1 for track in frame["tracks"])
                ambiguity_frames_total += int(ambiguous)
    for episode_id in CONTACT_EPISODES:
        frames = {int(value["sample_index"]): value for value in predictions["episodes"][episode_id]["frames"]}
        loss = list(selected[episode_id])
        required = [loss[0] - 1, *loss, loss[-1] + 1]
        base.require(all(index in frames for index in required), f"contact_continuity_frame_missing:{episode_id}")
        base.require(float(frames[required[-1]]["time_s"]) <= SCORE_WINDOW_END_SECONDS[episode_id] + EPSILON, f"contact_reappearance_outside_score_window:{episode_id}")
        route_risk = {str(index): bool(frames[index]["arms"][ARM_X31]["route_risk"]) for index in required}
        parent_sets: list[set[str]] = []
        for index in required:
            frame = frames[index]
            arm = frame["arms"][ARM_X31]
            confirmed_tracks = {
                str(value) for value in arm.get("confirmed_risk_track_ids", [])
            }
            confirmed_parents = {
                str(value)
                for value in arm.get("confirmed_risk_parent_track_ids", [])
            }
            confirmed_ambiguous = any(
                int(track.get("surface_transport_branch_count", 0)) > 1
                and (
                    str(track["track_id"]) in confirmed_tracks
                    or str(track.get("parent_track_id") or track["track_id"])
                    in confirmed_parents
                )
                for track in frame["tracks"]
            )
            if index in loss:
                ambiguity_selected_contact_frames += int(confirmed_ambiguous)
            parent_sets.append({str(value) for value in arm.get("confirmed_risk_parent_track_ids", [])})
        diagnosable = all(bool(value) for value in parent_sets)
        common = set.intersection(*parent_sets) if diagnosable else set()
        ancestry_status = "PRESERVED" if diagnosable and common else ("NOT_DIAGNOSABLE" if not diagnosable else "BROKEN")
        continuity[episode_id] = {
            "pre_loss_sample_index": required[0],
            "selected_loss_sample_indices": loss,
            "first_reappearance_sample_index": required[-1],
            "required_route_risk": route_risk,
            "continuous_route_risk": all(route_risk.values()),
            "parent_ancestry_diagnosable": diagnosable,
            "common_confirmed_parent_track_ids": sorted(common),
            "parent_ancestry_status": ancestry_status,
        }
    ambiguity = {
        "scored_ambiguity_frames": ambiguity_frames_total,
        "selected_contact_loss_ambiguity_frames": ambiguity_selected_contact_frames,
        "materially_exercised": ambiguity_selected_contact_frames > 0,
        "criterion": "EXISTS_CONFIRMED_ROUTE_RISK_ANCESTRY_WITH_MULTIPLE_X31_BRANCHES_ON_A_SELECTED_CONTACT_LOSS_FRAME",
        "numeric_score_threshold_added": False,
    }
    return continuity, ambiguity


def render_svg(result: Mapping[str, Any]) -> str:
    aggregate = result["aggregate"]
    rows = []
    for index, episode_id in enumerate(CONTACT_EPISODES):
        item = result["contacts"][episode_id][ARM_X31]
        rows.append(
            f'<text x="30" y="{160 + 26 * index}" font-size="15" fill="#111827">{html.escape(episode_id)} lead={c7.format_seconds(item["first_alert_lead_seconds"])}s recall={item["future_positive_recall"]:.3f} continuity={result["transport_continuity"][episode_id]["parent_ancestry_status"]}</text>'
        )
    decision = html.escape(str(result["decision"]))
    return "\n".join([
        '<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="340" viewBox="0 0 1120 340">',
        '<rect width="1120" height="340" fill="#f8fafc"/>',
        '<text x="30" y="45" font-size="25" font-family="sans-serif" fill="#0f172a">C8 X31 transport-cone pretruth Development score</text>',
        f'<text x="30" y="82" font-size="14" font-family="monospace" fill="#334155">{decision}</text>',
        f'<text x="30" y="116" font-size="17" fill="#111827">X31 precision={aggregate[ARM_X31]["precision"]:.3f} F1={aggregate[ARM_X31]["f1"]:.3f}; X24 F1={aggregate[ARM_X24]["f1"]:.3f}</text>',
        *rows,
        f'<text x="30" y="290" font-size="15" fill="#111827">ambiguity selected-loss frames={result["transport_ambiguity"]["selected_contact_loss_ambiguity_frames"]}; all 8 singleton occlusion contracts validated</text>',
        '<text x="30" y="320" font-size="13" fill="#475569">Scripted CARLA pretruth Development only; no deployment, real-world, product, or safety authority.</text>',
        '</svg>',
    ])


def score(args: argparse.Namespace) -> dict[str, Any]:
    protocol_path = args.protocol.resolve(strict=True)
    source_root = args.source_root.resolve(strict=True)
    run_root = args.run_root.resolve(strict=True)
    result_path = run_root / "result-x31.json"
    svg_path = run_root / "result-x31.svg"
    base.require(not result_path.exists() and not svg_path.exists(), "x31_score_outputs_exist")

    protocol = base.read_json(protocol_path)
    validate_protocol(protocol, protocol_path)
    x31_path = predictor_path().resolve(strict=True)
    validate_predictor_source(x31_path)
    x24 = base.read_json(run_root / "predictions-x24.json")
    x31 = base.read_json(run_root / "predictions-x31.json")
    freeze_x24 = base.read_json(run_root / "freeze-x24.json")
    freeze_x31 = base.read_json(run_root / "freeze-x31.json")
    validate_prediction_envelopes(x24, x31, freeze_x24, freeze_x31, run_root)

    # Evaluator-bearing files are opened only after all frozen identities above.
    source_result = base.read_json(source_root / "result.json")
    base.require(source_result.get("status") == SOURCE_COMPLETE_STATUS, "source_incomplete")
    base.require(bool(source_result.get("checks")) and all(bool(value) for value in source_result["checks"].values()), "source_gate_failed")
    base.require(source_result["protocol_sha256"] == EXPECTED_PROTOCOL_SHA256, "protocol_source_drift")
    base.require(int(source_result["episode_count"]) == len(EPISODES) and int(source_result["layout_count"]) == len(LAYOUTS), "source_cohort_count")
    model_manifest_path = source_root / "model" / "manifest.json"
    base.require(base.sha256_file(model_manifest_path) == freeze_x24["model_manifest"]["sha256"] == freeze_x31["source"]["model_manifest_sha256"], "source_model_manifest_drift")

    score_end = dict(SCORE_WINDOW_END_SECONDS)
    horizon_s = float(protocol["route_contract"]["future_horizon_seconds"])
    evaluator_full = {episode_id: base.read_jsonl(source_root / "evaluator" / "episodes" / episode_id / "frames.jsonl") for episode_id in EPISODES}
    predictions_full = {
        ARM_X24: {episode_id: c7.arm_frames_full(x24, episode_id, ARM_X24) for episode_id in EPISODES},
        ARM_X31: {episode_id: c7.arm_frames_full(x31, episode_id, ARM_X31) for episode_id in EPISODES},
    }
    for arm, episodes in predictions_full.items():
        for episode_id in EPISODES:
            base.align(evaluator_full[episode_id], episodes[episode_id], f"{arm}:{episode_id}")
    truth_tail_checks = {episode_id: float(rows[-1]["time_s"]) + EPSILON >= score_end[episode_id] + horizon_s for episode_id, rows in evaluator_full.items()}
    base.require(all(truth_tail_checks.values()), "right_censored_score_window")
    evaluator = {episode_id: c7.prefix(rows, score_end[episode_id]) for episode_id, rows in evaluator_full.items()}
    predictions = {arm: {episode_id: c7.prefix(rows, score_end[episode_id]) for episode_id, rows in episodes.items()} for arm, episodes in predictions_full.items()}
    aggregate = {arm: base.confusion(evaluator, episodes) for arm, episodes in predictions.items()}
    contacts = {episode_id: {arm: base.contact_metrics(evaluator[episode_id], episodes[episode_id]) for arm, episodes in predictions.items()} for episode_id in CONTACT_EPISODES}
    safe = {episode_id: {arm: base.false_segments(episodes[episode_id], SAFE_SEGMENT_START_SECONDS[episode_id]) for arm, episodes in predictions.items()} for episode_id in SAFE_EPISODES}
    occlusion_reports = base.read_json(source_root / "evaluator" / "physical_occlusion_report.json")
    selected_occlusions = validate_occlusion_reports(protocol, occlusion_reports, evaluator_full)
    continuity, ambiguity = contact_transport_continuity(x31, selected_occlusions)
    invariants = authority_invariants(x31, score_end)
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
        "schema": "blindassist-dtr-carla-c8-x31-source-disjoint-score-result-v1",
        "status": "COMPLETE",
        "decision": "DTR_CARLA_X31_AMBIGUITY_PRESERVING_TRANSPORT_SOURCE_DISJOINT_PRETRUTH_DEVELOPMENT_GATE_MET" if gate_met else "DTR_CARLA_X31_AMBIGUITY_PRESERVING_TRANSPORT_SOURCE_DISJOINT_PRETRUTH_DEVELOPMENT_GATE_NOT_MET",
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
        "deltas": {"frame_f1": aggregate[ARM_X31]["f1"] - aggregate[ARM_X24]["f1"], "frame_precision": aggregate[ARM_X31]["precision"] - aggregate[ARM_X24]["precision"], "frame_recall": aggregate[ARM_X31]["recall"] - aggregate[ARM_X24]["recall"], "safe_risk_segments": safe_segments},
        "source": {
            "source_result_sha256": base.sha256_file(source_root / "result.json"),
            "physical_occlusion_report_sha256": base.sha256_file(source_root / "evaluator" / "physical_occlusion_report.json"),
            "protocol_sha256": base.sha256_file(protocol_path),
            "x24_predictions_sha256": base.sha256_file(run_root / "predictions-x24.json"),
            "x31_predictions_sha256": base.sha256_file(run_root / "predictions-x31.json"),
            "x24_freeze_sha256": base.sha256_file(run_root / "freeze-x24.json"),
            "x31_freeze_sha256": base.sha256_file(run_root / "freeze-x31.json"),
            "x31_predictor_sha256": base.sha256_file(x31_path),
            "scorer_sha256": base.sha256_file(Path(__file__).resolve()),
        },
        "claim_boundary": {
            "fresh_scripted_carla_source_disjoint_pretruth_development": True,
            "same_map_numeric_anchor_and_witness_reuse": True,
            "source_disjoint_in_seed_weather_target_alias_occluder_trajectory_and_dense_support_pose": True,
            "source_disjoint_in_map_or_anchor_identity": False,
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
        parser.error(
            "--source-root and --run-root are required unless --validate-static is used"
        )
    return args


def main() -> int:
    args = parse_args()
    result = validate_static(args.protocol) if args.validate_static else score(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
